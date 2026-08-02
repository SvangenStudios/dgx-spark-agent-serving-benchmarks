"""Overlap test: what happens to an ongoing decode and to new requests during a large prefill.

  python3 prefill_decode_overlap.py <prefill_tokens> <label>

Reports, for the window in which a large prefill is actually running:
  - token-weighted decode throughput of an ALREADY RUNNING streaming generation
  - p50 / p95 / max output-chunk gaps (user-visible jitter — a separate metric)
  - TTFT for a NEW short request arriving mid-prefill
  - prefill tok/s and the long job's total time

Measurement rules this script enforces (see results/CORRECTION-2026-08-02.md — an earlier
version of this file violated all three and published invalid numbers):

  1. Tokens are counted from the server's `token_ids`, never from SSE event counts. On a
     speculative-decoding stack one chunk carries several accepted tokens (~2.5 here).
     If the server does not return token_ids, this script ABORTS. There is no fallback.
  2. The prefill window is explicit: from submission of the long request to the arrival of
     its first output token. Output outside that window is not "during prefill".
  3. The reference is warm, streamed, and measured first token -> last token, so it
     contains no TTFT and no cold-clock ramp.
"""
import json
import sys
import threading
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:8888"
U = BASE + "/v1/chat/completions"
MODEL = "deepseek-v4-flash-0731"

FILL = ("Driftanteckning: rutinkontroll av kylsystem och natverkslankar genomford utan anmarkning. "
        "Loggrotation verifierad. Backup slutford enligt schema. Inga larm registrerade under passet. ")


class MissingTokenIds(Exception):
    """The server returned visible output without token_ids — counting events would lie."""


class TokenCountMismatch(Exception):
    """Our token count disagrees with the server's own usage report."""


# --- pure measurement helpers (unit-tested in tests/test_overlap_measurement.py) ---

def chunk_token_count(chunk):
    """Number of accepted tokens a chat.completion.chunk carries.

    A chunk may carry several tokens, and may carry tokens with no visible text delta.
    Counting chunks instead of tokens understates decode by the speculation factor.
    """
    choices = chunk.get("choices") or []
    if not choices:
        return 0
    ids = choices[0].get("token_ids")
    if ids is not None:
        return len(ids)
    # A stream ends with a finish_reason chunk that carries no content and no token_ids.
    # That is legitimate and carries zero tokens. Visible text without token_ids is not:
    # it means the server is not reporting them and we would be counting events.
    if choices[0].get("delta", {}).get("content"):
        raise MissingTokenIds(
            "server returned visible output without token_ids; rerun with "
            '"return_token_ids": true, or use a server that supports it')
    return 0


def verify_against_usage(samples, usage):
    """Cross-check our token count against the server's own completion_tokens.

    This is the guard that would have caught the original event-counting bug: an event
    count never matches usage on a speculative stack. A written rule is not a guard.
    """
    if not usage or usage.get("completion_tokens") is None:
        return
    counted = sum(n for _, n in samples)
    reported = usage["completion_tokens"]
    if counted != reported:
        raise TokenCountMismatch(
            "counted %d tokens but the server reports completion_tokens=%d"
            % (counted, reported))


def tokens_in_window(samples, start, end):
    """Tokens from (timestamp, count) samples whose timestamp lies within [start, end]."""
    return sum(n for t, n in samples if start <= t <= end)


def decode_rate_in_window(samples, start, end):
    """Token-weighted decode throughput inside an explicit window."""
    if end <= start:
        return 0.0
    return tokens_in_window(samples, start, end) / (end - start)


def reference_decode_rate(samples):
    """Undisturbed decode rate, first token to last token — excludes TTFT.

    The first sample's tokens are excluded: they mark the start of the interval rather
    than being produced during it.
    """
    if len(samples) < 2:
        return 0.0
    span = samples[-1][0] - samples[0][0]
    if span <= 0:
        return 0.0
    return sum(n for _, n in samples[1:]) / span


def chunk_gaps(samples):
    """Wall-clock gaps between output chunks — user-visible jitter, not throughput."""
    return [samples[i + 1][0] - samples[i][0] for i in range(len(samples) - 1)]


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


# --- wire helpers ---

def body(content, max_tokens):
    return {"model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True}}


def stream_samples(content, max_tokens, samples=None, on_first_token=None, timeout=2400):
    """Stream a generation, recording (monotonic_timestamp, token_count) per chunk.

    Returns (samples, usage). Raises MissingTokenIds if the server does not report them.
    """
    samples = samples if samples is not None else []
    local = []
    usage = None
    req = urllib.request.Request(U, json.dumps(body(content, max_tokens)).encode(),
                                 {"Content-Type": "application/json"})
    first = True
    for raw in urllib.request.urlopen(req, timeout=timeout):
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:") or "[DONE]" in line:
            continue
        d = json.loads(line[5:])
        if d.get("usage"):
            usage = d["usage"]
        if not d.get("choices"):
            continue
        n = chunk_token_count(d)
        if not n:
            continue
        now = time.monotonic()
        if first and on_first_token:
            on_first_token(now)
            first = False
        samples.append((now, n))
        local.append((now, n))
    verify_against_usage(local, usage)
    return samples, usage


def salted(text):
    """Every measurement prompt gets a unique prefix.

    Not just the big prefill: an unsalted reference, ongoing or short-request prompt hits
    the prefix cache on the second repetition and measures the cache instead of the model.
    This build exposes no prefix-cache reset endpoint (`/reset_prefix_cache` returns 404),
    so salting is the only reset available short of restarting the server.
    """
    return "Session %s. %s" % (uuid.uuid4().hex[:12], text)


def build(n_tokens, salt):
    chunk = FILL * 8
    per = int(len(chunk) / 3.6)
    n = max(4, n_tokens // per)
    # the salt breaks the prefix cache so every measurement is fresh
    return ("Session %s-%s. " % (salt, uuid.uuid4().hex[:12])) + chunk * n + \
           "\n\nSummarize in one sentence."


def assert_token_ids_supported():
    """Fail loudly before measuring, rather than silently producing event rates."""
    try:
        samples, _ = stream_samples(salted("Reply with one word: hi"), 4, timeout=120)
    except MissingTokenIds as e:
        sys.exit("ABORT: %s" % e)
    if not samples:
        sys.exit("ABORT: server returned no tokens during the capability probe")
    print("  capability probe: token_ids returned OK", flush=True)


def run(target, label):
    print("CONFIG: %s   prefill=%dK" % (label, target // 1000), flush=True)
    assert_token_ids_supported()

    # --- warm-up: GB10 clocks ramp under load; a cold reference reads ~18% low ---
    print("  warming up (long generation)...", flush=True)
    stream_samples(salted("Write a long, detailed text about ocean currents."), 400)

    # --- reference: warm, streamed, first token -> last token ---
    ref_samples, _ = stream_samples(
        salted("Write a long, detailed text about coastal geology."), 400)
    ref_tps = reference_decode_rate(ref_samples)
    print("  reference decode (warm, undisturbed): %.2f tok/s  (%d tokens in %d chunks)"
          % (ref_tps, sum(n for _, n in ref_samples), len(ref_samples)), flush=True)

    stream = []
    window = {}
    short_ttft = []
    long_res = []

    def ongoing():
        stream_samples(
            salted("Write a long, detailed text about Baltic Sea ecology and hydrography."),
            2000, samples=stream)

    def long_call():
        window["submitted"] = time.monotonic()
        _, usage = stream_samples(build(target, label), 20,
                                  on_first_token=lambda t: window.__setitem__("first_token", t))
        long_res.append((usage or {}).get("prompt_tokens", 0))

    def short_call():
        t0 = time.monotonic()
        stream_samples(salted("Reply with one word: hi"), 5,
                       on_first_token=lambda t: short_ttft.append(t - t0))

    ta = threading.Thread(target=ongoing)
    ta.start()
    time.sleep(5)
    tb = threading.Thread(target=long_call)
    tb.start()
    time.sleep(20)
    ts = threading.Thread(target=short_call)
    ts.start()
    ta.join(); ts.join(); tb.join()

    start, end = window.get("submitted"), window.get("first_token")
    if start is None or end is None:
        sys.exit("ABORT: the long request never produced a first token; no prefill window")

    # --- metric 1: token-weighted throughput inside the prefill window ---
    n_tok = tokens_in_window(stream, start, end)
    tps_during = decode_rate_in_window(stream, start, end)
    share = 100 * tps_during / ref_tps if ref_tps else 0
    print("  prefill window:           %.1f s  (submit -> first output token)" % (end - start),
          flush=True)
    print("  decode DURING prefill:    %.2f tok/s  (%.1f %% of reference; %d tokens counted)"
          % (tps_during, share, n_tok), flush=True)

    # --- metric 2: jitter, reported separately and never mixed with throughput ---
    gaps = chunk_gaps([s for s in stream if start <= s[0] <= end])
    print("  output-chunk gap p50/p95/max: %.3f s / %.3f s / %.3f s"
          % (pct(gaps, 0.50), pct(gaps, 0.95), max(gaps) if gaps else 0), flush=True)

    print("  TTFT new short request:   %s"
          % ("%.1f s" % short_ttft[0] if short_ttft else "-"), flush=True)
    if long_res and long_res[0]:
        dur = end - start
        print("  long job: %s tok prefilled in %.1f s  (%.0f tok/s prefill)"
              % (format(long_res[0], ","), dur, long_res[0] / dur), flush=True)

    verdict = "GREEN" if share >= 25 else "RED"
    print("  >>> %s  (requirement: >=25 %% of reference decode)" % verdict, flush=True)


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 256000
    label = sys.argv[2] if len(sys.argv) > 2 else "?"
    run(target, label)


if __name__ == "__main__":
    main()
