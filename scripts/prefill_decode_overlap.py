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
  3. The baseline is the SAME stream, measured before the prefill was submitted. One
     request compared against itself carries no difference in prompt content, draft
     acceptance, sequence length, clock state or incidental load. A separate warm
     reference is still measured, but only as a diagnostic.
  4. The ongoing stream must still be producing when the window closes, or the divisor
     covers time in which no token could have arrived. The script aborts if it does not.
  5. Output-chunk gaps are clipped to the window rather than computed from samples
     filtered to it — otherwise the stall at the moment the prefill starts, typically the
     largest visible pause, is dropped because it straddles the boundary.
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

# The ongoing stream must outlive the prefill window. A 256K prefill runs for minutes; if
# decode is NOT starved it would burn ~30 tok/s the whole time, so the ceiling has to cover
# the undisturbed case, not the starved one. The script aborts rather than silently
# dividing by a window the stream did not survive.
ONGOING_MAX_TOKENS = 8000

# How long the ongoing stream runs undisturbed before the prefill is submitted. This is the
# stream's own baseline, so it needs enough tokens to be stable — 5 s is not enough.
UNDISTURBED_SECONDS = 25

# When the new short request is sent, relative to submission of the long request.
SHORT_REQUEST_DELAY = 20

FILL = ("Driftanteckning: rutinkontroll av kylsystem och natverkslankar genomford utan anmarkning. "
        "Loggrotation verifierad. Backup slutford enligt schema. Inga larm registrerade under passet. ")


class MissingTokenIds(Exception):
    """The server returned visible output without token_ids — counting events would lie."""


class TokenCountMismatch(Exception):
    """Our token count disagrees with the server's own usage report."""


class StreamEndedEarly(Exception):
    """The ongoing generation finished before the prefill window closed."""


class UsageMissing(Exception):
    """The server did not report usage, so the token count cannot be cross-checked."""


def guarded(fn, errors):
    """Wrap a thread target so its exception is recorded instead of discarded.

    Every integrity guard in this file runs inside a worker thread, and
    threading.Thread swallows exceptions from its target. Without this the guards
    would be decorative: a TokenCountMismatch in the ongoing stream would vanish and
    the run would report a number computed from partial data.
    """
    def wrapped():
        try:
            fn()
        except BaseException as e:            # noqa: BLE001 - deliberately broad
            errors.append(e)
    return wrapped


def admission_is_measurable(submit_t, window_start, window_end):
    """Was the new short request actually submitted while the prefill was running?

    Judged from the real timestamp, not from the delay we intended: thread scheduling
    decides when the request truly lands.
    """
    return window_start <= submit_t <= window_end


def stream_margin(samples, window_end):
    """How long the ongoing stream kept producing after the window closed."""
    if not samples:
        return 0.0
    return samples[-1][0] - window_end


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
        raise UsageMissing(
            "server did not report usage.completion_tokens; the token count cannot be "
            "cross-checked, and an unverified count is what produced the original bug. "
            'Ensure "stream_options": {"include_usage": true} is accepted.')
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


def gaps_overlapping_window(samples, start, end):
    """Output-chunk gaps clipped to a window.

    Filtering samples to the window BEFORE computing gaps silently drops the interval
    that straddles the boundary — which is the stall that occurs when the prefill starts,
    often the largest visible pause of the whole run. Instead, every adjacent interval is
    considered and its overlap with the window is taken.
    """
    out = []
    for i in range(len(samples) - 1):
        overlap = min(samples[i + 1][0], end) - max(samples[i][0], start)
        if overlap > 0:
            out.append(overlap)
    return out


def decode_share(samples, baseline_start, window_start, window_end):
    """Decode rate inside the prefill window as a fraction of the SAME stream's rate
    before it.

    Using one request as its own baseline removes every difference a separate reference
    request carries: prompt content, draft acceptance, sequence length, clock state and
    incidental server load.
    """
    baseline = decode_rate_in_window(samples, baseline_start, window_start)
    during = decode_rate_in_window(samples, window_start, window_end)
    return during / baseline if baseline else 0.0


def assert_stream_outlived_window(samples, window_end):
    """The ongoing generation must still be producing when the prefill window closes.

    If it hits max_tokens mid-prefill, tokens stop accruing while the window keeps
    running, and the divisor makes decode read artificially low.
    """
    if not samples or samples[-1][0] < window_end:
        last = samples[-1][0] if samples else None
        raise StreamEndedEarly(
            "ongoing decode ended before the prefill window closed "
            "(last token at %s, window closes at %.1f); raise max_tokens"
            % ("%.1f" % last if last is not None else "never", window_end))


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
            ONGOING_MAX_TOKENS, samples=stream,
            on_first_token=lambda t: window.__setitem__("stream_first", t))

    def long_call():
        window["submitted"] = time.monotonic()
        _, usage = stream_samples(build(target, label), 20,
                                  on_first_token=lambda t: window.__setitem__("first_token", t))
        long_res.append((usage or {}).get("prompt_tokens", 0))

    def short_call():
        t0 = time.monotonic()
        window["short_submitted"] = t0
        stream_samples(salted("Reply with one word: hi"), 5,
                       on_first_token=lambda t: short_ttft.append(t - t0))

    errors = []
    ta = threading.Thread(target=guarded(ongoing, errors))
    ta.start()
    time.sleep(UNDISTURBED_SECONDS)   # the stream's own baseline is measured here
    tb = threading.Thread(target=guarded(long_call, errors))
    tb.start()
    time.sleep(SHORT_REQUEST_DELAY)
    ts = threading.Thread(target=guarded(short_call, errors))
    ts.start()
    ta.join(); ts.join(); tb.join()

    if errors:
        for e in errors:
            print("  WORKER ERROR: %s: %s" % (type(e).__name__, e), flush=True)
        sys.exit("ABORT: a measurement thread failed; the run is not trustworthy")

    start, end = window.get("submitted"), window.get("first_token")
    base_start = window.get("stream_first")
    if start is None or end is None:
        sys.exit("ABORT: the long request never produced a first token; no prefill window")
    if base_start is None:
        sys.exit("ABORT: the ongoing stream never produced a token; no baseline")

    # The ongoing stream must still be running when the window closes, or the divisor
    # covers time in which no tokens could have arrived.
    try:
        assert_stream_outlived_window(stream, end)
    except StreamEndedEarly as e:
        sys.exit("ABORT: %s (currently ONGOING_MAX_TOKENS=%d)" % (e, ONGOING_MAX_TOKENS))
    margin = stream_margin(stream, end)
    if margin < 10.0:
        print("  WARNING: ongoing stream outlived the window by only %.1f s — raise "
              "ONGOING_MAX_TOKENS (currently %d)" % (margin, ONGOING_MAX_TOKENS), flush=True)

    # --- metric 1: the stream measured against ITSELF, before vs during ---
    n_tok = tokens_in_window(stream, start, end)
    baseline_tps = decode_rate_in_window(stream, base_start, start)
    tps_during = decode_rate_in_window(stream, start, end)
    share = 100 * decode_share(stream, base_start, start, end)
    print("  undisturbed baseline:     %.2f tok/s  (same stream, %.1f s before submit)"
          % (baseline_tps, start - base_start), flush=True)
    print("  prefill window:           %.1f s  (submit -> first output token)" % (end - start),
          flush=True)
    print("  decode DURING prefill:    %.2f tok/s  (%.1f %% of its own baseline; %d tokens)"
          % (tps_during, share, n_tok), flush=True)
    print("  [diagnostic] separate warm reference: %.2f tok/s  (%.1f %% of that)"
          % (ref_tps, 100 * tps_during / ref_tps if ref_tps else 0), flush=True)

    # --- metric 2: jitter, reported separately and never mixed with throughput ---
    gaps = gaps_overlapping_window(stream, start, end)
    print("  output-chunk gap p50/p95/max: %.3f s / %.3f s / %.3f s"
          % (pct(gaps, 0.50), pct(gaps, 0.95), max(gaps) if gaps else 0), flush=True)

    admission = "%.1f s" % short_ttft[0] if short_ttft else "-"
    short_t = window.get("short_submitted")
    if short_ttft and short_t is not None and not admission_is_measurable(short_t, start, end):
        admission += ("  (NOT an admission result: submitted %.1f s into a %.1f s window)"
                      % (short_t - start, end - start))
    elif short_ttft and short_t is None:
        admission += "  (submission time unknown — not usable as an admission result)"
    print("  TTFT new short request:   %s" % admission, flush=True)
    print("  stream margin past window: %.1f s" % margin, flush=True)
    if long_res and long_res[0]:
        dur = end - start
        print("  long job: %s tok prefilled in %.1f s  (%.0f tok/s prefill)"
              % (format(long_res[0], ","), dur, long_res[0] / dur), flush=True)

    verdict = "GREEN" if share >= 25 else "RED"
    print("  >>> %s  (requirement: >=25 %% of its own undisturbed baseline)" % verdict,
          flush=True)


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 256000
    label = sys.argv[2] if len(sys.argv) > 2 else "?"
    run(target, label)


if __name__ == "__main__":
    main()
