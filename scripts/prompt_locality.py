#!/usr/bin/env python3
"""Prompt Cache Analyzer — what does the difference between two prompts cost?

Uses the server's OWN tokenizer AND its own chat template (/tokenize in chat mode) so
block boundaries are exact.

  python3 prompt_locality.py a.json b.json   # two chat-completion bodies
  python3 prompt_locality.py a.txt b.txt     # two plain texts

Two modes:
  TOKEN MODE      exact first divergence, block index, theoretical reuse
  STRUCTURE MODE  (for JSON) which message/field moved or changed

If the server cannot tokenize `messages`/`tools`, this aborts. `--allow-approximate`
falls back to hand-rolled flattening with a warning — its absolute counts are unreliable
(measured 93 tokens against the template's 338 on one agent body with one tool).
"""
import json, os, sys, urllib.error, urllib.request

# REDACT=1 -> never print prompt content, only positions and fractions.
REDACT = os.environ.get("REDACT", "0") == "1"

BASE = os.environ.get("BASE", "http://127.0.0.1:8888")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-0731")
BLOCK = 256


# Measured prefill rate per context depth (DS4-0731, 2x DGX Spark, chunk 8192).
# Attention is O(n^2) -> rate drops with depth. A constant badly underestimates
# the cost at deep contexts. Recalibrate for other rigs.
PREFILL_CURVE = [(33089, 1810.0), (132041, 1758.0), (529151, 1308.0), (929733, 1017.0)]


def prefill_rate(n_tokens):
    """Linearly interpolated prefill rate at a given depth."""
    if n_tokens <= PREFILL_CURVE[0][0]:
        return PREFILL_CURVE[0][1]
    if n_tokens >= PREFILL_CURVE[-1][0]:
        return PREFILL_CURVE[-1][1]
    for (x0, y0), (x1, y1) in zip(PREFILL_CURVE, PREFILL_CURVE[1:]):
        if x0 <= n_tokens <= x1:
            return y0 + (y1 - y0) * (n_tokens - x0) / (x1 - x0)
    return PREFILL_CURVE[-1][1]


def tokenize_request(obj, approximate=False):
    """The body to POST to /tokenize.

    A chat body is sent as `messages` + `tools` so the SERVER applies its own chat
    template. Reconstructing the template by hand is not a small approximation: on an
    agent body with one tool definition the real template produced 338 tokens where the
    old hand-rolled flattening produced 93. See results/CORRECTION-2026-08-02.md.
    """
    if isinstance(obj, dict) and "messages" in obj and not approximate:
        req = {"model": MODEL, "messages": obj["messages"], "add_generation_prompt": True}
        if obj.get("tools"):
            req["tools"] = obj["tools"]
        return req
    return {"model": MODEL, "prompt": flatten(obj)}


def require_chat_tokenize(response):
    """Abort rather than silently falling back to an approximation."""
    if isinstance(response, dict) and response.get("__http"):
        sys.exit(
            "ABORT: the server rejected chat-mode /tokenize (HTTP %s: %s).\n"
            "       Exact block boundaries require the server's own chat template.\n"
            "       Re-run with --allow-approximate to use hand-rolled flattening,\n"
            "       and treat the absolute token counts as unreliable."
            % (response["__http"], str(response.get("__body"))[:200]))
    return response


def tokenize(obj, approximate=False):
    body = json.dumps(tokenize_request(obj, approximate)).encode()
    r = urllib.request.Request(BASE + "/tokenize", body, {"Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(r, timeout=120).read())
    except urllib.error.HTTPError as e:
        d = {"__http": e.code, "__body": e.read().decode("utf-8", "ignore")}
    if not approximate:
        require_chat_tokenize(d)
    return d.get("tokens") or d.get("token_ids") or []


def flatten(obj):
    """Chat body -> approximate prompt text. Only used with --allow-approximate.

    Keeps tool_calls: an assistant message may carry its entire payload there with
    content=None, and dropping it makes a divergence inside a tool call invisible.
    """
    if isinstance(obj, dict) and "messages" in obj:
        parts = []
        for m in obj["messages"]:
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
            piece = c or ""
            if m.get("tool_calls"):
                piece += json.dumps(m["tool_calls"], sort_keys=False)
            parts.append("<|%s|>%s" % (m.get("role", "?"), piece))
        if obj.get("tools"):
            parts.insert(0, "<|tools|>" + json.dumps(obj["tools"], sort_keys=False))
        return "\n".join(parts)
    return obj if isinstance(obj, str) else json.dumps(obj)


def load(path):
    raw = open(path, encoding="utf-8").read()
    try:
        return json.loads(raw)
    except Exception:
        return raw


def structure_diff(a, b):
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return []
    out = []
    ta, tb = json.dumps(a.get("tools"), sort_keys=True), json.dumps(b.get("tools"), sort_keys=True)
    if ta != tb:
        out.append("tools: definitions or their order differ")
    ma, mb = a.get("messages", []), b.get("messages", [])
    if len(ma) != len(mb):
        out.append("messages: count %d -> %d" % (len(ma), len(mb)))
    for i in range(min(len(ma), len(mb))):
        if ma[i] != mb[i]:
            ra = ma[i].get("role")
            ca, cb = str(ma[i].get("content")), str(mb[i].get("content"))
            j = next((k for k in range(min(len(ca), len(cb))) if ca[k] != cb[k]), min(len(ca), len(cb)))
            if REDACT:
                out.append("messages[%d] (%s): differs from character %d  [content hidden, REDACT=1]"
                           % (i, ra, j))
            else:
                out.append("messages[%d] (%s): differs from character %d  ...%s... -> ...%s..."
                           % (i, ra, j, ca[max(0, j - 25):j + 25].replace("\n", " "),
                              cb[max(0, j - 25):j + 25].replace("\n", " ")))
            if i < len(ma) - 1:
                out.append("  ^ this is NOT the last message -> everything after it is invalidated too")
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    approximate = "--allow-approximate" in sys.argv
    if len(args) < 2:
        print(__doc__)
        return
    A, B = load(args[0]), load(args[1])
    ta, tb = tokenize(A, approximate), tokenize(B, approximate)
    n = min(len(ta), len(tb))
    div = next((i for i in range(n) if ta[i] != tb[i]), n if len(ta) != len(tb) else -1)

    print("=" * 62)
    print("PROMPT CACHE ANALYZER")
    print("=" * 62)
    if approximate:
        print("TOKENIZATION MODE:         APPROXIMATE (hand-rolled flattening)")
        print("WARNING: absolute token counts and block indices are NOT reliable.")
        print("         The server's real chat template expands tool definitions far")
        print("         more than this flattening does. Use only when the server has")
        print("         no chat-mode /tokenize.")
    else:
        print("TOKENIZATION MODE:         SERVER CHAT TEMPLATE (EXACT)")
    print("Prompt A:                  %s tokens" % format(len(ta), ","))
    print("Prompt B:                  %s tokens" % format(len(tb), ","))
    if div < 0:
        print("\nIdentical -> 100 %% cache hit, 0 tokens re-prefilled.")
        return
    blk = div // BLOCK
    reuse = blk * BLOCK
    refill = len(tb) - reuse
    print("First divergence:          token %s" % format(div, ","))
    print("Block size:                %d" % BLOCK)
    print("First invalidated block:   %d of %d" % (blk, (len(tb) + BLOCK - 1) // BLOCK))
    print("Reusable prefix cache:     %s tokens (%.1f %%)" % (format(reuse, ","), 100.0 * reuse / len(tb)))
    print("Re-prefill:                %s tokens (%.1f %%)" % (format(refill, ","), 100.0 * refill / len(tb)))

    # --- time estimate: separate section, explicitly lower confidence ---
    # The cost is NOT proportional to the number of recomputed tokens. Attention
    # is O(n^2): a token at position i attends over its whole prefix. A suffix
    # from position p costs (n^2 - p^2)/2, i.e. the fraction 1 - (p/n)^2.
    # Verified against our own measurement: middle/top ratio measured 0.811;
    # the positional model gives 0.75-0.80, the naive token model 0.50.
    n = len(tb)
    frac = 1.0 - (reuse / float(n)) ** 2
    full = n / prefill_rate(n)
    est = full * frac
    print("\n--- TIME ESTIMATE (lower confidence than the cache figures above) ---")
    print("Positional share of full prefill: %.1f %%   [1 - (p/n)^2]" % (100 * frac))
    print("Estimated extra cost:             ~%.1f s" % est)
    print("Calibration:                      chunk 8192, DS4-0731 on 2x DGX Spark")
    print("WARNING: the time estimate is configuration-dependent and UNCALIBRATED")
    print("         for other runtime profiles. Measured deviation up to 1.7x when")
    print("         calibration came from a different MAX_NUM_BATCHED_TOKENS.")
    print("         The cache figures above, by contrast, are deterministic.")
    pos = 100.0 * div / len(tb)
    verdict = ("GOOD — the mutation sits at the end, nearly the whole cache is reused" if pos > 90 else
               "BAD — the mutation sits early, nearly everything must be recomputed" if pos < 25 else
               "MEDIUM — the mutation sits mid-prompt, half the context is lost")
    print("\nMutation position:         %.1f %% into the prompt\n%s" % (pos, verdict))
    sd = structure_diff(A, B)
    if sd:
        print("\n--- STRUCTURE MODE: likely cause ---")
        for s in sd:
            print("  " + s)
    print("\nRule: static first (system prompt, tools, stable history),")
    print("       dynamic last (timestamps, status, fresh tool results).")


if __name__ == "__main__":
    main()
