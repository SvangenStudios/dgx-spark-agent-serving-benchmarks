#!/usr/bin/env python3
"""Does this server report streamed token IDs — and how many tokens per SSE chunk?

  python3 probe_token_ids.py [base_url] [model]

Answers the question that invalidated our first overlap measurement: on a speculative
stack, one SSE chunk carries several accepted tokens, so counting chunks is not counting
tokens. Run this before trusting any streamed throughput number from a new server.

Exit code 0 = token_ids reported and consistent with the server's own usage.
Exit code 1 = do not measure throughput from this server's stream.
"""
import collections
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4-flash-0731"
PROMPT = "Write a detailed paragraph about tidal patterns in shallow coastal waters."


def probe(return_token_ids):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 200, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    if return_token_ids:
        body["return_token_ids"] = True
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    events = 0
    tokens = 0
    hist = collections.Counter()
    empty_with_tokens = 0
    null_with_text = 0
    usage = None
    for raw in urllib.request.urlopen(req, timeout=300):
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:") or "[DONE]" in line:
            continue
        d = json.loads(line[5:])
        if d.get("usage"):
            usage = d["usage"]
        if not d.get("choices"):
            continue
        c = d["choices"][0]
        text = c.get("delta", {}).get("content")
        ids = c.get("token_ids")
        if text:
            events += 1
        if ids:
            tokens += len(ids)
            hist[len(ids)] += 1
            if not text:
                empty_with_tokens += 1
        elif text:
            null_with_text += 1
    return events, tokens, hist, empty_with_tokens, null_with_text, usage


def main():
    print("server: %s   model: %s" % (BASE, MODEL))
    try:
        ev, tok, hist, empty, null_text, usage = probe(return_token_ids=True)
    except urllib.error.HTTPError as e:
        print("FAIL: the server rejected return_token_ids (HTTP %s): %s"
              % (e.code, e.read().decode("utf-8", "ignore")[:200]))
        return 1

    reported = (usage or {}).get("completion_tokens")
    print("  non-empty content events : %d   <- what a naive stream counter would report" % ev)
    print("  tokens via token_ids     : %d" % tok)
    print("  usage.completion_tokens  : %s" % reported)
    print("  tokens per event         : %.2f" % (tok / ev) if ev else "  tokens per event: n/a")
    print("  tokens per chunk         : %s" % dict(sorted(hist.items())))
    print("  chunks with tokens but no visible text: %d" % empty)
    print("  chunks with visible text but no token_ids: %d" % null_text)

    if not tok:
        print("\nVERDICT: no token_ids returned. Do NOT measure throughput from this stream.")
        return 1
    if null_text:
        print("\nVERDICT: some visible output carried no token_ids — counts would be low.")
        return 1
    if reported is not None and tok != reported:
        print("\nVERDICT: counted %d but usage reports %d. Counts are not trustworthy."
              % (tok, reported))
        return 1
    ratio = tok / ev if ev else 1.0
    print("\nVERDICT: token_ids reported and consistent with usage.")
    if ratio > 1.05:
        print("         Counting SSE events would understate decode by %.2fx on this server."
              % ratio)
    else:
        print("         One token per chunk here, but do not assume it stays that way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
