"""Prefix Cache Locality — what does a mutating token cost, depending on WHERE it sits?

Simulates an agent loop: the same context is sent repeatedly, growing ~200 tokens
per turn. Four variants differ only in WHERE a mutating field is placed.

  clean         nothing mutates             -> cache hits fully
  dirty-bottom  mutates at the end          -> only the tail is recomputed
  dirty-middle  mutates mid-prompt          -> half the context is recomputed
  dirty-top     mutates at the top          -> EVERYTHING is recomputed, every turn

TTFT is the metric: it is proportional to how much must be re-prefilled.
vLLM caches in blocks (--block-size, 256 here), so invalidation starts at the
block containing the mutation.

Env: MODEL, PORT, MULT (size of the static context; lower it if the model's
context window is small — tokenizers differ, the same text can exceed the cap).
"""
import json, time, urllib.request

import os
PORT = os.environ.get("PORT", "8888")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-0731")
U = "http://127.0.0.1:%s/v1/chat/completions" % PORT
BLOCK = ("History entry: the agent read the file and noted the configuration looks correct. "
         "No deviations reported. The tool returned exit status zero. ")
TURNS = 8


def ttft(prompt):
    p = {"model": MODEL,
         "messages": [{"role": "user", "content": prompt}],
         "max_tokens": 16, "temperature": 0, "stream": True}
    r = urllib.request.Request(U, json.dumps(p).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    ptok = None
    for raw in urllib.request.urlopen(r, timeout=900):
        line = raw.decode("utf-8", "ignore").strip()
        if line.startswith("data:") and "[DONE]" not in line:
            try:
                d = json.loads(line[5:])
                if d["choices"][0].get("delta", {}).get("content"):
                    return time.time() - t0
            except Exception:
                pass
    return time.time() - t0


def build(variant, turn, salt):
    """~20K tokens static base + growing history."""
    static = "SYSTEM INSTRUCTION: you are an operations assistant. " + BLOCK * 700   # ~20K tok
    hist = BLOCK * (turn * 12)                                              # +~200 tok/varv
    mut = "CURRENT TIME: 2026-08-01 %02d:%02d:%02d. " % (22, turn, salt % 60)
    if variant == "clean":
        return static + hist + "\n\nReply with the word OK."
    if variant == "dirty-top":
        return mut + static + hist + "\n\nReply with the word OK."
    if variant == "dirty-middle":
        half = len(static) // 2
        return static[:half] + mut + static[half:] + hist + "\n\nReply with the word OK."
    return static + hist + mut + "\n\nReply with the word OK."          # dirty-bottom


print("modell: %s  port: %s" % (MODEL, PORT), flush=True)
print("warming the cache...", flush=True)
for v in ("clean", "dirty-top", "dirty-middle", "dirty-bottom"):
    build(v, 0, 0)
ttft(build("clean", 0, 0))

print("\nvariant         turn-1 TTFT   turns 2-8 median   mean 2-8   vs clean", flush=True)
base = None
for v in ("clean", "dirty-bottom", "dirty-middle", "dirty-top"):
    ts = []
    for turn in range(TURNS):
        ts.append(ttft(build(v, turn, turn)))
    warm = sorted(ts[1:])
    med = warm[len(warm) // 2]
    avg = sum(ts[1:]) / len(ts[1:])
    if base is None:
        base = avg
    print("%-14s  %8.2f s     %8.2f s      %8.2f s   %5.1fx"
          % (v, ts[0], med, avg, avg / base), flush=True)
