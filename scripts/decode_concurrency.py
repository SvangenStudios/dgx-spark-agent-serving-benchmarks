"""Decode concurrency, isolated from prefill.

Short unique prompt (~100 tok) + long generation (1200 tok) -> decode dominates.
Per N: aggregate decode, per-stream decode, TTFT per stream, p50/p95 token
intervals, actual token counts, and DRAFT ACCEPTANCE per level.

Note: generation content language strongly affects draft acceptance (we measured
71 % on code vs 19.5 % on Swedish prose on the same stack). The topics below are
deliberately non-English as a worst case; edit TOPICS for your locale.
"""
import json, re, time, urllib.request, threading

U = "http://127.0.0.1:8888/v1/chat/completions"
M = "http://127.0.0.1:8888/metrics"

TOPICS = ["Ostersjons hydrografi", "kompilatorers optimeringssteg", "vattenkraftens reglering",
          "distribuerade konsensusprotokoll", "skogsbrukets historia i Norrland",
          "minneshierarkier i moderna processorer", "fjarrvarmenatens dimensionering",
          "kryptografiska hashfunktioner"]


def counters():
    t = urllib.request.urlopen(M, timeout=30).read().decode()
    a = d = 0
    for l in t.splitlines():
        if l.startswith("#"):
            continue
        m = re.match(r"vllm:spec_decode_num_accepted_tokens_total\{[^}]*\} ([0-9.e+]+)", l)
        if m:
            a += float(m.group(1))
        m = re.match(r"vllm:spec_decode_num_draft_tokens_total\{[^}]*\} ([0-9.e+]+)", l)
        if m:
            d += float(m.group(1))
    return a, d


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p))]


def stream(idx, out):
    topic = TOPICS[idx % len(TOPICS)]
    prompt = ("Session %d. Write a very thorough technical text about %s. "
              "Be concrete and detailed, at least a thousand words." % (idx, topic))
    p = {"model": "deepseek-v4-flash-0731",
         "messages": [{"role": "user", "content": prompt}],
         "max_tokens": 1200, "temperature": 0, "stream": True}
    r = urllib.request.Request(U, json.dumps(p).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    ts = []
    for raw in urllib.request.urlopen(r, timeout=1800):
        line = raw.decode("utf-8", "ignore").strip()
        if line.startswith("data:") and "[DONE]" not in line:
            try:
                d = json.loads(line[5:])
                if d["choices"][0].get("delta", {}).get("content"):
                    ts.append(time.time() - t0)
            except Exception:
                pass
    if len(ts) >= 2:
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        out.append({"ttft": ts[0], "n": len(ts), "wall": ts[-1],
                    "tps": (len(ts) - 1) / (ts[-1] - ts[0]), "gaps": gaps})


print("warming up (3 long generations)...", flush=True)
w = []
for i in range(3):
    stream(90 + i, w)
if w:
    print("  warmed decode: %.1f tok/s" % w[-1]["tps"], flush=True)

print("\n  N   aggregate  per-stream  TTFT min/max      p50/p95 gap     tokens   acceptance", flush=True)
for N in (1, 2, 4, 6):
    res = []
    a0, d0 = counters()
    ths = [threading.Thread(target=stream, args=(N * 10 + i, res)) for i in range(N)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    a1, d1 = counters()
    if not res:
        print("%3d   NO RESPONSES" % N, flush=True)
        continue
    agg = sum(r["n"] for r in res) / wall
    per = sum(r["tps"] for r in res) / len(res)
    allg = [g for r in res for g in r["gaps"]]
    acc = 100 * (a1 - a0) / (d1 - d0) if d1 > d0 else 0
    print("%3d   %6.1f     %6.1f      %.1f / %.1f s     %.3f / %.3f s   %5d    %4.1f %%"
          % (N, agg, per, min(r["ttft"] for r in res), max(r["ttft"] for r in res),
             pct(allg, 0.50), pct(allg, 0.95), sum(r["n"] for r in res), acc), flush=True)
    time.sleep(4)
