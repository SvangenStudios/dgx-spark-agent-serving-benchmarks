"""Pure concurrency test WITHOUT a giant prefill.

Question: does a larger KV pool actually buy practical concurrency, or does
the scheduler prevent us from using it?

Runs N parallel sessions with normal agent context (~32K) and measures
per-stream decode + aggregate. If the aggregate scales with N, concurrency
works. If per-stream falls proportionally, we are serialized.

Note: with a 32K prompt and ~150 generated tokens this workload is
PREFILL-dominated — it measures prefill concurrency, not decode concurrency.
Use decode_concurrency.py for the latter.
"""
import json, time, urllib.request, threading

U = "http://127.0.0.1:8888/v1/chat/completions"
FILL = ("Driftanteckning: rutinkontroll av kylsystem och natverkslankar genomford utan anmarkning. "
        "Loggrotation verifierad. Backup slutford enligt schema. Inga larm registrerade under passet. ")


def ctx(n_tokens, salt):
    chunk = FILL * 8
    per = int(len(chunk) / 3.6)
    n = max(4, n_tokens // per)
    return ("Session " + salt + ". ") + chunk * n + "\n\nWrite a short five-sentence summary."


def run(salt, out, n_ctx):
    p = {"model": "deepseek-v4-flash-0731",
         "messages": [{"role": "user", "content": ctx(n_ctx, salt)}],
         "max_tokens": 250, "temperature": 0, "stream": False}
    r = urllib.request.Request(U, json.dumps(p).encode(), {"Content-Type": "application/json"})
    t = time.time()
    try:
        d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
        e = time.time() - t
        out.append((d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"], e))
    except Exception as ex:
        out.append(("FEL", str(ex)[:50], 0))


# uppvarmning
print("warming up...", flush=True)
for i in range(2):
    run("warm%d" % i, [], 8000)

CTX = 32000
print("\nconcurrency   per-stream e2e     aggregate  worst latency   prompt_tok", flush=True)
for N in (1, 2, 4, 6):
    res = []
    ths = [threading.Thread(target=run, args=("c%d-%d" % (N, i), res, CTX)) for i in range(N)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    ok = [r for r in res if r[0] != "FEL"]
    if not ok:
        print("%6d        ALL FAILED: %s" % (N, res[0][1]), flush=True)
        continue
    per = sum(r[0] / r[2] for r in ok) / len(ok)
    agg = sum(r[0] for r in ok) / wall
    worst = max(r[2] for r in ok)
    print("%6d        %8.1f tok/s   %7.1f    %8.1f s     %s"
          % (N, per, agg, worst, format(ok[0][1], ",")), flush=True)
    time.sleep(5)
