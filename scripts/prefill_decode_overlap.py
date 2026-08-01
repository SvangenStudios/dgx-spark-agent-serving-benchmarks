"""Reusable overlap test. Usage: python3 prefill_decode_overlap.py <prefill_tokens> <label>

Measures, during a large ongoing prefill:
  - decode tok/s for an ALREADY RUNNING streaming generation
  - p50 / p95 token intervals (catches PARTIAL starvation that max-gap misses)
  - TTFT for a NEW short request arriving mid-prefill
  - prefill tok/s and the long job's total time
"""
import json, sys, time, urllib.request, threading

U = "http://127.0.0.1:8888/v1/chat/completions"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 256000
LABEL = sys.argv[2] if len(sys.argv) > 2 else "?"

FILL = ("Driftanteckning: rutinkontroll av kylsystem och natverkslankar genomford utan anmarkning. "
        "Loggrotation verifierad. Backup slutford enligt schema. Inga larm registrerade under passet. ")


def build(n_tokens, salt):
    chunk = FILL * 8
    per = int(len(chunk) / 3.6)
    n = max(4, n_tokens // per)
    # the salt breaks the prefix cache so every measurement is fresh
    return ("Session " + salt + ". ") + chunk * n + "\n\nSummarize in one sentence."


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p))]


stream_ts = []
short_ttft = []
long_res = []


def stream_gen():
    p = {"model": "deepseek-v4-flash-0731",
         "messages": [{"role": "user", "content": "Write a long, detailed text about Baltic Sea ecology and hydrography."}],
         "max_tokens": 600, "temperature": 0, "stream": True}
    r = urllib.request.Request(U, json.dumps(p).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    for raw in urllib.request.urlopen(r, timeout=2400):
        line = raw.decode("utf-8", "ignore").strip()
        if line.startswith("data:") and "[DONE]" not in line:
            try:
                d = json.loads(line[5:])
                if d["choices"][0].get("delta", {}).get("content"):
                    stream_ts.append(time.time() - t0)
            except Exception:
                pass


def short_call():
    p = {"model": "deepseek-v4-flash-0731",
         "messages": [{"role": "user", "content": "Reply with one word: hi"}],
         "max_tokens": 5, "temperature": 0, "stream": True}
    r = urllib.request.Request(U, json.dumps(p).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    for raw in urllib.request.urlopen(r, timeout=2400):
        line = raw.decode("utf-8", "ignore").strip()
        if line.startswith("data:") and "[DONE]" not in line:
            try:
                d = json.loads(line[5:])
                if d["choices"][0].get("delta", {}).get("content"):
                    short_ttft.append(time.time() - t0)
                    return
            except Exception:
                pass


def long_call():
    p = {"model": "deepseek-v4-flash-0731",
         "messages": [{"role": "user", "content": build(TARGET, LABEL)}],
         "max_tokens": 20, "temperature": 0, "stream": False}
    r = urllib.request.Request(U, json.dumps(p).encode(), {"Content-Type": "application/json"})
    t = time.time()
    d = json.loads(urllib.request.urlopen(r, timeout=2400).read())
    long_res.append((d["usage"]["prompt_tokens"], time.time() - t))


# --- 0. reference: decode without interference ---
ref = []
t0 = time.time()
p = {"model": "deepseek-v4-flash-0731",
     "messages": [{"role": "user", "content": "Write a short text about the sea."}],
     "max_tokens": 120, "temperature": 0, "stream": False}
r = urllib.request.Request(U, json.dumps(p).encode(), {"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(r, timeout=300).read())
ref_tps = d["usage"]["completion_tokens"] / (time.time() - t0)

print("CONFIG: %s   prefill=%dK" % (LABEL, TARGET // 1000), flush=True)
print("  reference decode (undisturbed): %.1f tok/s" % ref_tps, flush=True)

ta = threading.Thread(target=stream_gen)
ta.start()
time.sleep(5)
tb = threading.Thread(target=long_call)
tb.start()
time.sleep(20)
ts = threading.Thread(target=short_call)
ts.start()
ta.join(); ts.join(); tb.join()

gaps = [stream_ts[i + 1] - stream_ts[i] for i in range(len(stream_ts) - 1)]
during = [g for g in gaps]
tps_during = 1.0 / (sum(during) / len(during)) if during else 0
print("  decode DURING prefill:    %.2f tok/s  (%.1f %% of reference)"
      % (tps_during, 100 * tps_during / ref_tps), flush=True)
print("  token interval p50/p95/max: %.3f s / %.3f s / %.3f s"
      % (pct(during, 0.50), pct(during, 0.95), max(during) if during else 0), flush=True)
print("  TTFT new short request:   %s" % ("%.1f s" % short_ttft[0] if short_ttft else "-"), flush=True)
if long_res:
    print("  long job: %s tok in %.1f s  (%.0f tok/s prefill)"
          % (format(long_res[0][0], ","), long_res[0][1], long_res[0][0] / long_res[0][1]), flush=True)
verdict = "GREEN" if tps_during / ref_tps >= 0.25 else "RED"
print("  >>> %s  (requirement: >=25 %% of reference decode)" % verdict, flush=True)
