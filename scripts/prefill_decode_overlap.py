"""Aterbrukbart overlappstest. Anrop: python3 overlap.py <prefill_tokens> <etikett>

Mater under en pagaende stor prefill:
  - decode tok/s for en redan pagaende strommande generering
  - p50 / p95 tokenintervall (fangar PARTIELL svalt som max-gap missar)
  - TTFT for ett NYTT kort anrop som kommer in mitt i
  - prefill tok/s och langjobbets totala tid
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
    # salt bryter prefix-cachen sa varje matning ar aterta
    return ("Session " + salt + ". ") + chunk * n + "\n\nSammanfatta i en mening."


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
         "messages": [{"role": "user", "content": "Skriv en lang detaljerad text om Ostersjons ekologi och hydrografi."}],
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
         "messages": [{"role": "user", "content": "Svara med ett ord: hej"}],
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


# --- 0. referens: decode utan storning ---
ref = []
t0 = time.time()
p = {"model": "deepseek-v4-flash-0731",
     "messages": [{"role": "user", "content": "Skriv en kort text om havet."}],
     "max_tokens": 120, "temperature": 0, "stream": False}
r = urllib.request.Request(U, json.dumps(p).encode(), {"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(r, timeout=300).read())
ref_tps = d["usage"]["completion_tokens"] / (time.time() - t0)

print("KONFIG: %s   prefill=%dK" % (LABEL, TARGET // 1000), flush=True)
print("  referens decode (ostord): %.1f tok/s" % ref_tps, flush=True)

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
print("  decode UNDER prefill:     %.2f tok/s  (%.1f %% av referens)"
      % (tps_during, 100 * tps_during / ref_tps), flush=True)
print("  tokenintervall p50/p95:   %.3f s / %.3f s" % (pct(during, 0.50), pct(during, 0.95)), flush=True)
print("  TTFT nytt kort anrop:     %s" % ("%.1f s" % short_ttft[0] if short_ttft else "-"), flush=True)
if long_res:
    print("  langjobb: %s tok pa %.1f s  (%.0f tok/s prefill)"
          % (format(long_res[0][0], ","), long_res[0][1], long_res[0][0] / long_res[0][1]), flush=True)
verdict = "GRON" if tps_during / ref_tps >= 0.25 else "ROD"
print("  >>> %s  (krav: >=25 %% av referensdecode)" % verdict, flush=True)
