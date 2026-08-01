"""Prefix Cache Locality — vad kostar en muterande token, beroende pa VAR den star?

Simulerar en agentloop: samma kontext skickas om och om igen, vaxande med ~200 tok
per varv. Fyra varianter skiljer sig bara i VAR ett muterande falt ligger.

  clean        inget muterar               -> cachen traffar fullt
  dirty-bottom muterar sist i prompten     -> bara svansen rakans om
  dirty-middle muterar mitt i              -> halva kontexten rakans om
  dirty-top    muterar overst              -> ALLT rakans om, varje varv

TTFT ar mattet: den ar proportionell mot hur mycket som maste prefillas pa nytt.
vLLM cachar i block om 256 tokens (--block-size 256), sa invalideringen sker
fran och med det block dar mutationen ligger.
"""
import json, time, urllib.request

import os
PORT = os.environ.get("PORT", "8888")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-0731")
U = "http://127.0.0.1:%s/v1/chat/completions" % PORT
BLOCK = ("Historikpost: agenten laste filen och noterade att konfigurationen ser korrekt ut. "
         "Inga avvikelser rapporterade. Verktyget returnerade status noll. ")
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
    """~20K tokens statisk bas + vaxande historik."""
    static = "SYSTEMINSTRUKTION: du ar en driftassistent. " + BLOCK * 700   # ~20K tok
    hist = BLOCK * (turn * 12)                                              # +~200 tok/varv
    mut = "AKTUELL TID: 2026-08-01 %02d:%02d:%02d. " % (22, turn, salt % 60)
    if variant == "clean":
        return static + hist + "\n\nSvara med ordet OK."
    if variant == "dirty-top":
        return mut + static + hist + "\n\nSvara med ordet OK."
    if variant == "dirty-middle":
        half = len(static) // 2
        return static[:half] + mut + static[half:] + hist + "\n\nSvara med ordet OK."
    return static + hist + mut + "\n\nSvara med ordet OK."          # dirty-bottom


print("modell: %s  port: %s" % (MODEL, PORT), flush=True)
print("varmer upp cachen...", flush=True)
for v in ("clean", "dirty-top", "dirty-middle", "dirty-bottom"):
    build(v, 0, 0)
ttft(build("clean", 0, 0))

print("\nvariant         varv-1 TTFT   varv 2-8 median   snitt 2-8   vs clean", flush=True)
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
