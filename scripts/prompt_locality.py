#!/usr/bin/env python3
"""Prompt Cache Analyzer — vad kostar skillnaden mellan tva prompter?

Anvander serverns EGEN tokenizer (/tokenize) sa blockgranserna blir exakta.

  python3 cacheanalyze.py a.json b.json      # tva chat-completion-bodies
  python3 cacheanalyze.py a.txt b.txt        # tva rena texter

Tva lagen:
  TOKENLAGE     exakt forsta divergens, blockindex, teoretisk ateranvandning
  STRUKTURLAGE  (for JSON) vilken message/falt som flyttats eller andrats
"""
import json, os, sys, urllib.request

# REDACT=1 -> skriv aldrig ut promptinnehall, bara positioner och andelar.
REDACT = os.environ.get("REDACT", "0") == "1"

BASE = os.environ.get("BASE", "http://127.0.0.1:8888")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-0731")
BLOCK = 256


# Uppmatt prefillhastighet per kontextdjup (DS4-0731, 2x DGX Spark, chunk 8192).
# Attention ar O(n^2) -> hastigheten faller med djupet. En konstant underskattar
# kostnaden for djupa kontexter kraftigt. Kalibrera om for andra riggar.
PREFILL_CURVE = [(33089, 1810.0), (132041, 1758.0), (529151, 1308.0), (929733, 1017.0)]


def prefill_rate(n_tokens):
    """Linjart interpolerad prefillhastighet vid givet djup."""
    if n_tokens <= PREFILL_CURVE[0][0]:
        return PREFILL_CURVE[0][1]
    if n_tokens >= PREFILL_CURVE[-1][0]:
        return PREFILL_CURVE[-1][1]
    for (x0, y0), (x1, y1) in zip(PREFILL_CURVE, PREFILL_CURVE[1:]):
        if x0 <= n_tokens <= x1:
            return y0 + (y1 - y0) * (n_tokens - x0) / (x1 - x0)
    return PREFILL_CURVE[-1][1]


def tokenize(text):
    body = json.dumps({"model": MODEL, "prompt": text}).encode()
    r = urllib.request.Request(BASE + "/tokenize", body, {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=120).read())
    return d.get("tokens") or d.get("token_ids") or []


def flatten(obj):
    """Chat-body -> den text som faktiskt hamnar i prompten (approximativt men konsekvent)."""
    if isinstance(obj, dict) and "messages" in obj:
        parts = []
        for m in obj["messages"]:
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
            parts.append("<|%s|>%s" % (m.get("role", "?"), c or ""))
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
        out.append("tools: definitionerna eller deras ordning skiljer sig")
    ma, mb = a.get("messages", []), b.get("messages", [])
    if len(ma) != len(mb):
        out.append("messages: antal %d -> %d" % (len(ma), len(mb)))
    for i in range(min(len(ma), len(mb))):
        if ma[i] != mb[i]:
            ra = ma[i].get("role")
            ca, cb = str(ma[i].get("content")), str(mb[i].get("content"))
            j = next((k for k in range(min(len(ca), len(cb))) if ca[k] != cb[k]), min(len(ca), len(cb)))
            if REDACT:
                out.append("messages[%d] (%s): skiljer fran tecken %d  [innehall dolt, REDACT=1]"
                           % (i, ra, j))
            else:
                out.append("messages[%d] (%s): skiljer fran tecken %d  ...%s... -> ...%s..."
                           % (i, ra, j, ca[max(0, j - 25):j + 25].replace("\n", " "),
                              cb[max(0, j - 25):j + 25].replace("\n", " ")))
            if i < len(ma) - 1:
                out.append("  ^ detta ar INTE sista meddelandet -> allt efter det invalideras ocksa")
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    A, B = load(sys.argv[1]), load(sys.argv[2])
    ta, tb = tokenize(flatten(A)), tokenize(flatten(B))
    n = min(len(ta), len(tb))
    div = next((i for i in range(n) if ta[i] != tb[i]), n if len(ta) != len(tb) else -1)

    print("=" * 62)
    print("PROMPT CACHE ANALYZER")
    print("=" * 62)
    print("Prompt A:                  %s tokens" % format(len(ta), ","))
    print("Prompt B:                  %s tokens" % format(len(tb), ","))
    if div < 0:
        print("\nIdentiska -> 100 %% cachetraff, 0 tokens omprefill.")
        return
    blk = div // BLOCK
    reuse = blk * BLOCK
    refill = len(tb) - reuse
    print("Forsta divergens:          token %s" % format(div, ","))
    print("Blockstorlek:              %d" % BLOCK)
    print("Forsta ogiltiga block:     %d av %d" % (blk, (len(tb) + BLOCK - 1) // BLOCK))
    print("Ateranvandbar prefixcache: %s tokens (%.1f %%)" % (format(reuse, ","), 100.0 * reuse / len(tb)))
    print("Omprefill:                 %s tokens (%.1f %%)" % (format(refill, ","), 100.0 * refill / len(tb)))

    # --- tidsestimat: separat sektion, med explicit lagre konfidens ---
    # Kostnaden ar INTE proportionell mot antalet omraknade tokens. Attention ar
    # O(n^2): en token vid position i attendar over hela sitt prefix. Ett suffix
    # fran position p kostar (n^2 - p^2)/2, alltsa andelen 1 - (p/n)^2.
    # Verifierat mot egen matning: middle/top-kvoten blev 0,811 uppmatt;
    # positionsmodellen ger 0,75-0,80, den naiva tokenmodellen 0,50.
    n = len(tb)
    frac = 1.0 - (reuse / float(n)) ** 2
    full = n / prefill_rate(n)
    est = full * frac
    print("\n--- TIDSESTIMAT (lagre konfidens an cachesiffrorna ovan) ---")
    print("Positionsandel av full prefill: %.1f %%   [1 - (p/n)^2]" % (100 * frac))
    print("Uppskattad extrakostnad:        ~%.1f s" % est)
    print("Kalibrering:                    chunk 8192, DS4-0731 pa 2x DGX Spark")
    print("VARNING: tidsestimatet ar konfigurationsberoende och OKALIBRERAT for")
    print("         andra runtimeprofiler. Uppmatt avvikelse upp till 1,7x nar")
    print("         kalibreringen gjordes med annan MAX_NUM_BATCHED_TOKENS.")
    print("         Cachesiffrorna ovan ar daremot deterministiska.")
    pos = 100.0 * div / len(tb)
    verdict = ("BRA — mutationen ligger sist, nastan hela cachen ateranvands" if pos > 90 else
               "ILLA — mutationen ligger tidigt, nastan allt maste rakans om" if pos < 25 else
               "MEDEL — mutationen ligger mitt i, halva kontexten forloras")
    print("\nMutationens position:      %.1f %% in i prompten\n%s" % (pos, verdict))
    sd = structure_diff(A, B)
    if sd:
        print("\n--- STRUKTURLAGE: trolig orsak ---")
        for s in sd:
            print("  " + s)
    print("\nRegel: statiskt forst (systemprompt, verktyg, stabil historik),")
    print("       dynamiskt sist (tidsstamplar, status, nya verktygsresultat).")


main()
