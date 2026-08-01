# DeepSeek-V4-Flash-0731 på 2× DGX Spark — mätresultat 2026-08-01

Alla siffror är egenmätta samma kväll på samma rigg. Där en slutsats senare visade sig
fel står den kvar tillsammans med korrigeringen — se §7.

## Uppställning

| | |
|---|---|
| Hårdvara | 2× NVIDIA DGX Spark (GB10, sm_121), 128 GB unified LPDDR5X per nod |
| Sammankoppling | RoCE, direktkopplad, asymmetrisk portmappning (nod 1 port 1 ↔ nod 2 port 0) |
| Modell | `deepseek-ai/DeepSeek-V4-Flash-0731`, rev `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, 156 GB |
| Recept | `tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark` @ `d728faee` |
| Basimage | `ghcr.io/bjk110/vllm-spark@sha256:d8492e7677cf1b9aaa3344e0e6865efc468454013eee5ebabac85be90af027be` |
| Runtime | `vllm-dspark-runtime:dspark-nvfp4-stage-c`, vLLM `0.21.1rc1.dev339+g1967a5627bc3` |
| Topologi | TP=2, PP=1, `nvfp4_ds_mla` KV, block-size 256 |
| Spekulation | DSpark, k=5 statisk |
| GMU | 0,78 |

**Bygget tar ~20 sekunder, inte 30–60 minuter** — overlayet är ren Python på en färdig
basimage, ingen NVCC-kompilering. Basimagen visade sig dessutom vara byte-identisk med
`aidendle94/sparkrun-vllm-ds4-gb10:production-ready`.

**Patch 4** (shared-expert gate_up_proj) är **redan inbakad** i receptets overlay sedan
2026-07-31. Applicera den inte — verifiera hashen:
`4c11f54f7e74d658d40d4ec091f2b9911efd41ebde9f894f87102bcd3d3caba3`.

---

## 1. Produktionsstatus

- `Using 'B12X' Mxfp4 MoE backend` bekräftad aktiv *(utan den: ~60 → ~29 tok/s enligt receptet)*
- 1M-kontext verifierad både tekniskt och kvalitativt
- **Korrekthet: 40/40** — kod 10/10, svenska 10/10 med korrekt diakritik,
  verktygsanrop **10/10** korrekt formade och parsbara, nålsökning ~12K 10/10

**Decode, varmt, `stream:false`, median av tre:**

| Innehåll | Decode | Draft-acceptans |
|---|---|---|
| Kod / strukturerat | **62,5–68,1 tok/s** | **71,0 %** |
| Prosa, engelska | 35,3 tok/s | 28,4 % |
| Prosa, svenska | 30,3 tok/s | 19,5 % |
| Blandat (3 kod + 2 prosa) | 50,1 tok/s | — |

Kodacceptansen på 71,0 % överträffar receptets referens 68,7 % och bevisar att den
inbakade Patch 4 är aktiv — utan den ligger acceptansen runt 26 %.

**🇸🇪 Svensk text kostar ~9 procentenheter draft-acceptans och ~14 % decode mot engelsk**,
mätt med innehållsmässigt parade promptar. Det är en egenskap hos drafteren, inte ett
konfigurationsfel.

Per-positionsacceptans: 74,5 / 52,9 / 37,4 / 28,1 / 21,6 %.

---

## 2. Kontextdjup

3 nålar per djup (8 %, 50 %, 92 % in i dokumentet), med distraktorkoder.

| Djup | Prompt-tokens | Träffar | TTFT | Prefill |
|---|---|---|---|---|
| 32K | 33 089 | 3/3 | 0,30 min | 1 810 tok/s |
| 128K | 132 041 | 3/3 | 1,25 min | 1 758 tok/s |
| 512K | 529 151 | 3/3 | 6,74 min | 1 308 tok/s |
| 900K | 929 733 | **3/3** | 15,24 min | 1 017 tok/s |

**12/12 nålar.** Full 1M-retrieval fungerar — något receptet självt inte hade bevisat.
Prefillen klingar av med djupet (attentionens O(n²)).

> **1M är en fungerande kapacitet för batchanalys, inte ett interaktivt standardläge.**
> En 900K-prefill tar en kvart. Ingen sitter och väntar på den i en agentloop.

---

## 3. Tre motorer — den viktigaste systemförklaringen

En LLM-server är inte en resurs utan tre nästan oberoende. Samma maskin, samma
konfiguration, samma timme:

| Motor | Begränsas av | Skalar med samtidighet? |
|---|---|---|
| **Prefill** | compute | **Nej** — aggregatet helt platt |
| **Decode** | minnesbandbredd + batchning | **Ja** — +125 % till N=6 |
| **Admission** | schemaläggning | **Nej** under lång prefill |

### Prefill-samtidighet (32K per session, end-to-end inkl. prefill)

| N | per ström | aggregat | värsta latens |
|---|---|---|---|
| 1 | 6,8 tok/s | 6,8 | 22,1 s |
| 2 | 3,2 | 6,4 | 40,1 s |
| 4 | 1,7 | 6,6 | 77,4 s |
| 6 | 1,3 | **6,5** | 115,2 s |

Aggregatet varierar 5 % medan per-ström faller med faktor 5,2. **Ren serialisering.**
En enda 32K-prefill mättar systemet.

### Decode-samtidighet (kort unik prompt, 1200 tokens generering)

| N | aggregat | per ström | TTFT max | acceptans |
|---|---|---|---|---|
| 1 | 15,2 | 15,3 | 0,2 s | 21,5 % |
| 2 | 22,0 | 11,8 (77 %) | 0,4 s | 18,9 % |
| 4 | **31,5** | 8,3 (54 %) | 0,4 s | 23,4 % |
| 6 | 34,2 | 6,4 (42 %) | 1,7 s | 23,2 % |

**Mättnad vid N≈4.** Steget 4→6 ger +9 % aggregat men fyrdubblar värsta TTFT.
*(Baslinjen 15,3 tok/s är låg för att uppgiften var svensk prosa — vårt sämsta innehåll.
Det är skalningsformen som är resultatet, inte absoluttalet.)*

### Admission under lång prefill

Tre korta anrop startade 25 s in i ett 516 565-tokens prefill:

```
korta anrop:  367,5s · 367,7s · 367,7s      (normal TTFT varm: 1,66 s)
långjobbet:   516 565 tok på 391,2 s
```

**221× normal latens.** Alla tre släpptes när prefillen var klar.
Pågående decode svälts samtidigt till **~1/35 av normal hastighet** — inte fryst, men
strypt till en spillra.

> Detta gäller **inte** korta prompter: vid N=4 med korta prompter är TTFT 0,4 s.
> Blockeringen är specifikt kopplad till att lång prefill mättar compute.

---

## 4. Chunkstorlek — `MAX_NUM_BATCHED_TOKENS`

Mätt vid 256K prefill, `--async-scheduling` av i båda fallen.

| Mått | 8192 | 2048 |
|---|---|---|
| Prefill | 1 529 tok/s | 1 469 tok/s (**−3,9 %**) |
| p95 tokenlucka under prefill | 5,197 s | **1,590 s** (−69 %) |
| max tokenlucka | 6,417 s | 2,085 s |
| Decode-andel under prefill | 7,1 % | 7,3 % (**oförändrad**) |
| TTFT nytt kort anrop | 150,7 s | 158,1 s (oförändrad) |
| **KV-pool** | 1 598 763 tok | **2 671 557 tok** (+67 %) |
| Max samtidighet @1M | 1,59× | **2,55×** |

**Mekanismen är bekräftad:** tokenluckorna *är* prefill-chunkarna.
`8192 ÷ 1529 = 5,36 s` mot uppmätt p95 5,20 s. Vid 2048: `2048 ÷ 1469 = 1,39 s` mot uppmätt 1,59 s.

**Men chunkstorleken är en jitterspak, inte en rättvisespak.** Decode-andelen ligger fast
på ~7 %: vid 8192 hinner decode ~9 tokens per chunk, vid 2048 ~2,3 — fyra gånger fler
tillfällen, fyra gånger färre tokens per tillfälle, netto noll. Schemaläggaren ger decode
en *fast andel av tokenbudgeten*, och den andelen kan inte ändras genom att ändra budgetens storlek.

**Slutsats:** 2048 ger kraftigt bättre jitter och 67 % större KV-pool för 4 % prefill.
Den löser inte rättvisa eller admission. Den är ändå en stark agentprofil.

`--async-scheduling` av gav ingen meningsfull förbättring → flaggan var inte roten.

---

## 5. Prefix-cachelokalitet ⭐ det mest generellt användbara resultatet

Samma kontext skickad om och om igen med ~200 tokens tillväxt per varv. Varianterna
skiljer sig **bara i var ett muterande fält ligger**.

| Variant | TTFT snitt (varv 2–8) | vs clean |
|---|---|---|
| clean (inget muterar) | 0,65 s | 1,0× |
| **dirty-bottom** (muterar sist) | **0,45 s** | **0,7×** |
| dirty-middle | 14,64 s | 22,5× |
| **dirty-top** (muterar överst) | **18,05 s** | **27,7×** |

**40 gångers skillnad mellan att mutera sist och överst.**

`dirty-bottom` är inte bara nära clean — den är *snabbare*, eftersom mutationen hamnar i
ett eget delvis fyllt block. **Volatila fält sist är i praktiken gratis.**

### Verifierat med `prompt_locality.py` mot serverns egen tokenizer

Identisk 16 839-tokens prompt, bara tidsstämpelns placering skiljer:

```
TIDSSTÄMPEL ÖVERST              TIDSSTÄMPEL SIST
första divergens: token 13      första divergens: token 16 834
återanvändbar:    0 (0,0 %)     återanvändbar:    16 640 (98,8 %)
omprefill:        16 839 (100%) omprefill:        200 (1,2 %)
extrakostnad:     11,2 s/varv   extrakostnad:     0,1 s/varv
```

**112× skillnad i omprefill genom att flytta ett fält.** I en agentloop med tjugo steg är
det fyra minuter förlorade på en tidsstämpel.

Med `--block-size 256` är utfallet **binärt, inte gradvis**: en mutation i första blocket
ger noll procent återanvändning. En ändrad token kostar aldrig "en token" — den kostar
minst 256, och allt efter den.

> ## 🎯 Designregel
> **Statiskt först, dynamiskt sist.**
> Systemprompt, verktygsdefinitioner och stabil historik byte-identiska överst.
> Tidsstämplar, aktuell status och nya verktygsresultat sist.
>
> Gäller oavsett hårdvara och inferensmotor. Kostar ingenting att följa.

Vanliga cachebrytare att kontrollera: verktygslistans ordning · JSON-nyckelordning ·
whitespace · request-ID · omsorterat minne · dynamiskt tillagd systemtext.

---

## 6. Rekommenderade profiler

### Agentprofil *(standard)*
```
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=6          # cudagraph = seqs × (k+1) = 36
GPU_MEMORY_UTILIZATION=0.78
MTP_NUM_TOKENS=5
```
- högst ~4 samtidiga genereringar (mättnadspunkt)
- normal kontext ≤128K
- dynamiska promptfält sist
- `repetition_penalty` **filtreras i gateway** — kraschar DSpark-vägen med illegal memory access
- `presence_penalty` och `frequency_penalty` **tillåts** — verifierat säkra vid 0,6 och 1,5

### Batchprofil *(långkontext)*
```
MAX_NUM_BATCHED_TOKENS=8192
```
- 512K–1M
- **exklusiv körning**, ingen samtidig agenttrafik
- schemalagt batcharbete

Köseparationen är inte en workaround. Vi har testat den enda spak som rimligen kunde ha
löst samexistensen, och den kan det inte.

---

## 7. Slutsatser vi korrigerade efter bättre mätning

Dokumenterade för att de är lika användbara som resultaten.

1. **Max-gap-testet feltolkade partiell svält som "grönt".** Vår första
   överlappsmätning letade efter ett *stopp* längre än 20 s. Det inträffade aldrig, så
   skriptet skrev ut "decode fortsatte" — trots att decode var strypt till 1/35 av normal
   fart. **Mät p50/p95 av tokenintervall, inte bara största lucka.**
2. **`GMU 0.70` som "säkerhetsåtgärd" var destruktivt.** Vikterna tar 77,7 GiB av nodens
   ~119,6 GiB. Vid 0,70 blev KV-utrymmet 2,39 GiB — otillräckligt för ens 32K.
   **GMU är en KV-spak, inte en säkerhetsspak, när vikterna dominerar minnet.**
3. **`MAX_NUM_SEQS=1` bootar inte med k=5.** Cudagraph-storlekar måste vara multiplar av
   k+1; kandidaterna [1, 2, 4] innehåller ingen multipel av 6.
   **Allt runt spekulationen måste vara multiplar av k+1.** (Samma familj som k=7-förbudet.)
4. **Prefix-cachen kontaminerade ett samtidighetstest.** Två långa jobb byggda med samma
   funktion fick identisk text; jobb två fick cacheträff i stället för att prefilla.
   **Salta varje prompt unikt.**
5. **`rsync -aL` på en HF-cache dubblerar den** (~311 GiB i stället för ~155). `-L`
   dereferensar snapshot-symlänkarna. Använd `-a`.
6. **Admission-blockering är inte "mindre allvarlig" än decode-svält.** En agentloop är
   `modell → verktyg → modell`; varje pil är ett nytt API-anrop. Admission-blockering
   träffar varje steg efter varje verktygsanrop.
7. **`pkill -f <mönster>` self-matchar.** En väntloop på `pgrep -f conc.py` matchade
   `decconc.py` — sitt eget kommando — och väntade på sig själv. Använd PID.

## 8. Kända begränsningar i mätningarna

- Prefill-samtidighetstestet är prefill-dominerat (32K prompt, ~150 genererade tokens)
  och säger inget om decode-samtidighet — därav det separata decode-testet.
- Decode-baslinjen använde svensk prosa, vårt sämsta innehåll för draft-acceptans.
  Absoluttalen är därför låga; skalningsformen är resultatet.
- Referensdecode i överlappstesterna mättes på kall server (23–24 tok/s). Båda
  konfigurationerna mättes under samma förhållanden, så jämförelsen är rättvis.
- Andra långa jobbet i test B fick cacheträff → bandbreddsdelning mellan två prefills
  är fortfarande obesvarad.
- Ingen soak-körning gjord. Xid senaste 2 h: 0. Minnet återgick efter avslutade jobb.
- Stage C:s 584-byte-envelope och k=3:s påstådda 24 %-kostnad är hämtade från receptet
  och **inte** egenverifierade.

---

## 9. Laguna-replikationen och nattens driftfynd (2026-08-02, 00:00–00:40)

### Cachelokaliteten replikerad — se `HYPOTHESES-laguna-replication.md`

H1, H2 och H4 bekräftade på Laguna S 2.1 (vLLM 0.25.1, en nod, poolside-tokenizer):
`bottom ≈ clean (1,1×) << middle (9,1×) < top (15,1×)`, och analysatorn förutsade
block 0 / 0 % korrekt mot en okalibrerad tokenizer. H3:s absoluta del missade
(13× mot förutsagt 20–40×) — fasta kostnader komprimerar kvoterna vid mindre kontext.

### 🔴 Laguna 262K bootar inte — vLLM #48140 (stängd "not planned")

vLLM:s UMA-startkontroll läser i praktiken `MemFree`, inte `MemAvailable` — reclaimbar
sidcache bokförs som upptagen. Tre starter, tre nästan identiska budgetar oavsett
processhistorik och minnesstartläge: **4,22 / 5,18 / 5,34 GiB** rapporterat tillgängligt KV.
262K kräver 18,35 GiB → deterministisk krasch. 32K kräver ~2,3 GiB → bootar (67 236 tokens
KV-pool, 2,05× samtidighet).

| Laguna-profil | Status |
|---|---|
| 32K–64K | ✅ bootar normalt, behåll sidcachen (den *snabbar* laddningen) |
| 128K–262K | 🔴 kräver `drop_caches` före start (overifierat) eller lokal UMA-patch av `gpu_worker.py`. Kausalitetstest återstår: bootar 262K efter cachetömning är diagnosen komplett |

**Sidcachens dubbelroll:** samma sidor snabbar viktladdningen (8,75 s/shard i tredje
starten mot 11,5 kall) och sänker samtidigt den rapporterade KV-budgeten. Bra för
laddaren, gift för profileraren.

### Lagunas startprofil (tre starter uppmätta)

| Fas | Tid |
|---|---|
| FP4-JIT, kall kernelcache | ~24 min (engångs; cache i `~/.cache/vllm`, persistent) |
| Viktladdning | ~12–13 min oavsett startläge; första ~10 shards långsammare (UVM-ramp) |
| Profilering/capture/autotune | ~2–4 min |
| **Kall totalt** | **~39 min** |
| **Varm totalt (till API)** | **13 min 16 s uppmätt** |
| Ready (generation 1) | +5,1 s |
| Warm (generation 2) | 3,5 s |

Viktladdaren är enkeltrådad (101 % CPU) — golvet för varje omstart, oavsett cache.

### Nya mätdisciplinregler (dyrköpta i natt)

6. **Lita inte på process-RSS på GB10** — CUDA/UVM-allokeringar syns inte där. Läs
   systemets `free` och motorns egna loggar.
7. **Läs inte momentan ETA som trend** — shard-takt under uppramp gav en falsk
   2,6×-slutsats som självdog vid shard 26.
8. **Jämför samma fas mellan körningar** — tidiga shards mot tidiga shards, stabil
   fas mot stabil fas.
9. **`pkill`/`pgrep -f` self-matchar även över SSH** — mönstret finns i fjärrskalets
   kommandorad. Använd `[b]racket`-mönster eller PID-filer. (Bet oss tre gånger.)

---

## 10. Real agent workload (Hermes v0.19.0, 2026-08-02 01:10)

A real agent task was run through the capture proxy against the 2048 agent profile:
find and fix a planted bug in a small Python project, run the test suite, handle the
failing test, summarize. The agent solved it correctly (bug found: `len(stock)` →
`sum(stock.values())`; 4/4 tests passing) in 10 model calls with 20 tool definitions
and a ~15–18K-token growing prompt.

**Prompt locality across agent turns (server tokenizer, 256-token blocks):**

| Transition | Prompt size | First divergence | Reusable prefix | Re-prefill |
|---|---|---|---|---|
| turn 1→2 | 15,169 tok | token 15,168 (last) | **97.1 %** | 453 tok |
| turn 5→6 | 16,549 tok | token 16,548 (last) | **98.7 %** | 213 tok |
| turn 9→10 | 17,655 tok | token 17,654 (last) | **97.5 %** | 439 tok |

**The agent framework is already cache-optimal.** Prompts are built append-only: no
mutating timestamps, no reordered tool lists, no rewritten history. Divergence sits at
the very last token of the previous prompt, every turn. Each agent step re-prefills only
the new tail (~200–450 tokens ≈ 0.2 s) instead of the full context (~10 s).

This is the null-result counterpart to the synthetic dirty-top experiments: the tool's
value here was *verifying* cache health, not finding a problem. Frameworks that inject
volatile content early would show divergence at token ~13 instead — an 
order-of-magnitude-larger per-turn cost that this analysis makes visible in seconds.

### Code-content decode concurrency and end-to-end agent timing (01:30)

Decode concurrency re-measured with **code content** (the Swedish-prose run above was the
worst case; agent workloads are code-dominated):

| N | aggregate | per stream | draft acceptance |
|---|---|---|---|
| 1 | 48.0 tok/s | 48.0 | 62.3 % |
| 2 | 72.9 | 38.1 | 72.6 % |
| 4 | **126.3 tok/s** | 34.0 | **71.3 %** |

Acceptance holds ~71 % under concurrency — speculation does not degrade with parallel
streams. At N=4, each stream still receives roughly what the previous production model
delivered to a single user (35.1 tok/s).

A second end-to-end agent task (different bug class) completed correctly in **316 s**,
10-ish model calls, 4/4 tests passing.
