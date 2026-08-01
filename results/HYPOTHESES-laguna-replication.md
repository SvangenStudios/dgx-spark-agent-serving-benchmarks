# Pre-registrerade hypoteser — cache-lokalitet på Laguna S 2.1

**Skrivet 2026-08-01 kl 23:18, innan resultatet fanns.**
Syftet är att inte omedvetet anpassa tolkningen efter utfallet.

## Vad som replikeras

`cache_locality.py` kördes på DeepSeek-V4-Flash-0731 och gav:

| variant | TTFT snitt | vs clean |
|---|---|---|
| clean | 0,65 s | 1,0× |
| dirty-bottom | 0,45 s | 0,7× |
| dirty-middle | 14,64 s | 22,5× |
| dirty-top | 18,05 s | 27,7× |

Samma skript körs nu oförändrat mot Laguna S 2.1 på nod 3.

## Vad som skiljer uppställningarna

| | DS4-0731 | Laguna S 2.1 |
|---|---|---|
| Modell | 304B MoE, FP8-vikter + FP4-experter | 67 GB NVFP4 |
| Tokenizer | `deepseek_v4` | poolside |
| vLLM | 0.21.1rc1.dev339 (DSpark-overlay) | **0.25.1 upstream** |
| Spekulation | DSpark, k=5 | DFlash, k=15 |
| Topologi | TP=2 över RoCE, 2 noder | TP=1, 1 nod |
| block-size | 256 explicit | vLLM:s default |
| GMU | 0,78 | 0,85 |

Fem oberoende skillnader. Håller mönstret ändå är det motorns egenskap, inte modellens.

## Hypoteser

**H1 — Formen replikeras.** Kurvan `clean → bottom → middle → top` ser i huvudsak
likadan ut på Laguna: bottom ≈ clean, middle och top dramatiskt dyrare, top ≥ middle.

**H2 — Absoluttal skiljer, relativ effekt består.** Skillnader i sekunder beror på
modellens hastighet och tokenizer. Den *relativa* effekten av var första divergensen
ligger följer samma mönster.

**H3 (min egen, mer specifik).** Kvoten top/bottom följer aritmetiken
`total kontext ÷ oförändrad svans` snarare än något modellspecifikt. Eftersom kontexten
är ~17–20K tokens i båda fallen förväntar jag mig samma storleksordning, **20–40×**.

**H4.** Första divergensen hamnar på ungefär samma tokenposition (~13) i `dirty-top`,
och ger **0 % återanvändbar cache** även på Laguna — eftersom mutationen träffar block 0
oavsett hur stora blocken är.

## Vad som falsifierar

- `dirty-bottom` markant dyrare än clean → botten är *inte* gratis generellt
- `dirty-top` billigare än `dirty-middle` → blockmekanismen fungerar inte som antaget
- Kvoten under 5× → effekten är modellberoende och DS4-talet kan inte generaliseras
- Ingen skillnad alls mellan varianterna → prefix-caching är av eller fungerar annorlunda

## Vad som mäts utöver TTFT

`prompt_locality.py` körs separat mot samma promptpar på Laguna för att få
tokenantal, första divergens, blockindex och cache-hit-andel — så att de två modellerna
kan ställas bredvid varandra på tokennivå, inte bara i sekunder.

## Utfall — ifyllt 2026-08-02 00:36, hypoteserna ovan orörda

Kört på Laguna S 2.1, vLLM 0.25.1, **32K-kontextprofil** (262K-profilen bootar inte —
se #48140-fyndet i RESULTS), statisk kontext ~12K tokens (MULT=300; 700 spräckte 32K-taket
med poolsides tokenizer — i sig ett bevis på att tokenizers skiljer).

| variant | TTFT snitt (varv 2–8) | vs clean | DS4-referens |
|---|---|---|---|
| clean | 0,48 s | 1,0× | 1,0× |
| dirty-bottom | 0,55 s | **1,1×** | 0,7× |
| dirty-middle | 4,37 s | **9,1×** | 22,5× |
| dirty-top | 7,22 s | **15,1×** | 27,7× |

**H1 — BEKRÄFTAD.** `bottom ≈ clean << middle < top`. Rangordningen identisk.

**H2 — BEKRÄFTAD.** Absoluttalen skiljer (mindre modell, en nod, mindre kontext);
den relativa formen består.

**H3 — relativ form bekräftad, absolut del MISSAR.** top/bottom = 13,1× — under det
förutsagda 20–40×-bandet. top/middle = 1,65 mot positionsmodellens ~1,3. Båda
avvikelserna är konsistenta med att fasta kostnader komprimerar kvoterna vid mindre
kontext (12K mot 19K). Precis den tolkningssvårighet som §6 i THREATS förutsåg:
utan Laguna-egen tidskalibrering är den absoluta delen oinformativ, inte falsifierande.

**H4 — BEKRÄFTAD.** Toppmutation: divergens token 16 → block 0 av 53 → **0,0 %**
återanvändbart, 13 543 tokens omprefill. Bottenmutation: divergens token 13 536 →
block 52 av 53 → **98,3 %** återanvändbart, 231 tokens. Analysatorn förutsade
mekanismen korrekt mot en tokenizer den aldrig kalibrerats för.

### Slutsats

Cachelokalitetens form är replikerad över: annan modell (67 GB NVFP4 mot 304B MoE),
annan tokenizer (poolside mot deepseek_v4), annan chattmall och parser, annan
FP4-backend (FLASHINFER_CUTLASS mot B12X), annan vLLM-version (0.25.1 mot 0.21.1),
annan spekulationsmetod (DFlash k=15 mot DSpark k=5) och annan topologi (1 nod mot
TP=2 över RoCE). Den gemensamma nämnaren är vLLM:s sammanhängande prefix-cache.

**Effektstorleken är stackberoende. Mekanismen och regeln — statiskt först,
dynamiskt sist — är det inte.**
