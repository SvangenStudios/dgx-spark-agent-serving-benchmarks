# Threats to Validity

Vad som kan göra slutsatserna i `RESULTS-2026-08-01.md` fel eller mindre generella än de ser ut.
Varje punkt pekar också ut vilket experiment som skulle avgöra saken.

---

## 1. Kalibreringskurvan gäller en enda runtimeprofil

`PREFILL_CURVE` i analysatorn är mätt på DS4-0731, 2× DGX Spark, **chunk 8192**.
Den användes sedan för att skatta kostnad i ett test som kördes med **chunk 2048**.

**Uppmätt konsekvens: 1,7× fel.** Prediktion 9,3 s, verklighet 18,05 s.

Dessutom har kurvan **ingen datapunkt under 32K** — den extrapolerar med en konstant
1 810 tok/s i precis det intervall där verktyget faktiskt användes.

> **Avgörande experiment:** kortkontextmatris 2K → 5K → 10K → 20K → 32K → 64K → 128K vid
> den profil verktyget ska användas mot. Tre repetitioner, unika promptar, en output-token,
> spara median **och spridning**. Kalibrera mot tid direkt — `T(n) = a + bn + cn²` — inte via
> tok/s, eftersom fasta kostnader dominerar under 10K och då förklär sig till hastighet.

---

## 2. Positionsmodellen är en approximation, inte en runtime-lag

`1 − (p/n)²` modellerar **kausal attention och ingenting annat**. Utanför modellen ligger:
MoE-routing, TP-kommunikation över RoCE, chunkning, blockallokering, cache-eviktion,
kernelval och schemaläggningsoverhead.

Att den passade vår `middle/top`-kvot (0,811 uppmätt mot 0,75–0,80 predikterat) är stöd,
inte bevis. **En datapunkt.**

> **Avgörande experiment:** mät flera mutationspositioner — 10, 25, 50, 75, 90 % — och se
> om hela kurvan följer `1 − (p/n)²` eller bara råkar passa vid 50 %.

---

## 3. Promptlokaliteten är mätt på vLLM:s prefix-cache

Mekanismen förutsätter att motorn återanvänder **sammanhängande prefixblock från början**.
Det gäller vLLM. Det gäller inte nödvändigtvis:

- motorer helt utan prefix-cache (där finns effekten inte alls)
- radix-/trädbaserad cachning (SGLang) där grenar delas mellan requests
- llama.cpp/Ollama som håller en KV-sekvens per konversation och kan skifta den
- motorer med sliding-window attention där tidiga tokens ändå vräks ut

> **Avgörande experiment:** samma test på en motor med annan cachearkitektur.
> Ett **motexempel är mer värdefullt än en tredje bekräftelse** — det är där gränsen
> för modellen går, och gränsen är det som gör den användbar.

---

## 4. Två modellstackar är replikation, inte generalisering

DeepSeek-V4-Flash-0731 och Laguna S 2.1 skiljer sig på modell, tokenizer, chattmall,
parser, FP4-backend, vLLM-version och topologi — men delar prefix-cacheprincipen.

Håller mönstret är det **stark replikation**. Det är inte samma sak som att det gäller
alla inferensmotorer.

---

## 5. Mätspecifika svagheter

| Svaghet | Konsekvens |
|---|---|
| Prefill-samtidighetstestet är prefill-dominerat (32K prompt, ~150 genererade tokens) | Säger inget om decode-samtidighet — därav separat test |
| Decode-baslinjen använde svensk prosa, vårt sämsta innehåll | Absoluttalen är låga; **skalningsformen** är resultatet |
| Referensdecode i överlappstesterna mättes kall (23–24 tok/s) | Båda konfigurationerna mättes lika, så jämförelsen håller |
| Andra långa jobbet i test B fick prefix-cacheträff | Bandbreddsdelning mellan två prefills är **obesvarad** |
| `dirty-bottom` kan påverkas av chattmallen | Om mallen lägger till text efter användarmeddelandet är mutationen inte längre sist |
| Ingen soak-körning | Långtidsstabilitet ej prövad. Xid senaste 2 h: 0 |
| Stage C:s 584-byte-envelope, k=3:s 24 %-kostnad | Hämtade från receptet, **ej egenverifierade** |

---

## 6. Brist i pre-registreringen (noterad, ej efterhandsjusterad)

H3 påstod att kvoten top/bottom landar på 20–40×. **Den absoluta delen går inte att
tolka rent** utan Laguna-specifik tidskalibrering — missar den vet vi inte om det är
modellen eller kalibreringen, och utfallet blir oinformativt snarare än falsifierande.

Det borde ha stått i `HYPOTHESES-laguna-replication.md` från början. Hypotesen lämnas
oförändrad; bristen noteras här i stället för att skrivas om i efterhand.

---

## 7. Vad som INTE är hotat

Värt att skilja ut, eftersom listan ovan annars kan läsas som att allt är osäkert:

- **Cacheinvalideringen är deterministisk.** Första divergens, blockindex och
  återanvändbar andel följer direkt av tokenisering och blockstorlek.
- **Tre-motorsseparationen är direkt uppmätt**, inte modellerad: prefill-aggregatet platt
  över N=1→6, decode +125 %, admission blockerad. Samma maskin, samma timme.
- **`MAX_NUM_BATCHED_TOKENS` → KV-pool** är avläst ur bootloggen, inte härledd.
- **Korrekthetsresultaten** (40/40, 12/12 nålar) är räknade utfall.
