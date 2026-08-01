# Pre-registered hypotheses — cache locality on Laguna S 2.1

**Written 2026-08-01 23:18, before the result existed** (committed as `a267c3e` in the
original working repo). Purpose: prevent unconsciously fitting the interpretation to the
outcome.

## What is being replicated

`cache_locality.py` on DeepSeek-V4-Flash-0731 gave:

| variant | TTFT mean | vs clean |
|---|---|---|
| clean | 0.65 s | 1.0× |
| dirty-bottom | 0.45 s | 0.7× |
| dirty-middle | 14.64 s | 22.5× |
| dirty-top | 18.05 s | 27.7× |

The same script, unchanged, is now run against Laguna S 2.1.

## What differs between the setups

| | DS4-0731 | Laguna S 2.1 |
|---|---|---|
| Model | 304B MoE, FP8 weights + FP4 experts | 67 GB NVFP4 |
| Tokenizer | `deepseek_v4` | poolside |
| vLLM | 0.21.1rc1.dev339 (DSpark overlay) | **0.25.1 upstream** |
| Speculation | DSpark, k=5 | DFlash, k=15 |
| Topology | TP=2 over RoCE, 2 nodes | single node |
| block-size | 256 explicit | vLLM default |
| GPU mem util | 0.78 | 0.85 |

Five-plus independent differences. If the pattern still holds, it is the engine's
property, not the model's.

## Hypotheses

**H1 — The shape replicates.** `clean ≈ bottom << middle ≤ top`.

**H2 — Absolute times differ, relative effect persists.**

**H3 (more specific, hence easier to fell).** The top/bottom ratio follows
`total context ÷ unchanged tail`; with ~17–20K tokens in both cases, expect the same order
of magnitude, **20–40×**.

**H4.** The first divergence lands in block 0 for `dirty-top` and yields **0 % reusable
cache** on Laguna too, regardless of block size.

## Falsification criteria

- `dirty-bottom` markedly more expensive than clean → bottom is *not* generally free
- `dirty-top` cheaper than `dirty-middle` → the block mechanism does not work as assumed
- Ratio below 5× → the effect is model-dependent; the DS4 number does not generalize
- No difference at all → prefix caching is off or works differently

## Outcome — filled in 2026-08-02 00:36, hypotheses above untouched

Run on Laguna S 2.1, **32K context profile** (the 262K profile does not boot — see the
#48140 finding in RESULTS §6), static context ~12K tokens (MULT=300; 700 exceeded the 32K
cap with poolside's tokenizer — itself proof that tokenizers differ).

| variant | TTFT mean (turns 2–8) | vs clean | DS4 reference |
|---|---|---|---|
| clean | 0.48 s | 1.0× | 1.0× |
| dirty-bottom | 0.55 s | **1.1×** | 0.7× |
| dirty-middle | 4.37 s | **9.1×** | 22.5× |
| dirty-top | 7.22 s | **15.1×** | 27.7× |

**H1 — CONFIRMED.** Identical ordering.

**H2 — CONFIRMED.** Absolute numbers differ; relative shape persists.

**H3 — relative form confirmed, absolute part MISSED.** top/bottom = 13.1× — below the
predicted 20–40× band; top/middle = 1.65 vs the positional model's ~1.3. Both deviations
are consistent with fixed costs compressing ratios at smaller context (12K vs 19K) —
exactly the interpretive difficulty §6 of THREATS anticipated.

**H4 — CONFIRMED.** Top mutation: divergence at token 16 → block 0 of 53 → **0.0 %**
reusable, 13,543 tokens re-prefilled. Bottom mutation: divergence at token 13,536 →
block 52 of 53 → **98.3 %** reusable, 231 tokens. The analyzer predicted the mechanism
correctly against a tokenizer it was never calibrated for.

### Conclusion

The locality shape replicated across a different model, tokenizer, chat template, parser,
FP4 backend, vLLM version, speculation method and topology. The common denominator is
vLLM's contiguous prefix cache. **Effect size is stack-dependent. The mechanism — and the
rule "static first, dynamic last" — is not.**
