# Threats to Validity

What could make the conclusions in `RESULTS.md` wrong, or less general than they look.
Each item names the experiment that would settle it.

## 1. The calibration curve covers one runtime profile

`PREFILL_CURVE` in the analyzer was measured on DS4-0731 with **chunk 8192**, then used to
estimate cost in a test running **chunk 2048**. Measured consequence: a **1.7× error**
(prediction 9.3 s, reality 18.05 s). The curve also has **no data point below 32K** — it
extrapolates with a constant in exactly the range where the tool is used most.

> **Settling experiment:** a short-context matrix 2K → 5K → 10K → 20K → 32K → 64K → 128K at
> the target profile; three repetitions, unique prompts, one output token, median **and**
> spread. Calibrate against time directly — `T(n) = a + bn + cn²` — not via tok/s, since
> fixed costs dominate below 10K and masquerade as throughput.

## 2. The positional model is an approximation, not a runtime law

`1 − (p/n)²` models **causal attention and nothing else**. Outside the model: MoE routing,
TP communication over RoCE, chunking, block allocation, cache eviction, kernel choice,
scheduler overhead. That it matched our measured `middle/top` ratio (0.811 measured vs
0.75–0.80 predicted) is support, not proof. **One data point.**

> **Settling experiment:** several mutation positions — 10, 25, 50, 75, 90 % — to see
> whether the whole curve follows `1 − (p/n)²` or merely happens to fit at 50 %.

## 3. Prompt locality is measured on vLLM's prefix cache

The mechanism assumes the engine reuses **contiguous prefix blocks from the start**. That
holds for vLLM. It does not necessarily hold for engines without a prefix cache, for
radix-/tree-based caching (SGLang) where branches are shared, for llama.cpp-style
per-conversation KV sequences, or for sliding-window attention.

> **Settling experiment:** the same test on an engine with a different cache architecture.
> A **counter-example is worth more than a third confirmation** — it marks where the model's
> validity ends, and the boundary is what makes it useful.

## 4. Two model stacks are replication, not generalization

DeepSeek-V4-Flash-0731 and Laguna S 2.1 differ in model, tokenizer, chat template, parser,
FP4 backend, vLLM version and topology — but share the vLLM prefix-cache principle. A held
pattern is **strong replication**; it is not evidence for all inference engines.

## 5. Measurement-specific weaknesses

| Weakness | Consequence |
|---|---|
| Prefill concurrency test is prefill-dominated (32K prompt, ~150 generated tokens) | Says nothing about decode concurrency — hence the separate test |
| Decode baseline used Swedish prose, our worst content | Absolute numbers are low; the **scaling shape** is the result |
| Reference decode measured cool (23–24 tok/s) | Both configurations measured equally; comparison holds |
| Second long job in one test hit the prefix cache | Bandwidth sharing between two prefills is **unanswered** |
| `dirty-bottom` can be affected by the chat template | If the template appends text after the user message, the mutation is no longer last |
| No soak run | Long-term stability untested; session Xid count: 0 |
| Recipe claims (Stage C 584-byte envelope, k=3 ≈ −24 %) | Taken from the recipe, **not independently verified** |

## 6. A pre-registration flaw (noted, not rewritten)

H3 predicted a top/bottom ratio of 20–40×. **Its absolute part cannot be interpreted
cleanly** without Laguna-specific time calibration — a miss cannot be attributed to the
model vs the calibration, making the outcome uninformative rather than falsifying. This
should have been stated in the pre-registration. The hypothesis is left unchanged; the
flaw is recorded here instead. (Outcome: measured 13×, below the band — attributed to
fixed-cost compression at smaller context, which is an interpretation, not a test.)

## 7. What is NOT threatened

Worth separating, so the list above does not read as "everything is uncertain":

- **Cache invalidation is deterministic.** First divergence, block index and reusable
  fraction follow directly from tokenization and block size.
- **The three-engine separation is directly measured**, not modeled: prefill aggregate
  flat over N=1→6, decode +125 %, admission blocked. Same machine, same hour.
- **`MAX_NUM_BATCHED_TOKENS` → KV pool** is read from the boot log, not derived.
- **Correctness results** (40/40, 12/12 needles, two solved agent tasks) are counted
  outcomes.
- **The #48140 reproduction** is three deterministic boots with logged budgets.
