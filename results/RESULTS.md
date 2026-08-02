# Results — measured 2026-08-01/02

All figures were measured within the same night on the same rig. Where a conclusion later
turned out to be wrong, it is kept together with its correction — see §7.

## Setup

| | |
|---|---|
| Hardware | 2× NVIDIA DGX Spark (GB10, sm_121), 128 GB unified LPDDR5X per node |
| Interconnect | RoCE, direct-attached, asymmetric port mapping (node 1 port 1 ↔ node 2 port 0) |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, 156 GB |
| Recipe | `tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark` @ `d728faee` |
| Runtime | `vllm-dspark-runtime:dspark-nvfp4-stage-c`, vLLM `0.21.1rc1.dev339+g1967a5627bc3` |
| Topology | TP=2, PP=1, `nvfp4_ds_mla` KV, block-size 256 |
| Speculation | DSpark, k=5 static |
| GPU memory utilization | 0.78 |

Notes that save others time:

- **The image build takes ~20 seconds, not 30–60 minutes** — the recipe overlay is pure
  Python on a prebuilt base image; no NVCC compilation happens.
- **"Patch 4"** (shared-expert gate_up_proj) **is already baked into the recipe overlay**
  since 2026-07-31. Do not apply it again — verify the file hash instead. Without it,
  draft acceptance collapses to ~26% and decode roughly halves; dropped tensors are
  logged at DEBUG level only.

## 1. Production status

- `Using 'B12X' Mxfp4 MoE backend` confirmed active (without it: ~60 → ~29 tok/s per the recipe)
- Correctness: **40/40** — code 10/10, Swedish 10/10 with correct diacritics,
  tool calls **10/10** well-formed and parseable, ~12K needle search 10/10

Decode, warm, `stream:false`, median of three:

| Content | Decode | Draft acceptance |
|---|---|---|
| Code / structured | **62.5–68.1 tok/s** | **71.0%** |
| English prose | 35.3 tok/s | 28.4% |
| Swedish prose | 30.3 tok/s | 19.5% |
| Mixed (3 code + 2 prose) | 50.1 tok/s | — |

Code acceptance of 71.0% exceeds the recipe's own reference (68.7%), proving the baked-in
Patch 4 is active. Per-position acceptance: 74.5 / 52.9 / 37.4 / 28.1 / 21.6%.

**Swedish text costs ~9 percentage points of draft acceptance and ~14% decode versus
English**, measured with content-matched prompt pairs. This is a property of the drafter,
not a configuration error — consistent with
[Speculative Decoding Across Languages (arXiv:2605.30580)](https://arxiv.org/abs/2605.30580).

## 2. Context depth

3 needles per depth (at 8%, 50%, 92% of the document), with distractor codes.

| Depth | Prompt tokens | Hits | TTFT | Prefill |
|---|---|---|---|---|
| 32K | 33,089 | 3/3 | 0.30 min | 1,810 tok/s |
| 128K | 132,041 | 3/3 | 1.25 min | 1,758 tok/s |
| 512K | 529,151 | 3/3 | 6.74 min | 1,308 tok/s |
| 900K | 929,733 | **3/3** | 15.24 min | 1,017 tok/s |

**12/12 needles.** Full 1M retrieval works — something the recipe itself had not
demonstrated. Prefill decays with depth (attention's O(n²)).

> **1M is working capacity for batch analysis, not an interactive mode.** A 900K prefill
> takes a quarter of an hour. Nobody waits for that inside an agent loop.

## 3. Three engines — the most important systems finding

An LLM server is not one resource but three nearly independent ones. Same machine, same
configuration, same hour:

| Engine | Bounded by | Scales with concurrency? |
|---|---|---|
| **Prefill** | compute | **No** — aggregate completely flat |
| **Decode** | memory bandwidth + batching | **Yes** — +125% to N=6 |
| **Admission** | scheduling | **No** during long prefill |

### Prefill concurrency (32K per session, end-to-end incl. prefill)

| N | per stream | aggregate | worst latency |
|---|---|---|---|
| 1 | 6.8 tok/s | 6.8 | 22.1 s |
| 2 | 3.2 | 6.4 | 40.1 s |
| 4 | 1.7 | 6.6 | 77.4 s |
| 6 | 1.3 | **6.5** | 115.2 s |

Aggregate varies 5% while per-stream falls by 5.2×. **Pure serialization.** A single 32K
prefill saturates the system.

### Decode concurrency (short unique prompt, 1,200-token generation)

Swedish-prose content (worst case):

| N | aggregate | per stream | TTFT max | acceptance |
|---|---|---|---|---|
| 1 | 15.2 | 15.3 | 0.2 s | 21.5% |
| 2 | 22.0 | 11.8 (77%) | 0.4 s | 18.9% |
| 4 | **31.5** | 8.3 (54%) | 0.4 s | 23.4% |
| 6 | 34.2 | 6.4 (42%) | 1.7 s | 23.2% |

Code content:

| N | aggregate | per stream | acceptance |
|---|---|---|---|
| 1 | 48.0 tok/s | 48.0 | 62.3% |
| 2 | 72.9 | 38.1 | 72.6% |
| 4 | **126.3 tok/s** | 34.0 | **71.3%** |

**Saturation around N≈4.** Acceptance holds ~71% under concurrency — speculation does not
degrade with parallel streams.

### Admission during long prefill

Three short requests started 25 s into a 516,565-token prefill:

```
short requests:  367.5 s · 367.7 s · 367.7 s      (normal warm TTFT: 1.66 s)
long job:        516,565 tok in 391.2 s
```

**221× normal latency.** All three were released when the prefill finished. Ongoing decode
is simultaneously throttled to **~1/35 of normal speed** — not frozen, but starved.

> This does **not** apply to short prompts: at N=4 with short prompts, TTFT is 0.4 s.
> The blocking is specifically tied to long prefill saturating compute.

## 4. Chunk size — MAX_NUM_BATCHED_TOKENS

Measured at 256K prefill, `--async-scheduling` off in both cases.

| Metric | 8192 | 2048 |
|---|---|---|
| Prefill | 1,529 tok/s | 1,469 tok/s (**−3.9%**) |
| p95 token gap during prefill | 5.197 s | **1.590 s** (−69%) |
| max token gap | 6.417 s | 2.085 s |
| Decode share during prefill | 7.1% | 7.3% (**unchanged**) |
| TTFT of a new short request | 150.7 s | 158.1 s (unchanged) |
| **KV pool** | 1,598,763 tok | **2,671,557 tok** (+67%) |
| Max concurrency @1M | 1.59× | **2.55×** |

**The mechanism is confirmed:** the token gaps *are* the prefill chunks.
`8192 ÷ 1529 = 5.36 s` vs measured p95 5.20 s. At 2048: `2048 ÷ 1469 = 1.39 s` vs 1.59 s.

**But chunk size is a jitter lever, not a fairness lever.** Decode share is pinned at ~7%:
at 8192 decode fits ~9 tokens per chunk, at 2048 ~2.3 — four times more opportunities,
four times fewer tokens each, net zero. The scheduler gives decode a *fixed share of the
token budget*, and that share cannot be changed by changing the budget's size.

vLLM's chunked-prefill design states pending decodes should be prioritized and mixed with
prefill — this stack deviates from that stated intent. See [`repro/`](../repro/) for a
minimal reproduction.

**Conclusion:** 2048 buys much better jitter and a 67% larger KV pool for 4% prefill.
It does not fix fairness or admission. It is still a strong agent profile.

## 5. Prefix cache locality

Same context sent repeatedly with ~200 tokens of growth per turn. Variants differ **only
in where a mutating field sits**.

| Variant | TTFT mean (turns 2–8) | vs clean |
|---|---|---|
| clean (nothing mutates) | 0.65 s | 1.0× |
| **dirty-bottom** (mutates last) | **0.45 s** | **0.7×** |
| dirty-middle | 14.64 s | 22.5× |
| **dirty-top** (mutates first) | **18.05 s** | **27.7×** |

`dirty-bottom` is not merely close to clean — it is *faster*, because the mutation lands in
its own partially filled block. **Volatile fields at the end are effectively free.**

Verified with `prompt_locality.py` against the server's own tokenizer — identical
16,839-token prompt, only the timestamp's position differs:

```
TIMESTAMP AT TOP                  TIMESTAMP AT END
first divergence: token 13        first divergence: token 16,834
reusable:  0 (0.0%)              reusable:  16,640 (98.8%)
re-prefill: 16,839 (100%)        re-prefill: 200 (1.2%)
extra cost: ~11 s/turn            extra cost: ~0.1 s/turn
```

With `--block-size 256` the outcome is **binary, not gradual**: a mutation in the first
block yields zero reuse. A changed token never costs "one token" — it costs at least 256,
plus everything after it.

> **Design rule: static first, dynamic last.** System prompt, tool definitions and stable
> history byte-identical at the top; timestamps, current status and fresh tool results at
> the end. This is established vendor guidance — the numbers above are what it is worth
> on this stack.

### Cross-stack replication (Laguna S 2.1)

The same test, unchanged, on a substantially different stack — different model (67 GB NVFP4
vs 304B MoE), tokenizer, chat template, parser, FP4 backend (FLASHINFER_CUTLASS vs B12X),
vLLM version (0.25.1 upstream vs 0.21.1 DSpark), speculation (DFlash k=15 vs DSpark k=5)
and topology (single node vs TP=2):

| Variant | Laguna (12K ctx) | DS4 (19K ctx) |
|---|---|---|
| clean | 1.0× | 1.0× |
| dirty-bottom | **1.1×** | 0.7× |
| dirty-middle | **9.1×** | 22.5× |
| dirty-top | **15.1×** | 27.7× |

Form replicated; effect size is stack- and context-size-dependent (fixed costs compress the
ratios at smaller context). Token-level mechanism verified on Laguna too: top mutation →
divergence at token 16 → block 0 → 0.0% reuse; bottom mutation → 98.3% reuse.
Pre-registered hypotheses, including one honestly failed prediction:
[`HYPOTHESES-laguna-replication.md`](HYPOTHESES-laguna-replication.md).

## 6. UMA memory gate — systematic reproduction of vLLM #48140

Laguna S 2.1's 262K production profile **does not boot** on this node. Three boots, three
nearly identical reported budgets regardless of process history, JIT state and memory
start state:

| Boot | Reported available KV | Requirement (262K) | Outcome |
|---|---|---|---|
| cold JIT | 4.22 GiB | 18.35 GiB | deterministic refusal |
| warm JIT | 5.18 GiB | 18.35 GiB | deterministic refusal |
| warm JIT, 32K profile | 5.34 GiB | ~2.3 GiB | **boots** (67,236-token KV pool) |

Mechanism ([vLLM #48140](https://github.com/vllm-project/vllm/issues/48140), closed "not
planned"): the startup check effectively reads Linux `MemFree`, so ~20 GB of reclaimable
page cache (from reading the weights!) is booked as unavailable. The page cache has a dual
role: the same pages *speed up* weight loading (8.75 s/shard vs 11.5 cold) and *lower* the
reported KV budget.

Operational profiles: 32K–64K boots normally (keep the page cache); 128K–262K requires
`drop_caches` before start (**the proposed workaround remains unverified on this
deployment**) or a local UMA patch of `gpu_worker.py` to use `MemAvailable`.

### Startup profile (three starts measured, Laguna)

| Phase | Time |
|---|---|
| FP4 JIT, cold kernel cache | ~24 min (one-time; cache in `~/.cache/vllm`, persistent) |
| Weight loading | ~12–13 min regardless of start state; first ~10 shards slower (UVM ramp) |
| Profiling/capture/autotune | ~2–4 min |
| **Cold total** | **~39 min** |
| **Warm total (to API)** | **13 min 16 s measured** |
| Ready (generation 1) | +5.1 s |
| Warm (generation 2) | 3.5 s |

The weight loader is single-threaded (101% CPU) — the floor for every restart, cache or
not. Readiness should be measured to the first successful generation, not the open port.

## 7. Conclusions we corrected after better measurement

Kept because they are as useful as the results.

1. **A max-gap test misread partial starvation as "green".** Our first overlap measurement
   looked for a stall > 20 s; none occurred, so the script printed "decode continued" —
   while decode was throttled to 1/35 of normal. **Measure p50/p95 token intervals.**
2. **GPU-mem-util 0.70 as a "safety measure" was destructive.** Weights take 77.7 GiB of a
   ~119.6 GiB node; at 0.70 the KV budget fell to 2.39 GiB — insufficient for even 32K.
   **When weights dominate memory, GMU is a KV lever, not a safety lever.**
3. **`MAX_NUM_SEQS=1` does not boot with k=5.** CUDA-graph sizes must be multiples of k+1.
4. **The prefix cache contaminated a concurrency test** (two long jobs with identical text;
   job two got a cache hit instead of prefilling). **Salt every prompt.**
5. **`rsync -aL` on a HF cache doubles it** (~311 GiB instead of ~155): `-L` dereferences
   the snapshot symlinks. Use `-a`.
6. **Admission blocking is not "less serious" than decode starvation.** An agent loop is
   `model → tool → model`; every arrow is a new API call.
7. **`pkill -f` self-matches — even across SSH**, where the pattern sits in the remote
   shell's command line. Three incidents in one night. Use `[b]racket` patterns or PID files.
8. **Do not judge model loading or memory from process RSS on GB10** — CUDA/UVM
   allocations don't appear there. Read system `free` and the engine's own logs.
9. **Do not read an instantaneous ETA as a trend**, and **compare the same phase between
   runs** — early-shard rate vs steady-state rate produced a false 2.6× conclusion.

## 8. Known limitations

- The prefill concurrency test is prefill-dominated by design and says nothing about
  decode concurrency (hence the separate test).
- Reference decode in the overlap tests was measured on a cool server (23–24 tok/s); both
  configurations were measured under equal conditions, so the comparison holds.
- Whether two concurrent prefills share bandwidth is **unanswered** (cache contamination).
- No long soak yet. Xid errors during the session: 0.
- Quality gain vs the previous model (Artificial Analysis 40 → 50) is the vendor's figure
  at maximum reasoning effort; our production setting (`thinking=false`) is unmeasured.
- The old production model's admission behavior was never measured — the ~7% decode share
  may be inherited rather than new.

## 9. Real agent workload (Hermes v0.19.0)

A real agent task through the capture proxy against the 2048 agent profile: find and fix a
planted bug in a small Python project, run the test suite, handle the failing test,
summarize. The agent solved it correctly (bug found: `len(stock)` → `sum(stock.values())`;
4/4 tests passing) in 10 model calls with 20 tool definitions and a ~15–18K-token growing
prompt. A second task (string-conversion bug class) completed correctly in **316 s**.

**Prompt locality across agent turns (server tokenizer, 256-token blocks):**

| Transition | Prompt size | First divergence | Reusable prefix | Re-prefill |
|---|---|---|---|---|
| turn 1→2 | 15,169 tok | token 15,168 (last) | **97.1%** | 453 tok |
| turn 5→6 | 16,549 tok | token 16,548 (last) | **98.7%** | 213 tok |
| turn 9→10 | 17,655 tok | token 17,654 (last) | **97.5%** | 439 tok |

**The agent framework is already cache-optimal.** Prompts are built append-only: no
mutating timestamps, no reordered tool lists, no rewritten history. Each agent step
re-prefills only the new tail (~200–450 tokens ≈ 0.2 s) instead of the full context
(~10 s). This is the null-result counterpart to the synthetic dirty-top experiments: the
tool's value here was *verifying* cache health, not finding a problem.
