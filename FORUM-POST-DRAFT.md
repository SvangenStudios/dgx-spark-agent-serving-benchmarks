# Agent Serving on DGX Spark: DeepSeek V4 Flash 0731 — KV-cache trade-offs, mixed-load scheduling and UMA pitfalls

*(Draft for forums.developer.nvidia.com → DGX Spark / GB10 Projects)*

---

We spent a night characterizing **DeepSeek V4 Flash 0731** (vLLM/DSpark, TP=2 over RoCE)
on 2× DGX Spark — not for peak tok/s (60–67 on code is already well documented here), but
for the things that decide whether an agent workload is actually pleasant to run: KV-cache
capacity, scheduling fairness under mixed load, prompt-cache locality, and startup
behavior. Everything is reproducible:

**Repo (scripts, configs, full results, threats-to-validity):**
https://github.com/SvangenStudios/dgx-spark-agent-serving-benchmarks

**Scope.** These results characterize one DeepSeek/DSpark deployment on 2× DGX Spark.
The mixed-load scheduler behavior has not yet been reproduced on unmodified upstream vLLM,
the Laguna 262K workaround remains unverified, and the 72-hour soak is still pending.

Three findings we think are worth your time:

## 1. `max_num_batched_tokens` 8192 → 2048 is a surprisingly strong Pareto move

Same model, same night, same rig:

| Metric | 8192 | 2048 |
|---|---|---|
| KV pool | 1.60M tokens | **2.67M tokens (+67 %)** |
| p95 token gap during a 256K prefill | 5.20 s | **1.59 s (−69 %)** |
| Prefill | 1,529 tok/s | 1,469 tok/s (−3.9 %) |
| Decode share during prefill | 7.1 % | 7.3 % (unchanged) |

Smaller chunks free activation memory that vLLM's profiler hands straight to the KV pool.
On a 128 GB unified-memory box that's the difference between 1.59× and 2.55× max
concurrency at 1M context — for 4 % prefill.

## 2. Chunk size is a jitter lever, not a fairness lever (likely scheduling deviation)

The token gaps during prefill are exactly `chunk_size ÷ prefill_rate` — decode gets one
scheduling opportunity per chunk. But the decode *share* is pinned at ~7 % regardless of
chunk size, and **new requests are not admitted until the prefill finishes** (measured
221× normal TTFT). vLLM's chunked-prefill design says pending decodes should be
prioritized; this DSpark/speculative stack deviates from that stated intent. Minimal
reproduction in the repo (`repro/`). Practical consequence: **1M context is batch
capacity, not an interactive mode** — we run separate agent (≤128K) and batch profiles.

Notably: prefill does not scale with concurrency at all (aggregate flat N=1→6), while
decode scales to ~4 streams (126 tok/s aggregate on code, draft acceptance holding 71 %
under concurrency).

## 3. If your model "should fit" but vLLM refuses to build the KV pool — it's #48140

Laguna S 2.1's 262K profile would not boot: three boots, three nearly identical reported
budgets (4.22 / 5.18 / 5.34 GiB available KV) against an 18.35 GiB requirement —
regardless of JIT state and process history. Mechanism: on UMA, vLLM's startup check
effectively reads Linux `MemFree`, so ~20 GB of reclaimable page cache (from reading the
weights!) is booked as unavailable ([vLLM #48140](https://github.com/vllm-project/vllm/issues/48140),
closed "not planned"). The same pages *speed up* weight loading and *shrink* the reported
budget. Workarounds in the repo; the `drop_caches` route is documented but not yet
verified on our deployment.

## Bonus findings

- **Prompt cache locality, quantified:** a mutating field (timestamp) at the *top* of a
  17K-token prompt costs 28–40× more per turn than the same field at the *bottom*
  (0 % vs 98.8 % reusable prefix — binary at 256-token block granularity). Replicated on
  two very different vLLM stacks. The repo ships a standalone capture proxy +
  block-aware locality analyzer that works with any OpenAI-compatible client — we ran it
  against a real agent framework and verified 97–99 % cache reuse per turn.
- **Multilingual speculative decoding, field data:** draft acceptance 71 % (code) /
  28 % (English prose) / **19.5 % (Swedish prose)** — consistent with arXiv:2605.30580,
  now with concrete GB10 numbers.
- **Measurement discipline** that cost us real hours: `stream:false` for speculative
  stacks, p50/p95 token intervals instead of max-gap (partial starvation looks "green"
  otherwise), don't trust process RSS on GB10, and salt every prompt or you're measuring
  the prefix cache.

## What this is not

We didn't invent prefix caching, didn't discover the UMA bug, and 60–67 tok/s on code was
published here before us. The contribution is the controlled quantification, the
cross-stack replication with pre-registered hypotheses (including one honestly failed
prediction), and the reusable tools. Threats-to-validity is a first-class document in the
repo — including what is *not* threatened.

Happy to answer questions, and very interested in whether anyone sees the pinned ~7 %
decode share on other stacks. Follow-ups planned in this thread: `drop_caches` causality
test for #48140, and a 72 h soak.
