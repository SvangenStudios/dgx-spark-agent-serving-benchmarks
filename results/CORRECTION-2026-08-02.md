# Measurement correction — 2026-08-02

> **Note on commit hashes.** The git history was rewritten on 2026-08-02 to correct the
> commit author identity. Every commit hash changed. Hashes cited in this document and in
> `raw/ab-256k-2026-08-02.log` refer to the current history; if you fetched the repository
> before that rewrite, the hashes you saw will not resolve. Contents are unchanged.

A follow-up review of `scripts/prefill_decode_overlap.py` found **three independent faults**
in how decode share during prefill was measured. The reported values **7.1% and 7.3% are
invalid** and are being re-measured. A fourth fault, in `scripts/prompt_locality.py`,
affects the absolute token counts for prompts that contain tool definitions.

This document states exactly what is wrong, what follows from it, and what is untouched.
Corrected figures will be published in v0.1.1.

---

## Fault 1 — SSE events were counted as tokens

The script timestamped every SSE chunk with a non-empty `delta.content` and computed
`1 / mean(gap)`. That is **events per second**. It was then divided by a reference
expressed in **tokens per second**.

On this speculative-decoding stack one SSE chunk carries several accepted tokens. Measured
on the same server, 300 generated tokens arrived in 118–120 chunks:

```
tokens per chunk: {1: 31, 2: 36, 3: 25, 4: 14, 5: 6, 6: 6}
mean tokens per non-empty chunk: 2.50 – 2.54
usage.completion_tokens: 300     chunks: 118
```

The event rate therefore understated the token rate by roughly **2.5×**.

This trap was already documented in this repository's own measurement-discipline list
("streamed output represented speculative decode steps rather than accepted output
tokens") — and this script violated it. The lesson is recorded rather than quietly fixed.

## Fault 2 — there was no prefill window

```python
during = [g for g in gaps]     # a no-op copy of every gap in the stream
```

The stream is started 5 s before the long request is submitted and continues after the
prefill has completed. Every one of those undisturbed intervals was included in
"decode DURING prefill", diluting the measurement with unimpeded decode.

## Fault 3 — the reference was cold and included TTFT

The reference was a non-streaming call, `completion_tokens ÷ total_request_time`, and it
was the **first** request the script issued. Measured on the same server:

| | tok/s |
|---|---|
| cold, total time incl. TTFT (the method used) | 17.1 |
| warm, total time incl. TTFT | 20.9 |
| warm, first token → last token (pure decode) | 21.3 |

TTFT accounts for only ~1.9% of the error. The rest is **GPU clock ramp**: the reference
was measured on a cold GB10. The repository's own discipline rule — "three short calls are
not warm" — applies here too. Net effect: the reference was understated by ~18%, which
inflated the reported decode share.

## Fault 4 — `prompt_locality.py` did not use the server's chat template

`flatten()` built its own `<|role|>content` string instead of asking the server to apply
the real chat template. On the same agent-style body with one tool definition:

```
server /tokenize, chat mode (messages + tools + add_generation_prompt):  338 tokens
old flatten():                                                            93 tokens
```

The template expands a single tool definition to ~266 tokens; `flatten()` produced ~21.
`flatten()` also drops `assistant` messages whose content is `None` and whose payload is
`tool_calls` — i.e. it cannot see a divergence inside a tool call at all.

---

## Net effect on the reported decode share

The three faults do not point the same way:

| Fault | Effect on the reported 7.1% |
|---|---|
| events counted instead of tokens (×2.5) | understated |
| cold reference including TTFT (÷1.65) | overstated |
| no prefill window (undisturbed decode included) | overstated, factor unknown |

We estimated the corrected value would land in the **~8–15%** range. It landed at **5.0%**
on 2048 and **1.7%** on 8192 — outside the estimate, in the direction that makes the finding
stronger rather than weaker. The third fault dominated: the unbounded window mixed the
undisturbed head and tail of a 600-token stream into a measurement of a 179-second prefill.

That estimate is kept here rather than deleted. It is the point of the exercise: with three
faults of opposing sign, the corrected number could not be reasoned out, only measured.

## Re-measured: the A/B, and an overturned conclusion

The full A/B was re-run on 2026-08-02 with the repaired instrument — three repetitions per
profile, plus three more for 2048 after a server restart to rule out warm-up effects.

| | published | corrected |
|---|---|---|
| Decode share during prefill, 8192 | 7.1% | **1.7%** (1.6 / 1.7 / 1.8) |
| Decode share during prefill, 2048 | 7.3% | **5.0%** (4.5 / 4.8 / 5.0 / 5.0 / 5.1 / 5.2) |
| Prefill cost of 2048 | −3.9% | **−7.7%** (1,462 vs 1,584 tok/s) |
| p95 output-chunk gap, 8192 → 2048 | 5.20 → 1.59 s | 6.10 → 1.64 s |
| Decode throttled to | "~1/35 of normal" | 1/59 (8192), 1/20 (2048) |

**The headline conclusion is overturned.** We published that decode share is invariant to
chunk size — "chunk size is a jitter lever, not a fairness lever". It is not invariant:
**2048 delivers 2.9× the decode share and 3.1× the decode tokens** of 8192 during the same
prefill. The fairness benefit we measured as zero is in fact the largest single effect of
the setting.

The reasoning that produced the wrong conclusion was: "at 8192 decode fits ~9 tokens per
chunk, at 2048 ~2.3 — four times more opportunities, four times fewer tokens each, net
zero." Half right. Decode does get one scheduling opportunity per chunk. But the step
yields ~`k` accepted speculative tokens regardless of chunk size — measured 3.3 per chunk
at 8192 and 2.6 at 2048, not 9 and 2.3 — so total decode scales with the *number* of
chunks, not their size.

Note the direction: the corrected decode shares are **lower** than published, i.e. the
starvation is worse than we reported, and the corrected prefill cost of the 2048 profile is
**higher**. The correction makes both sides of the trade-off sharper, and still favours
2048 for agent workloads.

Three faults with opposing signs is why no corrected number could be reasoned out in
advance. Our own written estimate before measuring was "~8–15%"; the answer was 5.0%.

### Provenance caveat

The nine runs were taken with the instrument at commit `ffce6f1`, which did **not** yet
have four integrity guards added afterwards:

- exceptions raised inside the worker threads were discarded by `threading.Thread`, so a
  `TokenCountMismatch` or `MissingTokenIds` in the ongoing stream would have vanished
  silently instead of failing the run;
- a missing `usage.completion_tokens` skipped the cross-check instead of refusing it;
- the admission result was labelled from the *intended* 20-second delay rather than the
  short request's actual submission timestamp;
- the margin by which the ongoing stream outlived the window was not reported, so a run
  that only just survived looked identical to one with plenty of headroom.

None of these is likely to have corrupted the results — the nine runs are internally
consistent to ±0.35 percentage points and the per-run token counts cross-checked against
`usage` at the time — but "likely" is not the standard this repository claims. The raw log
is committed with this caveat rather than presented as a fully hardened measurement.
Verification runs with the hardened instrument are listed under **Pending** below.

### What was verified as unchanged

- Undisturbed decode baselines were 36.3–39.0 tok/s in **both** profiles — the precondition
  for the comparison to mean anything.
- The 2048 profile reproduces across a server restart (4.8 / 5.0 / 5.0 warm vs
  4.5 / 5.1 / 5.2 after restart), so uptime and cache warmth do not confound it.
- KV pool re-read from the restarted server's own startup accounting: 2,669,829 tokens
  against 2,671,557 published, 0.06% apart. The 8192 KV figure was not re-verified — that
  container no longer exists.
- Admission blocking: 145.4 s (8192) and 159.2 s (2048) against a 1.66 s warm baseline,
  released only on prefill completion. Chunk size does not fix admission; the difference
  between the profiles is just how long the prefill takes.

## What has already been re-measured: the agent captures (§9)

The Hermes bodies were re-analyzed against the server's real chat template. The captures
carry no `chat_template_kwargs`, `continue_final_message` or reasoning settings — the only
template-relevant fields are `messages` and `tools` — so nothing further needs forwarding.

| Transition | Published | Corrected | Re-prefill (was → is) |
|---|---|---|---|
| turn 1→2 | 97.1% | **95.7%** | 453 → 663 tok |
| turn 5→6 | 98.7% | **98.4%** | 213 → 266 tok |
| turn 9→10 | 97.5% | **98.1%** | 439 → 357 tok |

**A prediction we got wrong.** When the fault was found we argued the corrected figures
could only move *up*: divergence sits at the last token of each prompt, and

```
reuse ≈ (len − 256 − remainder) / len
```

grows monotonically with `len`, so a longer true prompt means higher reuse. That reasoning
ignored the other half. `flatten()` also omitted serialized `assistant` `tool_calls`, and
those sit in the *appended tail* — the part being re-prefilled. For turn 1→2 the newly
visible tail outweighed the longer prefix and reuse fell, 97.1% → 95.7%. Two of three
transitions moved as predicted; one did not.

The qualitative finding is unchanged and is now on firmer ground: the first divergence
lands on the last token of the previous prompt in **every** transition, so the client is
genuinely append-only.

## What is unaffected

Measured without the event/token ratio, and unchanged:

- **KV pool +67%** (1,598,763 → 2,671,557 tokens) — read from the runtime's own startup
  accounting.
- **p95 output-chunk gap 5.197 s → 1.590 s (−69%)** and max gap 6.417 s → 2.085 s. These
  are wall-clock intervals between output chunks — user-visible jitter — and are valid as
  measured. Note the naming correction below.
- **Prefill −3.9%** (1,529 → 1,469 tok/s), from `usage.prompt_tokens` on non-streaming
  requests.
- **Admission delay**: new short requests at 367.5 / 367.7 / 367.7 s against a 1.66 s warm
  baseline, released only when the prefill completed. TTFT is the timestamp of the first
  output chunk, which is also the first token — unaffected by how many tokens later chunks
  carry.
- **The chunk-gap mechanism**: observed gaps closely tracked
  `chunk_size ÷ effective_prefill_rate` — `8192 ÷ 1529 = 5.36 s` predicted vs 5.20 s
  measured, `2048 ÷ 1469 = 1.39 s` vs 1.59 s. Close, not exact; independent of token
  counting.
- **Prefill vs decode concurrency scaling**, computed from `usage` token counts.
- **Retrieval 12/12** at 32K/128K/512K/900K with no distractor hits.
- **The GB10 UMA reproduction of vLLM #48140** across three boots.
- **The synthetic prefix-cache locality experiments on both stacks.** `cache_locality.py`
  sends a single plain-text `user` message with no tools, so the chat template wraps all
  four variants identically and fault 4 does not apply. The §5 `prompt_locality.py`
  verification likewise used plain text, not a chat body.
- **Draft-acceptance figures** (71% code / 28.4% English prose / 19.5% Swedish prose), read
  from the runtime's metrics.

## Naming correction

What the tables call a "token gap" or "token interval" is the interval between **output
chunks**, not between tokens. With ~2.5 tokens per chunk these differ. The measurements are
correct; the label was not. Renamed to **output-chunk gap** throughout.

## What is being changed

1. `prefill_decode_overlap.py` requests `return_token_ids: true` and aborts loudly if
   `token_ids` comes back null — no silent fallback to event counting. Tokens are counted
   even when `delta.content` is empty.
2. The prefill window is defined explicitly: from submission of the long request to the
   arrival of its first output token. Only chunks inside that window count.
3. The reference is a warm, streamed, token-weighted decode rate measured from first token
   to last token, after a warm-up generation.
4. Token throughput and output-chunk jitter are reported as two separate metrics and are no
   longer combined.
5. `prompt_locality.py` uses the server's `/tokenize` in chat mode with `messages`, `tools`
   and `add_generation_prompt`. It aborts if the server does not support it;
   `--allow-approximate` restores the old behavior with an explicit warning.
6. Every measurement prompt is salted — reference, ongoing stream and short request, not
   just the large prefill. An unsalted prompt hits the prefix cache on the second
   repetition and measures the cache rather than the model.
7. Regression tests cover all four faults, and `scripts/probe_token_ids.py` is a standalone
   artifact that answers "can this server's stream be counted?" with an exit code.

A second review pass found three more weaknesses in the repaired instrument, before it was
ever used for a real measurement:

8. **The baseline is now the same stream, measured against itself.** The ongoing generation
   runs undisturbed for 25 s before the prefill is submitted, and that window is its
   baseline. A separate reference request — however warm — still differs in prompt content,
   draft acceptance, sequence length, clock state and incidental load. The standalone warm
   reference is retained as a diagnostic line only.

   This is not a theoretical refinement. On a 60K verification run both were measured at
   once:

   ```
   undisturbed baseline (same stream):  30.39 tok/s  ->  decode share 8.2%
   separate warm reference:             37.01 tok/s  ->  decode share 6.8%
   ```

   A 20% relative difference on a single run, in the direction that flatters the finding.
   Whatever the corrected 256K numbers turn out to be, they must state which baseline they
   used.
9. **The ongoing stream must outlive the prefill window.** It previously generated at most
   2,000 tokens; a 256K prefill runs for minutes, and if decode were *not* starved the
   stream would finish first, leaving tokens to stop accruing while the divisor kept
   running. The ceiling is now 8,000 and the script aborts with `StreamEndedEarly` rather
   than reporting an artificially low rate.
10. **Output-chunk gaps are clipped to the window, not filtered into it.** Selecting samples
    inside the window and then differencing them silently discards the interval that
    straddles the boundary — the stall at the moment the prefill begins, typically the
    largest visible pause of the run. Every adjacent interval is now considered and its
    overlap with the window taken.

Filing the scheduling issue is deferred until the minimal reproduction runs on the repaired
instrument.

## Pending before this is called final

- [x] Token-weighted A/B on both profiles, three repetitions each
- [x] Raw log committed with provenance (`results/raw/ab-256k-2026-08-02.log`)
- [x] Integrity guards: thread-exception propagation, hard refusal on missing `usage`,
      admission judged from real timestamps, stream margin reported
- [x] Verification repetition on **2048** with the hardened instrument: **5.1%**
      (348 tokens, 1,455 tok/s prefill, 180.1 s window, 160.1 s admission delay,
      31.5 s stream margin). Reproduces the 5.0% median from the six earlier runs. No
      worker errors, no missing-usage refusals, no margin warning — every guard passed.
- [x] Verification repetition on **8192** with the hardened instrument: **1.5%**
      (96 tokens, 1,563 tok/s prefill, 167.6 s window, 147.6 s admission delay, 27.5 s
      stream margin). Reproduces the 1.7% median from the three earlier runs. No worker
      errors, no missing-usage refusals, no margin warning.

**Both profiles are now confirmed on the hardened instrument.**

| | headline result | hardened verification |
|---|---|---|
| 8192 | **1.7%** median of 3 runs (1.6 / 1.7 / 1.8) | 1.5%, one run |
| 2048 | **5.0%** median of 6 runs (4.5 – 5.2) | 5.1%, one run |
| A/B effect | **2.9×**, from the multi-run medians | 3.4×, from the verification pair |

**The headline figure is 2.9×, from the multi-run medians.** The 3.4× from the
verification pair rests on a single run per profile and is reported as independent support,
not as the result.

The 8192 verification came in slightly below the original arm — 1.5% and 96 tokens against
1.6–1.8% and 105–107 — which is a small shift, not a contradiction. What matters is that no
worker errors were caught, `usage` was present and matched the counted tokens, the ongoing
stream had 27.5 s of margin past the window, and the value sits far closer to 1.7% than to
5.0%. The separation between the profiles is undisturbed.

We chose **not** to spend two further server restarts on additional 8192 repetitions. Three
tightly clustered runs plus one hardened verification is sufficient, provided the provenance
caveat above stays visible — which is why it is stated rather than retired.

## Re-measurement protocol

`/reset_prefix_cache` **does not exist on this build** — it returns HTTP 404, and the
server's OpenAPI document lists no cache-reset path (only `/tokenize`, `/detokenize`,
`/v1/messages/count_tokens`). Salting is therefore the only per-request cache reset
available short of restarting the server. Note that Linux `drop_caches` is a different
cache entirely and does nothing here.

Per configuration:

1. Start the profile (8192 or 2048). Switching between them requires a restart, so each
   profile's first run already begins with an empty prefix cache.
2. Run a long warm-up generation — GB10 clocks ramp, and a cold reference reads ~18% low.
3. Confirm the server is idle (no other clients on the port).
4. Run `probe_token_ids.py` and require exit 0.
5. Run the overlap test. All prompts are salted, so repetitions do not reuse cache.
6. Repeat at least three times per profile and report the **median and spread**, not a
   single run.

The short request fires 20 s after the long request is submitted. That is only an
admission test if the prefill is still running at that point — at 256K it takes minutes,
but on a small prefill the window closes first and the resulting TTFT says nothing about
admission. The script now labels that case explicitly in its output rather than leaving
the reader to notice.

Switching between the 8192 and 2048 profiles requires a server restart (~13 min warm
load). On this cluster DS4 carries production agent traffic, so the A/B belongs in a
planned maintenance window, not an opportunistic run.
