# Measurement correction — 2026-08-02

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
| cold reference including TTFT (÷1.2) | overstated |
| no prefill window (undisturbed decode included) | overstated, factor unknown |

A corrected value is expected in the **~8–15%** range, but the point of this notice is that
**the number is not known** until re-measured. It remains well below the 25% threshold the
test used for a pass, so the qualitative finding — decode is severely starved during long
prefill — is not in question. The exact percentage is.

## What must be treated as under revalidation

- The decode-share values **7.1% and 7.3%**.
- **The comparison between them.** Our first instinct was that both runs carried the same
  bias, so the comparison survived even if the absolutes did not. That does not hold:
  tokens per SSE chunk may differ between chunk sizes, the unbounded window diluted each
  run by a different amount, and the two reference calls were cold to different degrees.
  The supporting argument ("~9 tokens per chunk at 8192 vs ~2.3 at 2048") is stated in
  absolute terms and is not yet token-weighted either.
- The derived statement that decode is throttled to **~1/35 of normal speed**.

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
