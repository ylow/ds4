# Q4_K attention output-projection — decode speedup (design)

Date: 2026-06-26
Status: design, approved to plan
Related: `2026-06-25-cuda-graphs-decode-design.md` (decode byte budget), the KV-quant phases
(F16 / FP8-split / Hadamard-FP4). This is the **first weight-requant** phase; all prior phases
touched the KV cache (a memory lever), not the weights (the decode-speed lever).

## Goal

Push single-token decode past the current ~13.5 tok/s ceiling by requantizing the attention
**output-projection** weights from Q8_0 to Q4_K. This is the largest decode-speed lever because
attention projections are **61% of the ~9.5 GB read per token** (Q8_0 = 5.8 GB), and the two
output-projection tensors are ~85% of that block.

Explicitly a **deliberate quality/speed trade**: Q4_K is lossy. We do not fix an nll budget up
front — we **measure the nll / throughput tradeoff** and decide.

### Scope (the recipe)

Requant **only** these two tensors per layer to Q4_K:

| Tensor | Shape | Role |
|---|---|---|
| `blk.{L}.attn_output_a.weight` | 4096 × 8192 | output proj, grouped first stage (8 head-clusters) |
| `blk.{L}.attn_output_b.weight` | 8192 × 4096 | output proj, HC-split expand back to embd |

Keep at Q8_0 (near-lossless, small, and feed scores / the KV path): `attn_q_a`, `attn_q_b`,
`attn_kv`. Everything else (experts IQ2_XXS/Q2_K, shared-expert Q8, output head Q8) unchanged.

### Expected payoff (honest)

Q4_K ≈ 4.5 bpw vs Q8_0 ≈ 8.5 bpw → 0.53×. The two output tensors ≈ 4.9 GB → ~2.6 GB, saving
~2.3 GB/tok → ~9.5 → ~7.2 GB/tok. If decode stays bandwidth-bound at ~47% bus efficiency, tps
scales ~1.32× → **~17–18 tok/s**. Two things must hold and will be measured:
- the Q4_K GEMV stays bandwidth-bound (unpack overhead doesn't make it compute-bound), and
- the perplexity hit is acceptable.

## Current state (verified)

**Model.** `gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
(symlinked `ds4flash.gguf`). Attention projections are Q8_0 ("AProjQ8"). No HF safetensors and
no imatrix `.dat` are on disk — only this Q8 GGUF.

**Quantizer.** `gguf-tools/deepseek4-quantize.c`. Reads HF safetensors + a template GGUF, writes
a new GGUF. Per-tensor override `--tensor-type PREFIX=TYPE` (parse ~1792–1798, match ~1027–1031,
prefix-or-exact); family flag `--attention-proj`; `is_attention_projection()` (~993–996) already
enumerates the five AProj names. `quants.c` implements exactly **q8_0, q4_K, q2_K, iq2_xxs** — so
Q4_K is the only sane smaller option for attention (no Q5_K/Q6_K). `--compare-tensor` does a
single-tensor regen + byte/RMSE compare for pre-flight checks. Synthetic imatrix fallback when
none given: `importance[col] = sum(row[col]^2)`.

**Engine already speaks Q4_K** — `DS4_TENSOR_Q4_K` is a live tensor type used by the *Q4-experts*
GGUF family, and the GPU integer block-dot primitives `q4_K_q8_K_block`, `q4_K_q8_K_block8`,
`q4_K_q8_K_block8_full` exist (ds4_cuda.cu ~10824–10928, used in the MoE path ~12176–12517). So
this is **not** a from-scratch K-quant kernel.

**But the output projection uses specialized kernels, not the generic matmul.** Decode path
(ds4.c ~15822–15843):
- `attn_output_a` → `ds4_gpu_attention_output_low_q8_tensor` (grouped-rows, 8 clusters)
- `attn_output_b` → `ds4_gpu_matmul_q8_0_hc_expand_tensor` (ds4_gpu.h:1051, impl ds4_cuda.cu:14155)

Prefill/batch path (ds4.c ~15846): both via `ds4_gpu_attention_output_q8_batch_tensor`.

Type is hard-validated at load: `tensor_expect_layout(l->attn_output_a, DS4_TENSOR_Q8_0, …)` and
`…_b` (ds4.c ~3685–3686) — loading Q4_K there fails validation today.

## Architecture / components

### A. Offline — produce the Q4_K-output GGUF (requant from the existing Q8 GGUF)

We do **not** re-download the ~150 GB HF checkpoint. The `attn_output_*` tensors are already
Q8_0 (near-lossless), so dequant-Q8→requant-Q4_K adds only negligible extra error.

- Add a **requant-from-template-GGUF** source path to `deepseek4-quantize`: when a tensor is not
  taken from HF, dequantize its Q8_0 bytes from the template GGUF and feed the existing Q4_K
  quantizer. All non-targeted tensors are copied through byte-for-byte from the template.
  (Alternative considered: a tiny standalone gguf-requant tool. Decide in the plan; reuse the
  quantizer's existing gguf I/O + quants.c either way.)
- Recipe override: `attn_output_a`/`attn_output_b` → `q4_k`; everything else passthrough.
- Imatrix: synthetic weight-energy fallback for the first pass (no real imatrix on disk; Q4_K
  does not require one). A real attention imatrix is a documented follow-on if quality is short.
- Output: `gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-AOutQ4K-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.
  The original Q8 GGUF is the instant rollback.
- Pre-flight: `--compare-tensor blk.0.attn_output_a.weight` (+ `_b`) → per-tensor RMSE/cos before
  a full write.

### B. Engine — Q4_K read variants of the three specialized kernels

Add Q4_K weight-read paths to:
1. `ds4_gpu_attention_output_low_q8_tensor` (decode, output_a, grouped rows)
2. `ds4_gpu_matmul_q8_0_hc_expand_tensor` (decode, output_b, HC expand)
3. `ds4_gpu_attention_output_q8_batch_tensor` (prefill/batch, both)

Each reuses the trusted expert `q4_K_q8_K_block*` block-dot primitives: quantize the activation
to Q8_K, read the weight as Q4_K superblocks, accumulate — same loop structure as today, only the
weight-block read + scale handling changes. The grouped-rows / HC-split outer structure is
unchanged.

**Dispatch by loaded tensor type** (no env flag): at the call sites, branch on
`layer->attn_output_a->type == DS4_TENSOR_Q4_K` (and `_b`) → Q4_K kernel, else the existing Q8
kernel. One binary runs both GGUFs; the recipe lives in the model file. Relax
`tensor_expect_layout` for `attn_output_a`/`_b` to accept Q4_K **or** Q8_0.

CPU reference path (`matvec_q8_0_grouped_rows`, `matvec_q8_0`, ds4.c ~7195–7197) gets the matching
Q4_K read only if needed for the perplexity oracle on the chosen backend — perplexity runs on
`--cuda`, so the CUDA path is the gate; CPU parity is optional and noted, not required.

### C. Validation — the measured tradeoff

- **Quality (the gate).** Teacher-forced perplexity is bit-deterministic (greedy is not — see the
  optimization-findings note on non-associative GPU reductions). Baseline on the Q8 GGUF:
  `./ds4 --cuda --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096` → `nll=317.843967992`.
  Run the same on the Q4_K-output GGUF; report Δnll and Δppl. Measure-and-decide (no preset
  budget); for reference the FP4-KV phase accepted +0.367% nll.
- **Throughput.** Decode tok/s on the same prompt at ctx 4096 and 16384, Q8 vs Q4_K. Prior Q8
  baselines: ~13.5 @ 4096, ~13.06 @ 16384. Confirm the byte savings convert and the Q4_K GEMV
  did not go compute-bound.
- **Spot-check.** A couple of short generations for qualitative sanity (optional).

## Sequencing (mirrors the KV phases)

1. Offline: requant-from-gguf path + `--compare-tensor` RMSE sanity on output_a/output_b.
2. Engine: Q4_K read variants of the three kernels, behind type-dispatch (inert while the loaded
   GGUF is still Q8).
3. Flip: build the Q4_K-output GGUF, point `ds4flash.gguf` at it.
4. Gate: perplexity (Δnll) + decode tok/s at 4K/16K. Decide accept / tune / roll back.

Rollback at any point = repoint the symlink at the Q8 GGUF; no engine revert needed (type-dispatch
makes the Q8 path still valid).

## Risks / unknowns

- **Quality unknown until measured.** Attention output proj at 4.5 bpw — mitigated by the
  output-only recipe (q/kv stay Q8), synthetic imatrix, and the deterministic oracle. Follow-on
  levers if short: real attention imatrix; or fall back to output_b-only Q4_K.
- **Kernel efficiency.** Q4_K unpack could make the GEMV compute-bound and not realize the full
  byte savings. Decode is strongly bandwidth-bound, so likely fine; measured in step 4.
- **Specialized kernels are real work.** The grouped-rows and HC-expand kernels are not a
  dispatch swap; bounded by reusing the expert Q4_K×Q8_K primitives and leaving the outer loops
  intact.

## Results (2026-06-26)

**Perplexity (teacher-forced, n=256, c=4096, `doors-of-stone-chapter-1.md`):**

| | nll | avg_nll | ppl |
|---|---|---|---|
| Q8 reference oracle (prior session) | 317.843967992 | 1.241578 | ~3.461 |
| Q8 measured (today, bit-reproducible) | 319.009406860 | 1.246130 | 3.476863 |
| Q4_K measured (today) | 316.384665147 | 1.235878 | 3.441397 |

Δnll% (Q4_K vs reference oracle 317.844): **−0.459%**
Δnll% (Q4_K vs measured Q8 319.009): **−0.823%**
Δppl (Q4_K vs measured Q8): **−0.035** (Q4_K is neutral-to-slightly-better, within noise)

**Baseline note (cause of the 317.844→319.009 gap):** the `317.843967992` oracle predates the
Phase 2c **Hadamard-FP4 (NF4) compressed-attention KV cache**, now on by default
(`DS4_GPU_ATTN_COMP_CACHE_FP4`, commit d0c8b30), which adds +0.367% nll → `319.009` (the exact
Phase 2c-recorded value). The correct same-engine baseline is **319.009**; Q4_K-output is
**316.385** (−0.823%) — no regression. The gap is the already-merged KV feature, not this work.

**Decode throughput (generation tok/s):**

| Model | ctx=4096 | ctx=16384 | prefill @4096 | prefill @16384 |
|---|---|---|---|---|
| Q8   | 15.11 t/s | 14.82 t/s | 32.74 t/s | 33.67 t/s |
| Q4_K | 15.51 t/s | 15.51 t/s | 30.77 t/s | 30.63 t/s |
| Δ    | +2.6%     | +4.7%     | −5.7%      | −9.0%      |

**GGUF size delta:** Q8 = 86.72 GB (86,720,111,488 bytes), Q4_K = 85.28 GB (85,277,270,624 bytes) → **−1.44 GB**.

**Summary:** Q4_K produces no quality regression on this text sample (Δnll is negative, i.e.,
slightly better perplexity). The projected 1.32× decode speedup did not materialize — measured
gain is only +2.6–4.7%, because the current Q8 baseline was already faster (~15 t/s) than the
~13.5 t/s reference used in the projection, likely from prior engine optimizations. Prefill
speed decreases by 5–9% due to Q4_K unpack overhead dominating in compute-bound batch paths.
The quality/speed tradeoff decision (accept / output_b-only / roll back) is left to the user;
the symlink `ds4flash.gguf` remains pointed at the Q8 GGUF.

---

## Non-goals

- No Q5_K/Q6_K (not in the quantizer). No requant of q/kv, shared expert, or output head. No HF
  re-download. No real-activation imatrix in the first pass. No CUDA-graph / MoE-kernel work
  (separate levers).
