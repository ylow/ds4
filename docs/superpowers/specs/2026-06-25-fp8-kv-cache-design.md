# Phase 2b design: FP8 (split) compressed-attention KV on CUDA

**Date:** 2026-06-25
**Goal:** push the dominant KV term — the 512-dim compressed *attention* rows
(`layer_attn_comp_cache`) — below F16 to **FP8 E4M3 + per-row scale**, cutting that
cache ~1.77× vs F16 (~3.5× vs F32). This is the decode byte-budget lever (attention
projections are 61% of the ~9.5 GB/token read budget) and the memory lever toward a
usable 600–800K context. Precursor to TurboQuant (~3-bit), which reuses this layout.

Builds directly on Phase 2a (F16 comp-attn KV via **expand-on-read**, commit 08ad8e5,
and indexer-KV F16, commit d648b6d). Only the storage dtype, the dequant kernels, and
the write-commit change; the trusted F32 attention kernels stay byte-identical.

## Decisions (locked with the user)

- **Sequence:** FP8 this session; TurboQuant later (reuses this row layout).
- **Read path:** extend **expand-on-read** (not in-kernel reads). Lowest risk — the
  F32 attention kernels + the gather-of-selected decode path stay unchanged; only the
  expand/gather helpers and the write staging change.
- **RoPE handling:** **split row** — NoPE 448 dims → FP8+per-row scale, RoPE 64-dim
  tail (`DS4_N_ROT`=64) kept **F16**. De-risks RoPE up front and pre-sets the
  mixed-dtype row layout TurboQuant needs. (head_dim=512, n_rot=64 ⇒ NoPE=448; RoPE is
  the tail dims `rope_tail_layer_inplace` rotates.)
- **Indexer 128-dim cache:** untouched — stays F16 (separate cache, low-dim, already
  the long-context bandwidth path).
- **Correctness gate / tolerance:** teacher-forced perplexity. F32 ref nll=317.843968;
  current F16 build=317.842979 (256 tok, ctx 4096). **Accept FP8 if total nll ≤ ~319.4
  (< ~0.5% avg_nll drift).** Flag NaN, crash, or a larger jump as a bug. NB: per the Key
  Discovery below, this storage is actually **near-lossless** (NoPE bit-exact to the F32
  reference, RoPE F16 like the current build) — realistically expect nll ≈ 317.843,
  between the F32 ref and the current F16 build. The 0.5% bound is just a safety net; a
  drift anywhere near it would itself signal a codec/scale bug.

## Key discovery — the model already E4M3-quantizes the NoPE dims

The model's defined numerics (`dsv4_fp8_kv_quantize_row_inplace_cpu`, ds4.c:2489; CUDA
`fp8_kv_quantize_kernel`, ds4_cuda.cu:4465) **already** round the **NoPE 448 dims** to
E4M3 — in **7 groups of 64**, each with a **power-of-2 scale**
`scale = 2^ceil(log2(max(amax,1e-4)/448))` — and leave the **64-dim RoPE tail
untouched**. The F32 *reference* (nll=317.843968) bakes this rounding in; the current
F16 cache stores `F16(e4m3_dequant(x)*scale)`.

⇒ A genuine FP8 store that captures the **E4M3 magnitude index + per-group exponent**
reconstructs `e4m3_value(idx) * 2^k` in **F32 exactly** = bit-exact to the reference's
NoPE values (no extra loss; it actually drops the F16 rounding the current cache adds to
NoPE). RoPE stays F16 exactly like today. **So Phase 2b is near-lossless like F16 — not
0.5%-lossy.** The 0.5% gate stays only as a safety bound; expect nll ≈ 317.843 (between
the F32 ref and the current F16 build). The scale scheme is **dictated by the model**
(per-64 group, power-of-2) — not a free granularity choice.

## Storage layout (per layer, FP8 mode)

Replaces the single F16 `layer_attn_comp_cache[il]` with three parallel row-indexed
buffers, `layer_comp_cap[il]` rows each:

| buffer | dtype | stride/row | holds |
|---|---|---|---|
| `layer_attn_comp_cache[il]` | **E4M3 byte** (model codec) | 448 B (`n_nope`) | NoPE dims [0..447]: `(sign<<7) \| magnitude_index` |
| `layer_attn_comp_rope[il]`  | **F16** | 64 elems (`n_rot`) | RoPE tail dims [448..511] |
| `layer_attn_comp_scale[il]` | **int8 exponent** | 8/row (7 used, 1 pad) | per-64-group power-of-2 exponent `k` |

Per row: 448 + 128 + 8 = **584 B** vs F16 1024 B (~1.75×; ~3.5× vs F32 2048 B).

**Codec is the model's own E4M3FN** (`dsv4_e4m3fn_value_dev` / the binary-search rounding
in `dsv4_e4m3fn_dequant_dev`), NOT a generic hardware `__nv_fp8_e4m3` convert — matching
the model's tie-breaking is what makes NoPE bit-exact. The stored byte is the magnitude
index `best` (0..126) the model's rounding already selects, plus the sign bit. The
exponent `k = (int)ceil(log2(max(amax,1e-4)/448))` (int8; range ≈ [-30,+10]); dequant uses
`exp2f((float)k)`. The 448 NoPE dims are exactly 7×64, so the group split is clean.

## Read path — extend expand-on-read

Two new kernels reconstruct the full 512-dim **F32** row into the *same reused F32
scratch* (`g_comp_f32_expand_*`) the F16 path uses, so every downstream F32 attention
kernel sees byte-identical input:
- `expand_comp_fp8_to_f32` (prefill: all `n_comp` rows)
- `gather_comp_fp8_to_f32` (decode: only the `n_sel` ≤ top_k selected rows — keeps the
  per-token expand bounded at long ctx, mirroring `gather_comp_f16_to_f32_kernel`)

Per output row, group `g = d/64`, exponent `k = scale_exp[row*8 + g]`:
- dims [0..447] = `(byte&0x80 ? -1 : 1) * dsv4_e4m3fn_value_dev(byte & 0x7f) * exp2f((float)k)`
  where `byte = cache[row*448 + d]`
- dims [448..511] = `__half2float(rope_f16[row*64 + (d-448)])`

Codec is the **model's** `dsv4_e4m3fn_value_dev` (already `__device__`, ds4_cuda.cu:4374),
NOT a hardware `__nv_fp8_e4m3` convert — matching the model's rounding is what makes NoPE
bit-exact.

The 5 extern attention entry points that do expand-on-read
(`ds4_gpu_attention_decode_heads_tensor` 8957, `_decode_mixed_batch` 9248,
`_indexed_mixed_batch` 9283 [gather + expand], `_prefill_static_mixed` 9565,
`_prefill_masked_mixed` 9593) generalize the current `uint32_t comp_kv_f16` bool →
**`uint32_t comp_kv_dtype`** (0=F32, 1=F16, 2=FP8) and gain two params
`const ds4_gpu_tensor *comp_rope`, `const ds4_gpu_tensor *comp_scale` (NULL unless
dtype==FP8). They reset dtype→0 after expanding, so the inner launches
(`attention_decode_batch_launch` etc.) are untouched — they only ever see F32. The 8
comp-passing ds4.c call sites pass `metal_graph_attn_comp_cache_dtype()` +
`g->layer_attn_comp_rope[il]` + `g->layer_attn_comp_scale[il]`. This is the bulk of
commit 1's churn (the F16 plumbing was the same shape).

## Write path — extend the F32-staging commit (only the commit changes)

**Leave every compressor + QAT call untouched.** Staging stays F32: the compressor
pools/normalizes/RoPEs and the existing `ds4_gpu_dsv4_fp8_kv_quantize_tensor` QAT round
runs in `attn_comp_stage` exactly as today, so the staged NoPE is already
`e4m3_dequant(x/s)·s`. Only the **commit** (the FP8 branch of
`metal_graph_store_attn_comp_stage`) changes: a new helper
`ds4_gpu_tensor_quantize_f32_to_fp8split` runs a kernel that, per row, **re-runs the
model's per-64-group quantizer** (`amax → k = ceil(log2(max(amax,1e-4)/448))`, byte =
sign + `dsv4_e4m3fn` magnitude index of `clamp(x/2^k, ±448)`) and writes:
- E4M3 bytes → `layer_attn_comp_cache[il]` (448/row, at `first_row*448`)
- int8 exponents → `layer_attn_comp_scale[il]` (8/row, at `first_row*8`)
- F16 RoPE tail → `layer_attn_comp_rope[il]` (64/row, at `first_row*64`; reuse
  `f32_to_f16_kernel`)

**Value-exactness despite re-quantizing already-dequantized input:** re-running the
quantizer can pick a finer exponent (k−1) when the dequantized group max rounds down an
octave, but the e4m3 magnitude index then doubles, so `decode = e4m3(idx)·2^k` reproduces
the staged value **bit-for-bit** (an e4m3 magnitude ×2 is exactly e4m3; values stay in
range, no clamp). The reconstructed F32 equals the reference's stored value — no reliance
on a "canonical" exponent. This avoids touching the ~4 write sites and the prefill
compressor's `quantize_fp8` path; `metal_graph_attn_comp_{update_target,update_row,
row_view,prefill_target}` keep returning the F32 staging exactly as in F16 mode.

Session save/load + the cache-trace gain an FP8 branch alongside the existing F16/F32
ones: save/load go through F32 on disk (decode FP8→F32 on save, encode F32→FP8 on load —
or, simplest and equally correct, keep the stable F32 disk format by expanding the FP8
cache to F32 for save and re-running the commit quantizer on load). The trace decodes
FP8→F32 before diffing against the CPU `attn_comp_kv`.

## Flag & plumbing

New compile-time macro **`DS4_GPU_ATTN_COMP_CACHE_FP8`** next to the F16 one
(ds4.c ~10296). When 1 it **takes precedence** over `DS4_GPU_ATTN_COMP_CACHE_F16` (the
comp cache is FP8-split). Default **0 on all backends**; flipped on **for CUDA only** at
the end, revertible independently like the F16 macros. Allocation, staging-target
selection, save/load, and the memory estimate all branch FP8 → F16 → F32.

## Memory accounting

Update the comp-cache byte term in **three** places (all currently `DS4_N_HEAD_DIM ×
(F16?2:4)`): `ds4_context_memory_estimate_with_prefill` (ds4.c:21607-21609) **and** the
managed-KV policy estimator `metal_graph_kv_cache_bytes_for_context` (ds4.c:10793-10794)
**and** the staging-bytes terms (21622-21623 / 10822-10823 — staging stays F32 so those
are unchanged, but the FP8 cache needs its own 584 B/row term). FP8-split row term =
`n_nope·1 + 8 + n_rot·2` = 448 + 8 + 128 = **584 B/row**. Expected at ctx 4096: ~405.5 →
**~396 MiB** — the comp cache drops from F16 21.0 MiB (1026 rows × 512 × 2 B × 21 layers)
to FP8-split 12.0 MiB (1026 × 584 B × 21), a ~9.0 MiB saving. Verify the measured line
matches; treat a mismatch as an estimate bug. **Pre-existing bug to fix while here:**
`metal_graph_kv_cache_bytes_for_context` (10796) hard-codes `sizeof(float)` for the
*indexer* comp term — it does NOT branch on `DS4_GPU_INDEX_COMP_CACHE_F16`, so the
managed-KV policy already over-estimates the indexer cache 2× on CUDA (the main estimator
at 21613 does branch). Bring it in line.

## Incremental commits — each gated on the perplexity oracle

`./ds4 --cuda --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096`

1. **Read/expand plumbing inert** — new FP8 expand/gather kernels + `comp_kv_dtype` /
   sidecar-param signature change + call sites, all behind the OFF macro. Gate: nll
   **unchanged = 317.842979** (F16 still the live path), tree builds.
2. **Write staging + sidecar buffers inert** — alloc FP8/rope/scale per layer, the
   `ds4_gpu_tensor_quantize_f32_to_fp8split` commit (FP8 branch of
   `metal_graph_store_attn_comp_stage` only — compressor/QAT untouched), the FP8
   save/load + cache-trace branches. Macro still OFF. Gate: nll **unchanged**.
3. **Flip macro ON for CUDA** + memory-estimate updates. Gate: nll **≤ 319.4**, no
   NaN/crash, "context buffers" line drops as predicted.

Finish with `make cuda-spark` so `ds4-server` is rebuilt too. Build during iteration
with `make ds4 ds4-bench CUDA_ARCH=` (ds4.c relinks fast; ds4_cuda.cu re-runs nvcc).

## Gotchas (carried from Phase 2a / the brief)

- The existing `fp8_kv_quantize_kernel` / `dsv4_fp8_kv_quantize_row_inplace_cpu` are
  **fake-quant** (round-trip f32→fp8→f32 in the SAME F32 buffer) — as a *storage* path
  they save nothing. BUT their per-64-group scale + E4M3 rounding **is** the model's
  numeric definition, so the new commit kernel re-implements exactly that math while
  emitting (E4M3 byte, int8 exponent) to narrow buffers — that is what makes the store
  genuine AND bit-exact. Reuse the `__device__` `dsv4_e4m3fn_value_dev` (4374) and the
  rounding in `dsv4_e4m3fn_dequant_dev` (4381); the latter returns a float — write a
  sibling that returns the magnitude index `best` for encoding.
- 4 session save/load sites (23914 save-layer, 24126 load-layer, 24621 save, 24968 load)
  + 1 cache-trace site (21766) need an FP8 branch alongside the F16/F32 ones.
- ONE ds4 process at a time (instance lock); never `pkill -f ds4-server` — kill by exact
  PID. Don't `make cpu` (overwrites CUDA binaries). Greedy decode is NOT a valid oracle
  (run-to-run nondeterministic); only teacher-forced perplexity is bit-deterministic.
- Decode is GPU-bound at benchable ctx, and the indexer bandwidth win only showed at
  hundreds-of-K ctx — so the **near-term FP8 win is MEMORY**, not tok/s at 4K–25K.

## Status — LANDED (bit-identical to F16, ~1.75× less cache)

Commits: `582c3ad` read plumbing (inert), `c8da47c` write/commit/buffers/save-load/trace
(inert), `bf7c814` flip-on + the exact-exponent fix, `53b8e7e` memory accounting + reload
codec + indexer over-estimate fix. Built with `make cuda-spark` (ds4-server rebuilt).

**Result: teacher-forced nll = 317.842979423 @ ctx 4096 — BIT-IDENTICAL to the F16
build.** The FP8-split comp-attn KV is exactly as accurate as F16 (F16 is itself lossless
on the model's E4M3-quantized NoPE values), at **396.21 MiB context buffers vs 405.53
(F16) @ ctx 4096** (the comp cache: F16 21.0 MiB → FP8-split 12.0 MiB; ~9 MiB saved,
linear in ctx → ~0.6 GB @128K, multi-GB toward 800K). So this is a **memory** win at zero
quality cost, not the 0.5%-lossy tradeoff originally budgeted.

**The validation bug (worth remembering).** The first flip-on gave nll 317.158649 — a
0.68 drift that *passed* the ≤319.4 gate but contradicted the predicted bit-identity.
Root cause: the commit re-quantizes the already-QAT'd F32 staging and recomputed the
per-64-group exponent with `(int)ceilf(log2f(amax/448))`. When a group's max equals the
E4M3 max (448), `amax/448 == 2^k` exactly but `log2f(2^k)` returns `k + 1ulp`, so `ceilf`
yields `k+1`; the scale doubles and small already-quantized values shift down into the
E4M3 subnormal grid, losing **2⁻¹⁶** on a handful of values. That tiny seed compounds
chaotically over the perplexity's 255 teacher-forced **decode** steps (the comp cache is
rebuilt from the model's own hidden states — a recurrence), inflating to 0.68 nll.
Methodology lesson: a passing tolerance gate is NOT proof of correctness here; the comp
cache must be verified **value-exact** (decode == staged value, bit-for-bit), because the
recurrent decode amplifies sub-ulp differences. **Fix:** compute the exponent with
`frexpf` (exact `ceil(log2)`, no overshoot), which provably yields `k ≤` the staging's own
exponent (shift-up only) so decode reproduces the staged value exactly. Applied to both
the commit kernel and the CPU session-load codec.

**Follow-ons (unchanged priority):** in-kernel `__half`/FP8 reads for the *attention* comp
cache (recover read bandwidth, drop the expand pass — the indexer already does this);
raw-KV → F16 (lossless, low value); then TurboQuant (~3-bit) on the 512-dim rows, which
reuses this split row layout + per-group scale buffers.
