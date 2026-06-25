# Phase 2a execution plan: F16 KV cache on CUDA (memory + long-context speed)

**Date:** 2026-06-25
**Goal:** cut KV-cache memory ~2× (toward the 600–800K-context goal) and reduce long-context
KV read bandwidth by storing the compressed attention KV (and later raw + indexer KV) as F16
instead of F32. Near-lossless. Precursor to FP8/TurboQuant (Phase 2b).

## What's already there vs the gap

- Macro **`DS4_GPU_ATTN_COMP_CACHE_F16`** (ds4.c:10296) = 1 on Apple/Metal (production), 0 on CUDA.
  ~25 sites in ds4.c (alloc, `attn_comp_stage` staging, save/load payload) already compile on CUDA.
- Probe result: enabling the macro for CUDA links after adding **one** helper
  (`ds4_gpu_tensor_copy_f32_to_f16` — trivial, reuse `f32_to_f16_kernel` at ds4_cuda.cu:3681),
  and **memory really drops** (ctx 4096 ctx-buffers 432→410 MiB; scales up a lot at long ctx).
- **The real gap:** NO CUDA attention kernel reads comp-KV as `__half`. The `comp_kv_f16` flag is
  threaded for Metal parity but every CUDA site bails (`if (comp_kv_f16) return 0;`). So the macro
  is **all-or-nothing**: it can't be enabled until every comp-KV reader supports `__half`.
  Confirmed failure with macro on: `attention batch encode failed` at layer 2 (first ratio-4 layer)
  prefill.

## The 7 kernels that read `comp_kv` (must gain a `__half` read path)

Scalar reads (`comp_kv[idx*head_dim + d]` — easy to template):
- `attention_prefill_mixed_kernel` (ds4_cuda.cu:4531) — reads at 4572, 4605
- `attention_decode_mixed_kernel` (4778) — 4854, 4875, 4930, 4940
- `attention_indexed_mixed_kernel` (4946) — 5040, 5090, 5100

float4-vectorized reads (`(const float4*)(comp_kv + ...)` — need a half-vectorized replacement,
e.g. load `int4`=8×`__half` and convert; these are the head_dim==512 hot path):
- `attention_indexed_mixed_heads8_rb4_kernel` (5106) — 5203, 5251
- `attention_indexed_mixed_heads8_online_kernel` (5282) — 5379
- `attention_static_mixed_heads8_online_kernel` (5447) — 5503
- `attention_decode_mixed_heads8_online_kernel` (5571) — 5668

Dispatch note: at SMALL n_comp the scalar `attention_decode_mixed_kernel` is used; the float4
heads8 kernels kick in at LARGE n_comp (`!cuda_attention_score_buffer_fits`) / head_dim==512. So the
scalar kernels alone let us validate F16 correctness at small context first.

## Approach

1. **Device load helper** (templated on comp dtype) so each kernel reads float or half uniformly:
   ```
   template<bool F16> __device__ __forceinline__ float ld1(const void* p, uint64_t i);
   template<bool F16> __device__ __forceinline__ void   ld4(const void* p, uint64_t i, float out[4]);
   ```
   F16 path: `__half2float`; float4 path reads `int4` (8 halves) → 8 floats (two ld4).
2. **Template each of the 7 kernels** on `bool COMP_F16` (and instantiate both). Change the
   `comp_kv` param to `const void*`. Dispatch picks the instantiation from `comp_kv_f16`.
   Keep raw_kv F32 for now (separate later step).
3. Add `ds4_gpu_tensor_copy_f32_to_f16` in ds4_cuda.cu (see probe; ~12 lines).
4. **Keep the macro OFF while editing** (all new code inert, repo keeps building/behaving
   identically). Flip `DS4_GPU_ATTN_COMP_CACHE_F16` for CUDA only once all 7 kernels are done:
   `#if defined(__APPLE__) || (!defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU))`.

## Validation (gate)

- **Correctness:** teacher-forced perplexity must stay within a small tolerance of the F32 ref
  `nll=317.843967992` (F16 is lossy, expect ~±0.0X, NOT bit-identical; flag if >~1 nll or NaN/crash).
  Test small ctx first (scalar kernels), then large ctx (heads8 float4 kernels), via
  `./ds4 --cuda --perplexity-file <txt> -n 256 -c <ctx>`.
- **Memory:** confirm ctx-buffer MiB drop; measure at 128K/256K with a long prompt.
- **Speed:** ds4-bench decode tok/s at long ctx — expect a *gain* (less KV bandwidth in
  attention; the indexer KV is still F32 until its own step).
- Regression: SSD-streaming + CPU build unaffected (macro gated to CUDA/Metal).

## Follow-on steps (separate, after comp-attn-KV lands)

- **Indexer KV (128-dim) → F16**: the long-context indexer-scoring bandwidth hotspot; the WMMA
  indexer kernels already `__float2half` internally, so storing F16 lets them read half directly.
  6 indexer kernels (indexer_scores_kernel + 4 WMMA variants + indexer_score_one_direct).
- **Raw KV → F16**: already F16-rounded (lossless); read by the same attention kernels (extend the
  template to raw_kv too).
- **Phase 2b — FP8 / TurboQuant (~3-bit)** on the 512-dim compressed rows: add parallel per-row
  scale buffers + dequant in the read kernels. Keep RoPE-key and 128-dim indexer higher precision.

## Status — Phase 2a LANDED (commit 08ad8e5)

Compressed attention KV is now F16 on CUDA, via **expand-on-read** (lower risk than the 7-kernel
in-kernel __half rewrite above): the 5 extern attention entry points dequantize the F16 comp rows
to a reused F32 scratch (`cuda_expand_comp_f16_to_f32` + `f16_to_f32_kernel`) and run the existing
F32 kernels unchanged; write side uses `ds4_gpu_tensor_copy_f32_to_f16`. Validated:
- Correctness: teacher-forced perplexity nll 317.843968 (F32) → 317.842979 (F16), ~3e-6 relative
  (pure F16 rounding). Near-lossless.
- Memory: ctx-buffers 432 → 410 MiB at ctx 4096 = exactly the ratio-4 comp cache halving
  (1026 rows × 512 × 2 bytes × 21 layers); scales to ~4.3 GB saved at 800K.
- Speed: ~neutral at normal context (13.36 → 13.22 tok/s @2048; 12.90 @8192). Expand overhead
  grows at extreme length.

### Remaining follow-ons (in priority order)
1. **gather-of-selected** expand (indexed layers): DONE (commit 5dc12e6).
2. **Indexer KV (128-dim) → F16**: **DONE** (commits b4b8e63 read plumbing, 450d513 write staging,
   d648b6d flip-on). See "Indexer KV → F16 LANDED" below.
3. **Raw KV → F16**: already F16-rounded (lossless).
4. **In-kernel __half comp reads** for the *attention* comp cache (the templating plan above):
   recovers attention read bandwidth and removes the expand pass entirely. (The indexer cache now
   uses exactly this in-kernel-__half approach — it scores all rows so expand-on-read would save no
   bandwidth, so reading __half directly was the right call there.)
5. **Phase 2b — FP8 / TurboQuant ~3-bit** on the 512-dim compressed rows (parallel scale buffers).

## Indexer KV → F16 LANDED (commits b4b8e63, 450d513, d648b6d)

The 128-dim indexer-compressed KV cache is now F16 on CUDA, gated by a dedicated macro
**`DS4_GPU_INDEX_COMP_CACHE_F16`** (CUDA only; Metal/ROCm/CPU stay F32). Unlike the attention cache
(expand-on-read), the indexer cache uses **in-kernel `__half` reads**: the indexer scores ALL visible
compressed rows every step, so expand-on-read would save no bandwidth — reading `__half` directly in
the 6 scoring kernels both halves the read bytes and lets the WMMA scorers skip the `float`→`__half`
convert they already did.

Implementation, in three bit-/near-lossless-gated steps:
- **Read side** (b4b8e63): threaded a runtime `comp_f16` flag through `indexer_scores_kernel`,
  `indexer_score_one_direct_kernel`, and the 4 WMMA scorers (`index_comp` is now `const void*`;
  `ld_index_comp_f32`/`ld_index_comp_h` device helpers), their `indexer_scores_launch` dispatcher
  (element-width-aware size check), and the 3 extern entry points + the 4 ds4.c call sites. Metal
  defs accept and ignore the flag.
- **Write side** (450d513): mirrored the attention F16 staging — `index_comp_stage` (F32) +
  `metal_graph_{store,commit}_index_comp_stage` / `_index_comp_{update_target,update_row,row_view,
  prefill_target,prefill_target_free}`; routed all indexer-cache writes (single-token decode,
  prefill zero-prefix, aligned-chunk replay, per-token unaligned) through them (compressor + QAT run
  in the F32 stage, then commit `copy_f32_to_f16`); handled F16 in session save/load + the cache
  trace.
- **Flip on** (d648b6d): macro on for CUDA + F16 cache allocation + memory-estimate/policy fixes so
  reported "context buffers" stays accurate.

Validated:
- **Correctness:** teacher-forced perplexity `nll=317.842979423` — **bit-identical** to the
  F32-indexer baseline (the F16 score rounding didn't change the top-k; n_index_comp=1026 > top_k=512
  at ctx 4096, so selection is genuinely exercised). Near-lossless.
- **Memory:** ctx-buffers 410.29 → 405.53 MiB at ctx 4096; 689.04 → 663.22 MiB at ctx ~20.5K
  (−25.8 MiB). Scales linearly with context.
- **Speed:** decode neutral at benchable contexts (16384: 13.10→13.06; 20480: 12.97→12.92 tok/s,
  within noise) — at ≤20K the indexer reads are <1% of the ~9.5 GB/token budget. The bandwidth win
  only becomes material at hundreds-of-K context (indexer reads grow to ~1 GB/token at 800K), which
  the 25K-token bench prompt can't reach. Primary value here and now is **memory** (fitting 600–800K).
