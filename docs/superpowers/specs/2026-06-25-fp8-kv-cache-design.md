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
  (< ~0.5% avg_nll drift).** Flag NaN, crash, or a larger jump as a bug. (F16 was
  near-bit-identical; FP8 E4M3's 3-bit mantissa is meaningfully lossy.)

## Storage layout (per layer, FP8 mode)

Replaces the single F16 `layer_attn_comp_cache[il]` with three parallel row-indexed
buffers, `layer_comp_cap[il]` rows each:

| buffer | dtype | stride/row | holds |
|---|---|---|---|
| `layer_attn_comp_cache[il]` | **FP8 E4M3** | 448 B (`n_nope`) | NoPE latent dims [0..447] |
| `layer_attn_comp_rope[il]`  | **F16** | 64 elems (`n_rot`) | RoPE tail dims [448..511] |
| `layer_attn_comp_scale[il]` | **F32** | 1 | per-row scale over the 448 NoPE dims |

Per row: 448 + 128 + 4 = **580 B** vs F16 1024 B (~1.77×; ~3.5× vs F32 2048 B). Scale
F32 (4 B is negligible; avoids scale-rounding loss). **Per-row** scale granularity to
start; **per-group** (e.g. 2×224) is the documented fallback knob if the 0.5% gate fails.

## Read path — extend expand-on-read

Two new kernels reconstruct the full 512-dim **F32** row into the *same reused F32
scratch* (`g_comp_f32_expand_*`) the F16 path uses, so every downstream F32 attention
kernel sees byte-identical input:
- `expand_comp_fp8_to_f32` (prefill: all `n_comp` rows)
- `gather_comp_fp8_to_f32` (decode: only the `n_sel` ≤ top_k selected rows — keeps the
  per-token expand bounded at long ctx, mirroring `gather_comp_f16_to_f32_kernel`)

Per output row:
- dims [0..447] = `float(__nv_fp8_e4m3 q[row*448+d]) * scale[row]`
- dims [448..511] = `__half2float(rope_f16[row*64 + (d-448)])`

Hardware E4M3 convert via `<cuda_fp8.h>` (`__nv_fp8_e4m3`), supported on GB10/sm_121/CUDA 13.

The 5 extern attention entry points generalize the current `uint32_t comp_kv_f16` bool
→ **`uint32_t comp_kv_dtype`** (0=F32, 1=F16, 2=FP8) and gain two params
`const ds4_gpu_tensor *comp_rope`, `const ds4_gpu_tensor *comp_scale` (NULL unless
dtype==FP8). Their ds4.c call sites pass the per-layer sidecars. This is the bulk of
commit 1's churn (the F16 plumbing was the same shape).

## Write path — extend the F32-staging commit

Staging stays F32 (full 512-dim rows; compressor + QAT run unchanged in
`attn_comp_stage`). The commit replaces the `ds4_gpu_tensor_copy_f32_to_f16` call (FP8
mode only) with a new kernel `quantize_f32_to_fp8_rows`:
- per row: `maxabs` over the 448 NoPE dims → `scale = maxabs / 448` (E4M3 max ≈ 448;
  guard `maxabs==0` → scale=1) → write 448 FP8 = `__nv_fp8_e4m3(x/scale)`, write `scale`
- copy the 64 RoPE dims F32→F16 (reuse `f32_to_f16_kernel`)

Wire all ~4 write sites (single-token decode, prefill zero-prefix, aligned-chunk replay,
per-token unaligned) + session save/load (3 dtypes now) + the cache-trace, mirroring the
F16 staging structure (`metal_graph_{store,commit}_attn_comp_stage` and friends).

## Flag & plumbing

New compile-time macro **`DS4_GPU_ATTN_COMP_CACHE_FP8`** next to the F16 one
(ds4.c ~10296). When 1 it **takes precedence** over `DS4_GPU_ATTN_COMP_CACHE_F16` (the
comp cache is FP8-split). Default **0 on all backends**; flipped on **for CUDA only** at
the end, revertible independently like the F16 macros. Allocation, staging-target
selection, save/load, and the memory estimate all branch FP8 → F16 → F32.

## Memory accounting

Update `ds4_context_memory_estimate_with_prefill` **and** the managed-KV policy estimate
so the reported "context buffers … MiB" reflects 580 B/row (both account for the F16
caches today). Expected at ctx 4096: ~405.5 → **~396 MiB** — the comp cache drops from F16
21.0 MiB (1026 rows × 512 × 2 B × 21 layers) to FP8-split 11.9 MiB (1026 × 580 B × 21), a
~9.1 MiB saving (RoPE-F16 + scale offset part of the NoPE FP8 halving). Verify the measured
line matches; treat a mismatch as an estimate bug to fix.

## Incremental commits — each gated on the perplexity oracle

`./ds4 --cuda --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096`

1. **Read/expand plumbing inert** — new FP8 expand/gather kernels + `comp_kv_dtype` /
   sidecar-param signature change + call sites, all behind the OFF macro. Gate: nll
   **unchanged = 317.842979** (F16 still the live path), tree builds.
2. **Write staging + sidecar buffers inert** — alloc FP8/rope/scale per layer, the
   `quantize_f32_to_fp8_rows` commit, wire ~4 write sites + save/load + cache-trace.
   Macro still OFF. Gate: nll **unchanged**.
3. **Flip macro ON for CUDA** + memory-estimate updates. Gate: nll **≤ 319.4**, no
   NaN/crash, "context buffers" line drops as predicted.

Finish with `make cuda-spark` so `ds4-server` is rebuilt too. Build during iteration
with `make ds4 ds4-bench CUDA_ARCH=` (ds4.c relinks fast; ds4_cuda.cu re-runs nvcc).

## Gotchas (carried from Phase 2a / the brief)

- The existing `fp8_kv_quantize_kernel` / `dsv4_fp8_kv_quantize_row_inplace_cpu` are
  **fake-quant** (round-trip f32→fp8→f32 in the SAME F32 buffer) — they save nothing and
  are NOT a storage path. This design needs a genuine narrow FP8 store + per-row scales +
  a real dequant in the expand/gather kernels.
- ~4 session save/load sites + 1 cache-trace site **per cache** need the new dtype
  (mirror the F16 wiring exactly).
- ONE ds4 process at a time (instance lock); never `pkill -f ds4-server` — kill by exact
  PID. Don't `make cpu` (overwrites CUDA binaries). Greedy decode is NOT a valid oracle
  (run-to-run nondeterministic); only teacher-forced perplexity is bit-deterministic.
- Decode is GPU-bound at benchable ctx, and the indexer bandwidth win only showed at
  hundreds-of-K ctx — so the **near-term FP8 win is MEMORY**, not tok/s at 4K–25K.
