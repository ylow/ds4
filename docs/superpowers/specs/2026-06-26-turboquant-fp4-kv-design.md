# Phase 2c design: Hadamard-FP4 (TurboQuant) compressed-attention KV on CUDA

**Date:** 2026-06-26
**Goal:** push the dominant KV term — the 512-dim compressed *attention* rows
(`layer_attn_comp_cache`) — below FP8 to **~4-bit Hadamard-rotated FP4 (E2M1) + per-group
scale**, cutting that cache ~1.62× vs the current FP8-split (5.69× vs F32). This is the
memory lever toward a usable 600–800K context (the comp cache dominates at long ctx).

Builds directly on **Phase 2b** (FP8-split comp-attn KV via expand-on-read, commits
582c3ad → b8aa049). Only the storage dtype of the NoPE dims, the dequant kernels, and the
write-commit change; the trusted F32 attention kernels stay byte-identical, and the RoPE
tail + per-group scale buffers are **reused verbatim** from the FP8 layout.

## What "TurboQuant (~3-bit)" means here

TurboQuant = **rotation + scalar quant**: an orthogonal rotation Gaussianizes/decorrelates
the vector so a cheap per-coordinate scalar quantizer becomes near-optimal. The model
**already uses exactly this recipe for the indexer** — `dsv4_indexer_qat_row_inplace_cpu`
and `indexer_hadamard_fp4_kernel` run a normalized 128-wide Walsh–Hadamard transform then
per-32-group **E2M1 FP4** (values `{0,±.5,±1,±1.5,±2,±3,±4,±6}`, scale
`exp2(ceil(log2(amax/6)))`). We apply the same recipe to the **attention** comp NoPE dims.

"~3-bit" is realized as **4-bit FP4 storage whose Hadamard rotation buys ~3-bit-or-better
quality** — chosen over a true sub-byte 3-bit codec because FP4 is the model's own,
byte/nibble-aligned, has proven CPU+device kernels, and the 4→3-bit memory delta (360→303
B/row, ~15%) does not justify a fresh codec + sub-byte packing + extra loss on this first
genuinely-lossy step. True 3-bit remains a clean follow-on on this same split layout.

## How this differs from Phase 2b (why it is lossy now)

FP8 was **bit-identical** because the model *itself* defines E4M3 on the 448 NoPE dims —
the FP8 store merely captured the magnitude index + exponent the model already produces.
**The model does NOT define FP4 on the 512-dim attention comp rows.** FP4 is applied as a
pure *storage* compression on top of the model's defined E4M3 cache value, so Phase 2c is
the **first genuinely-lossy** KV step. The chaotic-amplification caveat (the comp cache is
a recurrence rebuilt from the model's own hidden states over the perplexity oracle's 255
teacher-forced decode steps) is now live: a passing tolerance gate is **not** proof of
correctness — value-exactness verification is.

## Decisions (locked with the user)

- **Codec:** Hadamard-64 + E2M1 **FP4** on the 448 NoPE dims (the model's own E2M1 codec).
- **Rotation block = 64** (not the indexer's 128): 448 = 7×64 divides cleanly, and 64
  aligns with the existing per-64 group structure (the FP8 scale buffer is already 8/row).
- **Scale granularity = per-64** (one scale per Hadamard block, 7 used/row). Hadamard-64
  equalizes variance within the block, so per-64 loses little and **reuses the FP8 8/row
  scale buffer verbatim**. Per-32 (14/row) is the documented fidelity fallback if drift
  exceeds the gate.
- **Staging stays E4M3-QAT'd F32** (unchanged from 2b): compressor + the existing
  `ds4_gpu_dsv4_fp8_kv_quantize_tensor` QAT run untouched; FP4 is applied only in the
  commit. The change is isolated to "storage codec only"; the intended drift is exactly
  FP4-vs-E4M3.
- **RoPE 64-dim tail stays F16; indexer 128-dim cache untouched** (low dim — per-coordinate
  scalar quant weakens there; both already at their chosen precision).
- **Read path:** extend **expand-on-read** (lowest risk — trusted F32 attention kernels +
  the gather-of-selected decode path stay unchanged; only the expand/gather dequant and the
  write-commit change).
- **Tolerance / acceptance gate:** teacher-forced perplexity. F32 ref nll=317.843968;
  current FP8/F16 build=317.842979423 (256 tok, ctx 4096). **Accept FP4 if total nll ≤
  ~319.4 (< ~0.5% avg_nll drift)** AND the value-exact self-check passes AND it is stable
  over a 16384 long-ctx run. Flag NaN, crash, or a larger jump as a bug. The self-check —
  not the nll gate — is the correctness proof; the nll gate only judges whether the
  intended 4-bit loss is acceptable.

## Storage layout (per layer, FP4 mode)

Replaces the FP8 NoPE byte buffer with a half-size nibble buffer; the scale and rope
buffers are byte-for-byte the same as FP8 mode:

| buffer | dtype | stride/row | holds |
|---|---|---|---|
| `layer_attn_comp_cache[il]` | **FP4 nibble** | **224 B** (`n_nope/2`) | NoPE dims [0..447], 2 dims/byte: even `d`→low nibble, odd→high; nibble = `(sign<<3) \| e2m1_index` |
| `layer_attn_comp_scale[il]` | int8 exponent | 8/row (7 used) | per-64-group power-of-2 exponent `k` (rotated domain) — **same buffer as FP8** |
| `layer_attn_comp_rope[il]` | F16 | 64 elems (128 B) | RoPE tail dims [448..511] — **same buffer as FP8** |

Per row: 224 + 8 + 128 = **360 B** vs FP8 584 B (1.62×), F16 1024 B (2.84×), F32 2048 B
(**5.69×**).

## Codec details

- **Hadamard-64, normalized & self-inverse.** `H64_norm = H64 / sqrt(64) = H64 · 0.125`.
  Because `(H64·0.125)² = H64²/64 = 64·I/64 = I`, the *same* normalized transform both
  rotates (write) and un-rotates (read). Implement the butterfly in the identical stride
  order on CPU and device (stride 1,2,4,…,32; then ×0.125) so host and device are
  bit-for-bit equal — pure adds/subs + a power-of-2 multiply, no FMA contraction, so equal
  op order ⇒ equal result (required for the self-check). New host
  `dsv4_hadamard64_inplace_cpu` (trivial variant of the existing `…hadamard128…`); device
  is an inline butterfly in the kernels (mirrors `indexer_hadamard_fp4_kernel`).
- **E2M1 FP4 encode.** New `dsv4_e2m1fn_index_dev/cpu` — sibling of the existing
  `dsv4_e2m1fn_dequant_*` that returns the **magnitude index** `best` (0..7) with the
  model's exact tie-break, for the nibble. Decode reuses `dsv4_e2m1fn_value_dev(idx)`.
- **Exact per-group exponent (carry the Phase 2b lesson).** `amax` over the **rotated**
  group, floored at the model's FP4 floor `7.052966104933725e-38f`; `k = exact
  ceil(log2(amax/6))` via **`frexpf`** (no `log2f` octave overshoot): `int e; float m =
  frexpf(amax/6, &e); k = (m == 0.5f) ? e-1 : e`. Same on host and device. Dequant uses
  `exp2f((float)k)`. (FP4 has no subnormal grid like E4M3, but a consistent, exactly-defined
  exponent keeps host==device and avoids a coarser-cell drift; encode and decode share `k`.)

## Read path — extend expand-on-read

Two new kernels reconstruct the full 512-dim **F32** row into the *same reused F32 scratch*
(`g_comp_f32_expand_*`) the FP8 path uses, so every downstream F32 attention kernel sees
byte-identical input:
- `expand_comp_fp4_to_f32` (prefill: all `n_comp` rows)
- `gather_comp_fp4_to_f32` (decode: only the `n_sel ≤ top_k` selected rows — keeps the
  per-token expand bounded at long ctx, mirroring `gather_comp_fp8_to_f32`)

Per output row, per 64-group `g` (`k = scale_exp[row*8 + g]`):
1. dims [g·64 .. g·64+63] rotated-domain value `r_j = (nibble&8 ? -1:1) ·
   dsv4_e2m1fn_value_dev(nibble&7) · exp2f((float)k)` where `nibble` is unpacked from
   `cache[row*224 + d/2]`.
2. **apply Hadamard-64** to the 64-vector `r` → original-basis F32 → scratch[g·64 ..].
3. dims [448..511] = `__half2float(rope_f16[row*64 + (d-448)])` (no Hadamard).

The 5 extern attention entry points that do expand-on-read generalize `comp_kv_dtype`
(0=F32, 1=F16, 2=FP8) → add **`3=FP4`**, and reset dtype→0 after expanding so the inner
launches (`attention_decode_batch_launch` etc.) are untouched — they only ever see F32. The
comp-passing ds4.c call sites already forward `g->layer_attn_comp_rope[il]` +
`g->layer_attn_comp_scale[il]`; FP4 reuses both.

## Write path — extend the F32-staging commit (only the commit changes)

**Leave every compressor + QAT call untouched.** Staging stays F32 with the model's E4M3
QAT applied, exactly as today. Only the **commit** (a new FP4 branch of
`metal_graph_store_attn_comp_stage`, taking precedence over the FP8 branch) changes: a new
helper `ds4_gpu_tensor_quantize_f32_to_fp4split` runs a kernel that, per 64-group of each
row: **Hadamard-64 the staged F32**, compute `amax → k` (frexp-exact), then nibble =
`(sign<<3) | e2m1_index(clamp(rotated/2^k, ±6))`, and writes:
- FP4 nibbles → `layer_attn_comp_cache[il]` (224/row, at `first_row*224`)
- int8 exponents → `layer_attn_comp_scale[il]` (8/row, at `first_row*8`)
- F16 RoPE tail → `layer_attn_comp_rope[il]` (64/row, at `first_row*64`; reuse
  `f32_to_f16_kernel`)

`metal_graph_attn_comp_{update_target,update_row,row_view,prefill_target}` keep returning
the F32 staging exactly as in FP8/F16 mode (no write-site churn). Session save/load + the
cache-trace gain an FP4 branch alongside FP8/F16/F32: simplest stable choice — keep the F32
disk format (decode FP4→F32 for save; re-run the commit quantizer on load). The trace
decodes FP4→F32 before diffing against the CPU `attn_comp_kv`.

## Flag & precedence

New compile-time macro **`DS4_GPU_ATTN_COMP_CACHE_FP4`** next to the FP8/F16 ones
(ds4.c ~10296). When 1 it **takes precedence over `…_FP8` → `…_F16` → F32** (the comp cache
is FP4-split). Default **0 on all backends**; flipped on **for CUDA only** at the end,
revertible independently. Allocation, staging-target selection, save/load, the cache-trace,
and the memory estimate all branch FP4 → FP8 → F16 → F32. `metal_graph_attn_comp_cache_dtype()`
returns 3 in FP4 mode.

## Memory accounting

Update the comp-cache byte term in both estimators — `ds4_context_memory_estimate_with_prefill`
(ds4.c:21607-21609) and the managed-KV policy estimator
`metal_graph_kv_cache_bytes_for_context` (ds4.c:10793-10794) — to the FP4-split row term
`n_nope/2 + 8 + n_rot·2 = 224 + 8 + 128 = 360 B/row`. Staging stays F32 (staging terms
unchanged). Expected at ctx 4096: comp cache **12.0 MiB (FP8) → 7.4 MiB** (1026 rows × 360 B
× 21 layers = 7,756,560 B), context buffers **396.21 → ~391.6 MiB**. Verify the measured
line matches; treat a mismatch as an estimate bug. The saving is linear in ctx → ~0.4 GB
@128K, multi-GB toward 800K.

## Incremental commits — each gated on the perplexity oracle

`./ds4 --cuda --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096`

1. **Read/expand plumbing inert** — `expand_comp_fp4_to_f32` / `gather_comp_fp4_to_f32` +
   Hadamard-64 device helper + `dsv4_e2m1fn_index_dev` + `comp_kv_dtype==3` in the 5 entry
   points, all behind the OFF macro. Gate: nll **unchanged = 317.842979423** (FP8 still
   live), tree builds.
2. **Write staging + buffers inert** — alloc the 224 B/row nibble buffer (rope/scale
   buffers already exist from FP8), the `ds4_gpu_tensor_quantize_f32_to_fp4split` commit
   (FP4 branch of `metal_graph_store_attn_comp_stage` only — compressor/QAT untouched), the
   FP4 save/load + cache-trace branches, `dsv4_hadamard64_inplace_cpu` +
   `dsv4_e2m1fn_index_cpu` host codec. Macro still OFF. Gate: nll **unchanged**.
3. **Value-exactness harness** (the correctness proof; run before trusting nll). Host
   reference computes expected `(nibbles, k, F16 rope)` from the E4M3-staged F32 row via the
   identical Hadamard-64 + frexp-exact-k + e2m1 codec; assert the GPU cache buffers match
   **byte-for-byte** (selfcheck), and assert GPU-expand(cache) == host-decode(nibbles,k,rope)
   **bit-for-bit** (readcheck). This separates intended FP4 loss from bugs (the class of bug
   Phase 2b hit). Temporary debug scaffold; gated behind a debug env/flag, removed before the
   final commit.
4. **Flip macro ON for CUDA** + memory-estimate updates. Gate: **nll ≤ 319.4**, value-exact
   harness passes, no NaN/crash, the "context buffers" line drops to ~391.6 MiB @4096, AND a
   long-ctx run (`doors-of-stone` concatenated ~8× ≈ 26K tokens, `-c 16384`) is stable (no
   blow-up/NaN, avg_nll not diverging).

Finish with `make cuda-spark` so `ds4-server` is rebuilt too. Build during iteration with
`make ds4 ds4-bench CUDA_ARCH=` (ds4.c relinks fast; ds4_cuda.cu re-runs nvcc ~minutes).
Update `ds4-optimization-findings` memory + the Status section below afterward.

## Gotchas (carried from Phase 2a/2b + the brief)

- The existing `indexer_hadamard_fp4_kernel` / `dsv4_fp4_act_quantize_row_inplace_cpu` are
  **fake-quant** (round-trip f32→fp4→f32 in the same F32 buffer) — as a *storage* path they
  save nothing. BUT their Hadamard + per-group E2M1 math is the model's proven FP4 numeric
  definition, so the new commit kernel re-implements exactly that math (at block 64, scale
  per 64) while emitting (FP4 nibble, int8 exponent) to narrow buffers — that is what makes
  the store genuine. Reuse the `__device__` `dsv4_e2m1fn_value_dev` (4537) + the rounding in
  `dsv4_e2m1fn_dequant_dev` (4550); write a sibling returning the magnitude index for encode.
- **Exact exponent via `frexpf`**, not `(int)ceilf(log2f(...))` — the Phase 2b octave
  overshoot lesson. Host and device must compute `k` identically for the byte-for-byte
  selfcheck.
- **Host==device Hadamard bit-exactness** is required for the readcheck: identical butterfly
  stride order, no FMA contraction (none in add/sub butterfly), final ×0.125 is exact.
- Session save/load sites (save-layer, load-layer, save, load) + the cache-trace site need an
  FP4 branch alongside FP8/F16/F32. Keep the F32 disk format (decode on save, re-commit on
  load).
- ONE ds4 process at a time (instance lock); never `pkill -f ds4-server` — kill by exact
  PID. Don't `make cpu` (overwrites CUDA binaries). Greedy decode is NOT a valid oracle
  (run-to-run nondeterministic); only teacher-forced perplexity is bit-deterministic.
- Decode is GPU-bound at benchable ctx and the read-side Hadamard adds a tiny per-row cost,
  so the **near-term win is MEMORY**, not tok/s at 4K–25K. In-kernel FP4 reads (recover read
  bandwidth, drop the expand pass — the indexer pattern) remain a later follow-on.

## Status — LANDED (NF4 codec; +0.367% nll, within the 0.5% gate)

Implemented 2026-06-26 in five commits: `bc4ac73` read plumbing (inert), `dbeb484`
write/buffers/save-load/trace (inert), `a2ba661` value self-check (env-gated), then the
flip `d0c8b30`.

**Codec pivot — E2M1 → NF4.** The design above specifies the model's own **E2M1** FP4 on
the Hadamard-rotated NoPE dims. On flip-on, E2M1 was byte-exact (self-check clean) but
**+2.13%** nll (324.62 @4096) — 4× the 0.5% gate. Root cause: E2M1's exponential level
spacing `{0,±.5,±1,±1.5,±2,±3,±4,±6}` is mismatched to the **~Gaussian** post-Hadamard
distribution. Swapped the FP4-comp codec to **NF4** (NormalFloat4): the 16 unit-Gaussian
**quantile** levels on [-1,1], a full 4-bit nibble (no sign bit), max level 1.0. Same
per-64 power-of-2 int8 scale, same 360 B/row layout, same row split — only the value table
+ nearest-level search changed (`nf4_level_dev/_cpu`, `nf4_index_dev/_cpu`,
`nf4_decode_nibble_dev/_cpu`). The indexer keeps its own E2M1; the now-dead FP4-comp E2M1
device helpers are retained `DS4_CUDA_UNUSED`.

**Result.** Value self-check **byte-perfect** across all 41 layers (byte/val_mismatch=0
over 1384 checks) ⇒ the drift is intended NF4 loss, not a bug. Teacher-forced perplexity
**nll=319.009406860 @4096** (avg 1.2461, ppl 3.462→3.477) = **+0.367%** vs the FP8 build
317.842979423 — **within the ≤319.4 (0.5%) gate** (NF4 cut E2M1's +2.13% by ~6×). Context
buffers **396.21 → 391.46 MiB @4096** (comp cache 12.0 → ~7.25 MiB), 547.52 MiB @16384;
linear in ctx → multi-GB toward 600–800K. Stable at ctx 16384 (no NaN, avg_nll 1.246).

**Follow-ons** (unchanged priority): chase the last range bit with a per-group **F16
absmax** scale instead of power-of-2 (NF4's [-1,1] is currently under-filled by ~0.3 bit
when a group's absmax sits just above a power of 2); in-kernel FP4 reads for the *attention*
comp cache (recover read bandwidth, drop the expand pass); raw-KV → F16 (lossless, low
value); then true sub-byte **3-bit** on this same Hadamard-split layout if the memory lever
needs more.
