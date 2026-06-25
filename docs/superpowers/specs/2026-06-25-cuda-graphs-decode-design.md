# CUDA Graphs for the ds4 single-token decode path

**Date:** 2026-06-25
**Status:** Approved (design), implementation pending
**Scope:** Phase 1 of a multi-phase effort to make DeepSeek-V4-Flash decode faster / fit longer
context on a single GB10. This phase = **CUDA Graphs only**, no numeric changes.

## Background / motivation

Measured on this GB10 (sm_121, 273 GB/s LPDDR5X, CUDA 13): decode ~13–15 tok/s. Byte
accounting shows ~9.5 GB read/token, so the bus alone allows ~24–29 tok/s — decode runs at
**~47% of bandwidth**, i.e. the bottleneck is the decode *path*, not the silicon. Of the
~9.5 GB/token, attention (Q8) is ~61%, routed experts ~19%, shared expert ~12%, output head
~6%. (Reducing those bytes — e.g. attention Q8→Q5 — is a later, higher-risk phase.)

This phase targets the cheapest, lowest-risk slice of the gap: the per-token kernel **launch
overhead and GPU-idle bubbles** across the hundreds of kernels launched per token, by capturing
the decode token once and replaying it as a CUDA graph. Expected gain ~1.1–1.3×. Equally
important, it builds the capture/measure harness reused by later phases.

Non-goals this phase: changing any kernel math, quantization, attention requant, KV
quantization/TurboQuant, prefill, SSD-streaming, or distributed paths.

## Key facts established from the code

- **Capture seam already exists.** `metal_graph_eval_token_raw_swa` (ds4.c:19461) runs
  `ds4_gpu_begin_commands()` → `metal_graph_encode_token_raw_swa()` (embed → 43 layers →
  output head) → `ds4_gpu_end_commands()` → `ds4_gpu_tensor_read(logits)`. The encode sits
  exactly between begin/end. On CUDA, begin = no-op (ds4_cuda.cu:2479), end =
  `cudaDeviceSynchronize` (2494). These are the only functions we modify for capture.
- **Default Flash full-residency decode has no mid-token host sync** except the optional
  "split-after-layer-4" flush (`ds4_gpu_flush_commands`, ds4.c:16993, gated by the
  `allow_split_flush` arg + env `split_after_layers`, default 4). The CPU-router path
  (`metal_graph_decode_cpu_router_applicable`, ds4.c:13682) is **not** taken for Flash full
  residency — only PRO-Q4 or SSD-streaming. The GPU router is used → no routing D2H sync.
- **Single inference worker thread** owns the session/KV (ds4_server.c:10–11, 11768); CLI/bench
  are single-threaded for inference. Model loading uses explicit streams and finishes before
  decode. → `-default-stream per-thread` is safe.
- **Scratch is one growing buffer.** `cuda_tmp_alloc` (ds4_cuda.cu:254) returns the shared
  `g_cuda_tmp`, reused sequentially within a token, `cudaMalloc`-grown only when a request
  exceeds the high-water mark. Several requests scale with `n_comp` (grows with context):
  e.g. "indexed attention topk sort" (8990), "indexer topk tree" (7575), attention/output
  temps (8740/9118/9306/...). `cudaMalloc` is illegal during capture → must be avoided.
- One genuinely context-dependent **grid**: `indexer_score_one_direct_kernel <<<n_comp,128>>>`
  (ds4_cuda.cu:7373). Other decode grids are fixed (n_head / n_tokens=1). Counts (`pos`,
  `n_comp`, ...) are passed as host kernel args → baked into captured nodes.

## Design

### Capture strategy: recapture-per-token + `cudaGraphExecUpdate` (the llama.cpp model)

Because `pos` and growing counts are baked into kernel args, a once-captured graph goes stale
each token. So each token: re-run the encode under stream capture into a fresh `cudaGraph_t`
(bakes current params), then `cudaGraphExecUpdate(exec, newGraph)` to patch the existing
executable in place (topology is identical token-to-token, only params/grid dims differ). On
update success → `cudaGraphLaunch(exec)`. On failure (topology change — the rare sparse
threshold crossing, or first token) → destroy + `cudaGraphInstantiate` from the new graph. The
recapture is a few hundred cheap *record* calls (<1% of a ~74 ms token) and overlaps the prior
token's GPU execution. **No kernel changes.** (A future phase may move pos/counts into a
device-side params buffer + over-provisioned grids for a single static graph that never
recaptures — deferred; more kernel surface area.)

### Making decode kernels capturable: `-default-stream per-thread`

Stream capture only records work on a non-legacy stream, but decode kernels launch on the
legacy default stream today. Compile `ds4_cuda.cu` with `nvcc -default-stream per-thread`
(one Makefile flag) so every existing `<<<…>>>` uses `cudaStreamPerThread`, which is
capturable — **zero launch-site edits**. We then capture/launch/sync on `cudaStreamPerThread`.
Top verification item: confirm this doesn't perturb the model loader's explicit streams or
multi-thread server behavior (expected safe: single worker, loader done before decode, explicit
streams + syncs elsewhere). If it misbehaves, fall back to threading an explicit decode stream
through launch sites.

### What begin/end_commands do in graph mode

`ds4_gpu_begin_commands()`:
1. If not graph-eligible → return 1 (today's no-op; direct launch as before).
2. Ensure scratch is pre-reserved for the current token's worst case (grow `g_cuda_tmp`
   outside capture; see below). Set cuBLAS stream to `cudaStreamPerThread` if cuBLAS is on the
   decode path.
3. `cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)`.

`ds4_gpu_end_commands()`:
1. If not capturing → `cudaDeviceSynchronize` (today's behavior).
2. `cudaStreamEndCapture(&newGraph)`. If capture errored (e.g. an in-capture `cudaMalloc`):
   discard, re-run the encode **directly** for this token (graceful fallback → correct result,
   no speedup this token), then sync. This self-heals scratch growth.
3. `cudaGraphExecUpdate(exec, newGraph)`; on failure re-instantiate.
4. `cudaGraphLaunch(exec, cudaStreamPerThread)` → `cudaStreamSynchronize`.
5. Logits readback (`ds4_gpu_tensor_read`) stays outside the graph, unchanged.

`ds4_gpu_flush_commands()` (the split flush): no-op while capturing. We also pass
`allow_split_flush=false` on the graph path so it isn't relied on.

### Eligibility gate (else fall back to today's direct-launch path, byte-identical)

Graph mode only when: CUDA backend, single-token decode (the `metal_graph_eval_token_raw_swa`
path), **not** SSD-streaming, **not** distributed, **not** CPU-router-applicable, and capture
is supported. AGENT.md requires these other paths stay untouched.

### Scratch pre-reservation

- Warm the first decode token(s) of a session via the direct path (no capture) so `g_cuda_tmp`
  reaches the size that context needs.
- Before each capture, if the about-to-run token might need more scratch than the current
  high-water mark, grow it outside capture. Exact worst-case sizing is fragile, so correctness
  rests on the **capture-failure fallback** (step 2 above); pre-reservation just keeps the
  fallback rare.

### Lifetime / state

Globals in ds4_cuda.cu: `g_decode_graph_exec`, capture-active flag, an "owner signature"
(session KV-cache base pointers + ctx) to invalidate the cached exec when the active session
changes. Destroy on `ds4_gpu_*` teardown.

### Runtime toggle

`DS4_CUDA_GRAPH` env: `1` = on, `0` = off (force direct launch). Default off until validated,
then on. Always available as an instant escape hatch.

## Correctness & testing (gate before believing any speedup number)

1. **Baseline oracle:** build current code; record decode tok/s and dump reference logits at a
   couple of context frontiers (`ds4-bench --dump-frontier-logits-dir`).
2. **Stream-flag isolation:** add `-default-stream per-thread`, build, run **non-graph** decode;
   logits must be **bit-identical** to baseline and tok/s unchanged. (Isolates the flag from
   graph logic; if logits drift, the flag is unsafe → backtrack to explicit-stream route.)
3. **Graph correctness:** graph-on vs graph-off logits **bit-identical** at small context, then
   across a 2K→130K sweep with generation. Per AGENT.md: no unexplained logits drift.
4. **Speed:** `ds4-bench` decode tok/s on/off across the sweep + a large frontier.
5. **Fallback intact:** standard and SSD-streaming paths unchanged (byte-identical) with graphs
   gated off for them.

Runs happen on this GB10 (loads ~86 GB, takes the box a few minutes); confirm timing with the
user before long sweeps.

## Incremental plan (each step = a git commit, independently testable; backtrack = revert)

- **0.** Baseline build + measure + reference logits.
- **1.** Add `-default-stream per-thread`; verify non-graph logits bit-identical + tok/s flat.
- **2.** Graph cache + capture in begin/end_commands behind `DS4_CUDA_GRAPH` (default off);
  verify off = no regression.
- **3.** Graph on for one token at small context; achieve bit-identical logits; debug capture.
- **4.** Context growth: recapture + ExecUpdate + topology-change fallback; verify across sweep;
  measure tok/s.
- **5.** Eligibility gating + fallback robustness + session-change invalidation + cleanup;
  confirm SSD-streaming path intact; default `DS4_CUDA_GRAPH` on.
- **6.** Final measurement; update README/notes.

## Risks

- `-default-stream per-thread` interaction with loader/server threading (top item; isolated in
  step 1).
- cuBLAS on the decode path during capture (must share the capture stream / avoid internal
  allocs). Verify whether single-token decode touches cuBLAS at all; fallback covers it.
- Scratch growth causing in-capture `cudaMalloc` (handled by capture-failure fallback).
- Gain may land at the low end (~1.1×) if decode is already mostly GPU-bound; acceptable —
  this phase also exists to build the harness for the higher-impact KV/indexer phase.
