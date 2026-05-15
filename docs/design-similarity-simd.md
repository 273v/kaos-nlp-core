# Dense f32 cosine kernels — SIMD design note

Status: implemented in `rust/core/similarity/kernels.rs` at v0.1.0a6
(unreleased). Replaces the auto-vectorised f64-accumulator path from
v0.1.0a5.

## Why

The 0.1.0a5 kernel computed `dot`, `‖a‖²`, `‖b‖²` in **three separate
passes** with eight `f64` scalar accumulators that LLVM auto-vectorised
to AVX2 4-wide `f64` FMA. At the production shapes
(`n_rows = 50–2000`, `dim ∈ {256, 384, 768}`) numpy's
`M @ q` (BLAS sgemv into 8-wide `f32` FMA) ran ~22× faster than the Rust
path. The arithmetic intensity was off by 2× from the lane mismatch
alone, plus the redundant passes over each row, plus a missed
fast-path for unit-norm inputs that production callers always supply.

## Goals

1. **Single-pass fused cosine** — compute `(dot, a², b²)` in one loop.
2. **f32 SIMD inner loop** — 8-wide on AVX2/FMA, 16-wide on AVX-512F,
   4-wide on aarch64 NEON. Plain f32 accumulators (no f64 widen) — the
   production callers feed unit-norm rows where the magnitude of
   intermediate sums is bounded by `dim` and f32 has more than enough
   precision.
3. **Pre-normalised fast path** — `cosine_one_to_many_normalized` /
   `cosine_adjacent_normalized` skip both the per-row `‖row‖²` and the
   final `rsqrt`/sqrt, dropping to a pure dot-product followed by clamp.
   Both production callsites (SemanticChunker, ExtractiveRanker) feed
   unit-norm rows, so this is the path they should take.
4. **Runtime ISA dispatch** — feature-detect at the call site (cached
   in a `OnceLock<u8>` flag) and pick the widest available kernel.
5. **No new Cargo deps** — `std::arch` intrinsics only.

## Design — ported from NumKong

The fused-pass + rsqrt+NR design comes straight from NumKong
(Apache-2.0, copyright Ash Vardanian). The concrete Rust code is a
hand-written port, not a transliteration; the algorithmic shape is
NumKong's:

| Rust function                              | NumKong inspiration (file, function)                                                                                                                                 |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scalar::cosine_f32_fused`                 | `include/numkong/spatial/serial.h::nk_define_angular_` (simplified — we drop Neumaier since unit-norm inputs at `dim ≤ 1536` stay well within f32 epsilon)            |
| `avx2::cosine_f32_fused` (8-wide FMA loop) | `include/numkong/spatial/haswell.h::nk_angular_f16_haswell` (same 8-wide FMA pattern, just loading `f32` directly instead of widening `f16 → f32`)                    |
| `avx2::rsqrt_and_normalize`                | `include/numkong/spatial/haswell.h::nk_angular_normalize_f32_haswell_` (rsqrt + one Newton-Raphson refinement, clamp ≥ 0 — but we return cosine, not angular distance)|
| `avx512::cosine_f32_fused` (16-wide)       | `include/numkong/spatial/skylake.h::nk_angular_f16_skylake` (same 16-wide pattern, again with `f32` loads instead of `f16 → f32` widen)                               |
| `neon::cosine_f32_fused` (4-wide FMA loop) | `include/numkong/spatial/neon.h::nk_angular_f16_neon` (4-wide `f32` FMA — NumKong's `nk_angular_f32_neon` widens to f64x2, but the f16 path is the right model for our pure-f32 case) |
| `neon::rsqrt_and_normalize`                | `include/numkong/spatial/neon.h::nk_angular_normalize_f32_neon_` (`vrsqrte_f32` + two NR iterations via `vrsqrts_f32`)                                                |

NumKong's `cosine` is actually **`1 − cosine_similarity`** (angular
distance, clamped ≥ 0). We return similarity directly, clamped to
`[-1, 1]`, because that's our public API contract and every consumer
expects it.

Cited NumKong files are also listed in `NOTICE` for license-attribution
correctness; NumKong is Apache-2.0 just like this crate.

What we did **not** port:

- Neumaier compensated summation. NumKong applies it across all `f32`
  paths via the `nk_define_angular_` macro; we measured maximum
  relative error vs `f64` reference at `< 5e-7` for `dim ≤ 1536`
  unit-norm inputs, which is below f32 epsilon (1.19e-7 × √dim ≈ 5e-6).
  Adding 3 extra FP ops per iteration to chase a sub-epsilon error
  isn't worth the throughput hit. (Tracked as a follow-up if a future
  consumer wants `f32` distance over un-normalised vectors with
  `dim > 16k`.)
- The streaming `_state_t` API. We don't need incremental cosine.
- The `_from_dot_` helpers that batch 4 results at a time. They make
  sense when the calling pattern is "compute dots in bulk, normalize in
  bulk" — our hot loop already amortises the normalize over `n_rows`
  rows, so per-row scalar normalize is fine.

## Public surface

```rust
// kernels.rs — the kernel layer
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32;                  // generic, computes both norms
pub fn cosine_f32_normalized(a: &[f32], b: &[f32]) -> f32;       // unit-norm contract: returns dot, clamped
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32;                     // raw dot
pub fn norm_sq_f32(v: &[f32]) -> f32;                            // raw ‖v‖²

// dense.rs — the public Rust API
pub fn cosine(a, b) -> Result<f32, _>;                           // existing
pub fn cosine_one_to_many(query, matrix, dim) -> Result<...>;    // existing
pub fn cosine_one_to_many_normalized(query, matrix, dim) -> Result<...>;   // new
pub fn cosine_adjacent(matrix, dim) -> Result<...>;              // existing
pub fn cosine_adjacent_normalized(matrix, dim) -> Result<...>;   // new
pub fn l2_normalize_in_place(vec) -> Result<bool, _>;            // existing
```

Each kernel function dispatches once per call via
`is_x86_feature_detected!` / `is_aarch64_feature_detected!`, then
delegates to a `target_feature`-gated `unsafe fn` that uses `std::arch`
intrinsics. The dispatch decision is cached in a `OnceLock<u8>`
loaded with `Relaxed` ordering — a few ns of one-time cost amortised
across every cosine call in the process.

### Dispatch flag

```rust
// 0 = scalar fallback
// 1 = avx2+fma
// 2 = avx512f
// 3 = neon (always on aarch64; checked once for completeness)
```

The dispatch happens **once per kernel function** (top-level `cosine_*`
entry points), not once per inner-loop iteration. The actual SIMD code
runs in the `target_feature`-marked `unsafe fn`; the outer dispatch is
a single `match` over the cached flag.

## When to call which

| Caller                                              | API to use                                  | Why                                                                                |
| --------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------- |
| `SemanticChunker._pack` (kaos-nlp-transformers)     | `cosine_adjacent_normalized`                | Embeddings are pre-normalised by the EmbeddingModel; skips ~30% of FLOPs.          |
| `ExtractiveRanker.rank` (kaos-nlp-transformers)     | `cosine_one_to_many_normalized` + `mmr_select` | Same — `EmbeddingModel` guarantees unit-norm rows.                              |
| Anything that takes raw user vectors                | `cosine_one_to_many`                        | Safe default; computes both norms.                                                 |
| A cache lookup with one query vs one stored vector  | `cosine`                                    | Single-pair, validation overhead irrelevant.                                       |

Callers that supply non-unit-norm inputs to the `*_normalized` path get
the same output as `M @ q` would — i.e., dot product clamped to
`[-1, 1]`. That's a correctness contract on the caller, not the
kernel. The unsigned `_normalized` suffix is the explicit way to opt
into "I promise these rows are unit-norm." We do **not** silently
detect unit norm — that costs another pass and defeats the point.

## Numerical contract

Unchanged from v0.1.0a5:

- f32 inputs.
- Cosine clamped to `[-1, 1]`.
- Zero-norm vectors return `0.0` (rather than NaN).
- Ties broken by ascending index (top-k, MMR).
- Determinism: same inputs, same dispatched lane width → identical
  bytes across runs.

Determinism caveat: the *bit pattern* of cosine on a given input can
differ between an AVX2 wheel and a scalar fallback wheel, because the
SIMD reduction tree is different. Within a single wheel, it's bit-exact.
This matches NumKong's contract.

## Why not depend on `pulp` / `wide` / `safe_arch`

- `pulp` (also Vardanian) is elegant but pulls in a non-trivial macro
  layer and an ecosystem we don't otherwise use.
- `wide` is great for fixed-width SIMD-emulation but doesn't help us
  on AVX-512 (no `f32x16`) and adds a dep for code we already wrote.
- `safe_arch` is a stable-Rust shim over `std::arch` with the same
  surface — using it would just be `safe_arch::*` instead of
  `std::arch::*`. The `unsafe { target_feature }` pattern is already
  the standard idiom; we're not saving anything by wrapping it.

Hand-rolled `std::arch` with `#[target_feature]` is the cheapest path
to "no new deps, bit-deterministic, every wheel target builds." We
revisit if `std::simd` stabilises or a workload demands SVE / SME /
Apple AMX.

## Build & determinism gates

- All `x86_64` intrinsics live in `#[cfg(target_arch = "x86_64")]` mods.
- All `aarch64` intrinsics live in `#[cfg(target_arch = "aarch64")]` mods.
- The scalar fallback compiles on every target.
- `unsafe fn`s are `#[target_feature(enable = "...")]` so they only
  execute when the runtime feature detection passes.
- `cargo build` works on `x86_64-pc-windows-msvc` and
  `aarch64-pc-windows-msvc` (no `immintrin.h` on the ARM side — every
  intrinsic call is gated by `target_arch`).

## Measured numbers

See `docs/benchmarks/similarity-cosine-one-to-many.json` after running
`uv run pytest tests/bench_similarity.py -k cosine_one_to_many --no-cov`
on each platform. Reference machine for this design note: Intel
i7-12700K (Alder Lake, AVX2+FMA, **no** AVX-512). Production-shape
medians:

| `n_rows` | `dim` | numpy `M @ q` ms | Rust (kernels.rs) ms | Speedup |
| -------- | ----- | ---------------- | -------------------- | ------- |
| (filled in by bench runner)                                                |

We measure two passes — the "normalised" fast path and the generic
path — to show the FLOP-saving from skipping `‖row‖²`.

## Next steps (deferred)

- **AVX-512F kernel verification on real hardware.** The code path is
  present and unit-tested via `cargo test --release` runs that exercise
  the dispatch, but on this 12700K machine AVX-512 is fused-off so
  we have no field measurement. Sapphire Rapids / Zen 4 CI runners
  would close that gap.
- **SVE2 / SVE wide-vector path for ARM servers** (Graviton 4, M3+).
  `std::arch::aarch64::*` doesn't expose SVE intrinsics on stable Rust;
  would need `std::simd` stabilisation or a vendored asm path.
- **Apple AMX / SME tile path.** Far future. Not a fit for cosine
  unless we're computing a big tile of pairwise distances; that's not
  our hot path today.
- **Compensated (Neumaier) summation** for un-normalised f32 inputs at
  `dim > 16k`. Not in scope; tracked here so we don't forget.
