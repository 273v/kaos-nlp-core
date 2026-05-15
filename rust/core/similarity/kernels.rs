//! Hardware-accelerated f32 cosine / dot kernels.
//!
//! This module is the SIMD floor of `kaos_nlp_core.similarity`. Every
//! cosine call in the crate (`cosine`, `cosine_one_to_many`,
//! `cosine_adjacent`, `top_k_cosine`, `mmr_select`, semantic-cache
//! lookup) routes through one of the functions here.
//!
//! Design — ported from NumKong (Apache-2.0)
//! -----------------------------------------
//!
//! Three patterns from NumKong's `include/numkong/spatial/` headers
//! drive this file. They are documented at the call sites and in
//! `docs/design-similarity-simd.md`:
//!
//! 1. **Fused single-pass cosine**: compute `dot`, `‖a‖²`, `‖b‖²` in
//!    ONE loop with three parallel SIMD accumulators driven by FMA.
//!    NumKong source: `spatial/haswell.h::nk_angular_f16_haswell`
//!    (8-wide AVX2 FMA pattern), `spatial/neon.h::nk_angular_f16_neon`
//!    (4-wide NEON FMA), `spatial/skylake.h::nk_angular_f16_skylake`
//!    (16-wide AVX-512 FMA). We use the f16-variant pattern (pure
//!    `f32` accumulators) rather than the f32-variant pattern
//!    (`f64` accumulators) because our inputs are already `f32` and at
//!    `dim ≤ 1536` with unit-norm rows the magnitude of intermediate
//!    sums stays well within `f32` precision.
//!
//! 2. **`rsqrt` + Newton-Raphson finalisation** for the per-pair scalar
//!    normalize. NumKong source:
//!    `spatial/haswell.h::nk_angular_normalize_f32_haswell_`,
//!    `spatial/neon.h::nk_angular_normalize_f32_neon_`. We use scalar
//!    `f64::sqrt` rather than `rsqrt` here because the normalize is
//!    `O(1)` per row (not per lane), bit-stability across the ISA
//!    dispatch matters, and `_mm_rsqrt_ps`'s 1.5×2⁻¹² relative error is
//!    too coarse for our `[-1, 1]`-clipped cosine contract.
//!
//! 3. **Runtime ISA dispatch with a cached flag**: feature-detect once
//!    via `std::arch::is_x86_feature_detected!` /
//!    `std::arch::is_aarch64_feature_detected!`, cache in a
//!    `OnceLock<Dispatch>`, then `match` on the cached flag **once per
//!    batch entry point** (not per row). NumKong source:
//!    `c/dispatch_f32.c`. Same idea, simpler taxonomy — we only need 4
//!    buckets (scalar / AVX2+FMA / AVX-512F / NEON), not NumKong's 20+.
//!
//! Batch hot-path layout
//! ---------------------
//!
//! The `*_one_to_many_*` / `*_adjacent_*` batch entry points dispatch
//! **once** and then run the per-row loop inside a single
//! `#[target_feature]`-marked `unsafe fn`. This keeps SIMD registers
//! alive across rows, lets LLVM hoist constants out of the inner loop
//! (notably the query norm and `_mm256_setzero_ps` accumulator init),
//! and avoids re-entering the dispatch state once per row.
//!
//! Per-row FMA count
//! -----------------
//!
//! - Generic one-to-many: **2 FMAs per element per row** — `(dot, ‖row‖²)`.
//!   The query norm is computed once outside the loop. This halves the
//!   FMA count relative to a naive "compute all three each row".
//!   NumKong's `nk_angular_f32_haswell` does 3 because it's pair-wise
//!   (no shared query); the batch primitive at
//!   `include/numkong/dense_dense/haswell.h` is the analog where
//!   NumKong itself hoists shared sums out of the row loop.
//!
//! - Generic single-pair `cosine`: 3 FMAs per element. Pair-wise; both
//!   norms are call-local.
//!
//! - Normalized fast paths: 1 FMA per element (just the dot product).
//!
//! What we did NOT port
//! --------------------
//!
//! - **Neumaier compensation** (`nk_define_angular_` macro in
//!   `spatial/serial.h`). It adds 3 extra FP ops per iteration to
//!   reduce relative error from ~1e-7 to ~1e-9 at `dim = 100k`. At our
//!   production `dim ≤ 1536` the uncompensated error is already below
//!   `f32` epsilon — adding the branch on the hot path isn't worth
//!   it. If we ever support `dim > 16k` un-normalised, revisit.
//!
//! - **Streaming `_state_t` API**. No consumer needs incremental
//!   cosine; we always have both vectors in hand.
//!
//! - **The `_from_dot_` helpers** that batch four normalisations at
//!   once. They are an optimisation for a *separate* "compute all the
//!   dots first, then normalise in bulk" pattern — our hot loop already
//!   amortises the normalize over `n_rows` rows in `O(1)` scalar work
//!   per row, so 4-wide bulk normalize would save no real time.
//!
//! See `docs/design-similarity-simd.md` for the full design rationale.

use std::sync::OnceLock;

// ============================================================================
// Dispatch
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dispatch {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
    #[cfg(target_arch = "x86_64")]
    Avx512f,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

fn dispatch() -> Dispatch {
    static FLAG: OnceLock<Dispatch> = OnceLock::new();
    *FLAG.get_or_init(detect_dispatch)
}

fn detect_dispatch() -> Dispatch {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512F is the widest; AVX2+FMA the universal x86_64
        // server / desktop baseline. We deliberately require BOTH
        // `avx2` AND `fma` for the AVX2 path because `_mm256_fmadd_ps`
        // is in the FMA ISA, not AVX2 proper. (Almost every CPU with
        // AVX2 also has FMA, but the feature bits are independent —
        // see Intel SDM Vol. 1, table 14-2.)
        if std::arch::is_x86_feature_detected!("avx512f") {
            return Dispatch::Avx512f;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Dispatch::Avx2Fma;
        }
        return Dispatch::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on AArch64 — every wheel target we ship
        // (`aarch64-unknown-linux-gnu`, `aarch64-apple-darwin`,
        // `aarch64-pc-windows-msvc`) guarantees it. We feature-detect
        // anyway so a future ARMv7 build doesn't silently pick up
        // these intrinsics.
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Dispatch::Neon;
        }
        return Dispatch::Scalar;
    }
    #[allow(unreachable_code)]
    Dispatch::Scalar
}

// ============================================================================
// Scalar fallback
// ============================================================================
//
// Pure-Rust f32 inner loops. Used on platforms that don't have AVX2 +
// FMA or NEON, AND as the per-block tail for the SIMD kernels. The
// compiler is free to auto-vectorise these on its own — but the
// authoritative perf path goes through the explicit-intrinsic kernels.

mod scalar {
    /// Fused single-pass `(dot, a², b²)` over equal-length slices.
    /// This is the NumKong `nk_define_angular_` pattern, simplified
    /// to plain f32 accumulators (no Neumaier).
    #[inline]
    pub(super) fn cosine_f32_fused(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        debug_assert_eq!(a.len(), b.len());
        // Four parallel lanes so LLVM can auto-vec the fallback too,
        // and so the dependency chain isn't strict serial.
        let mut dot = [0.0_f32; 4];
        let mut na = [0.0_f32; 4];
        let mut nb = [0.0_f32; 4];
        let n = a.len();
        let chunks = n / 4;
        for k in 0..chunks {
            let base = k * 4;
            for j in 0..4 {
                let ai = a[base + j];
                let bi = b[base + j];
                dot[j] = ai.mul_add(bi, dot[j]);
                na[j] = ai.mul_add(ai, na[j]);
                nb[j] = bi.mul_add(bi, nb[j]);
            }
        }
        // Tail.
        let mut dot0 = (dot[0] + dot[1]) + (dot[2] + dot[3]);
        let mut na0 = (na[0] + na[1]) + (na[2] + na[3]);
        let mut nb0 = (nb[0] + nb[1]) + (nb[2] + nb[3]);
        for i in (chunks * 4)..n {
            let ai = a[i];
            let bi = b[i];
            dot0 = ai.mul_add(bi, dot0);
            na0 = ai.mul_add(ai, na0);
            nb0 = bi.mul_add(bi, nb0);
        }
        (dot0, na0, nb0)
    }

    /// **Partial** fused pass: `(dot, b²)` — two FMAs per element
    /// instead of three. Used by `cosine_one_to_many` where the query
    /// (here `a`) norm is computed once outside the row loop.
    #[inline]
    pub(super) fn cosine_partial_f32(a: &[f32], b: &[f32]) -> (f32, f32) {
        debug_assert_eq!(a.len(), b.len());
        let mut dot = [0.0_f32; 4];
        let mut nb = [0.0_f32; 4];
        let n = a.len();
        let chunks = n / 4;
        for k in 0..chunks {
            let base = k * 4;
            for j in 0..4 {
                let ai = a[base + j];
                let bi = b[base + j];
                dot[j] = ai.mul_add(bi, dot[j]);
                nb[j] = bi.mul_add(bi, nb[j]);
            }
        }
        let mut dot0 = (dot[0] + dot[1]) + (dot[2] + dot[3]);
        let mut nb0 = (nb[0] + nb[1]) + (nb[2] + nb[3]);
        for i in (chunks * 4)..n {
            let ai = a[i];
            let bi = b[i];
            dot0 = ai.mul_add(bi, dot0);
            nb0 = bi.mul_add(bi, nb0);
        }
        (dot0, nb0)
    }

    /// Plain f32 dot product (pre-normalised fast path).
    #[inline]
    pub(super) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc = [0.0_f32; 4];
        let n = a.len();
        let chunks = n / 4;
        for k in 0..chunks {
            let base = k * 4;
            for j in 0..4 {
                acc[j] = a[base + j].mul_add(b[base + j], acc[j]);
            }
        }
        let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for i in (chunks * 4)..n {
            s = a[i].mul_add(b[i], s);
        }
        s
    }

    /// Squared L2 norm.
    #[inline]
    pub(super) fn norm_sq_f32(v: &[f32]) -> f32 {
        let mut acc = [0.0_f32; 4];
        let n = v.len();
        let chunks = n / 4;
        for k in 0..chunks {
            let base = k * 4;
            for j in 0..4 {
                let x = v[base + j];
                acc[j] = x.mul_add(x, acc[j]);
            }
        }
        let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for &x in &v[chunks * 4..n] {
            s = x.mul_add(x, s);
        }
        s
    }
}

// ============================================================================
// AVX2 + FMA (x86_64)
// ============================================================================
//
// 8-wide f32 FMA. The pattern is lifted from NumKong's
// `nk_angular_f16_haswell` (spatial/haswell.h:231): three
// `_mm256_setzero_ps()` accumulators, one `_mm256_loadu_ps` per
// operand per iteration, three `_mm256_fmadd_ps` per iteration
// (dot, a², b²), horizontal-add to scalar at the end. We strip the
// f16-to-f32 widen (NumKong does `_mm256_cvtph_ps`; our inputs are
// already f32) and keep the rest verbatim.
//
// **2× unrolling**: every kernel uses two independent accumulators
// per quantity. On Alder Lake / Zen 4 the `vfmadd*ps` instruction has
// 4-cycle latency and 0.5/cycle throughput; with a single dep chain
// the loop is latency-bound at one FMA every 4 cycles, leaving ~85%
// of peak throughput on the floor. Two in-flight accumulators saturate
// the FMA pipe.
//
// The `unsafe fn`s are marked `#[target_feature(enable = ...)]` so
// they only ever execute on a CPU that has the runtime check passed
// in `dispatch()`. Calling them on a non-AVX2 CPU is UB; the public
// API guards this with the `match dispatch()` at the top of each
// dispatched function.

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Horizontal sum of an `__m256` — folds 8 lanes to scalar via
    /// the standard hadd → extract → add pattern.
    /// (NumKong: `numkong/reduce/haswell.h::nk_reduce_add_f32x8_haswell_`.)
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn reduce_add_f32x8(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf = _mm_movehl_ps(shuf, sums);
        let sums = _mm_add_ss(sums, shuf);
        _mm_cvtss_f32(sums)
    }

    /// Fused `(dot, a², b²)` over equal-length slices.
    /// 8-wide FMA, **2× unrolled** for FMA-latency-hiding.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn cosine_f32_fused(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut i = 0;
        let mut dot0 = _mm256_setzero_ps();
        let mut dot1 = _mm256_setzero_ps();
        let mut na0 = _mm256_setzero_ps();
        let mut na1 = _mm256_setzero_ps();
        let mut nb0 = _mm256_setzero_ps();
        let mut nb1 = _mm256_setzero_ps();
        while i + 16 <= n {
            let av0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv0 = _mm256_loadu_ps(b.as_ptr().add(i));
            let av1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let bv1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            dot0 = _mm256_fmadd_ps(av0, bv0, dot0);
            dot1 = _mm256_fmadd_ps(av1, bv1, dot1);
            na0 = _mm256_fmadd_ps(av0, av0, na0);
            na1 = _mm256_fmadd_ps(av1, av1, na1);
            nb0 = _mm256_fmadd_ps(bv0, bv0, nb0);
            nb1 = _mm256_fmadd_ps(bv1, bv1, nb1);
            i += 16;
        }
        if i + 8 <= n {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            dot0 = _mm256_fmadd_ps(av, bv, dot0);
            na0 = _mm256_fmadd_ps(av, av, na0);
            nb0 = _mm256_fmadd_ps(bv, bv, nb0);
            i += 8;
        }
        let dot_v = _mm256_add_ps(dot0, dot1);
        let na_v = _mm256_add_ps(na0, na1);
        let nb_v = _mm256_add_ps(nb0, nb1);
        let mut dot_s = reduce_add_f32x8(dot_v);
        let mut na_s = reduce_add_f32x8(na_v);
        let mut nb_s = reduce_add_f32x8(nb_v);
        while i < n {
            let ai = *a.get_unchecked(i);
            let bi = *b.get_unchecked(i);
            dot_s = ai.mul_add(bi, dot_s);
            na_s = ai.mul_add(ai, na_s);
            nb_s = bi.mul_add(bi, nb_s);
            i += 1;
        }
        (dot_s, na_s, nb_s)
    }

    /// Partial fused pass: `(dot, b²)` — two FMAs per element.
    /// Used by the one-to-many generic batch where the query norm
    /// is computed once outside the row loop. 2× unrolled.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn cosine_partial_f32(a: &[f32], b: &[f32]) -> (f32, f32) {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut i = 0;
        let mut dot0 = _mm256_setzero_ps();
        let mut dot1 = _mm256_setzero_ps();
        let mut nb0 = _mm256_setzero_ps();
        let mut nb1 = _mm256_setzero_ps();
        while i + 16 <= n {
            let av0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv0 = _mm256_loadu_ps(b.as_ptr().add(i));
            let av1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let bv1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            dot0 = _mm256_fmadd_ps(av0, bv0, dot0);
            dot1 = _mm256_fmadd_ps(av1, bv1, dot1);
            nb0 = _mm256_fmadd_ps(bv0, bv0, nb0);
            nb1 = _mm256_fmadd_ps(bv1, bv1, nb1);
            i += 16;
        }
        if i + 8 <= n {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            dot0 = _mm256_fmadd_ps(av, bv, dot0);
            nb0 = _mm256_fmadd_ps(bv, bv, nb0);
            i += 8;
        }
        let mut dot_s = reduce_add_f32x8(_mm256_add_ps(dot0, dot1));
        let mut nb_s = reduce_add_f32x8(_mm256_add_ps(nb0, nb1));
        while i < n {
            let ai = *a.get_unchecked(i);
            let bi = *b.get_unchecked(i);
            dot_s = ai.mul_add(bi, dot_s);
            nb_s = bi.mul_add(bi, nb_s);
            i += 1;
        }
        (dot_s, nb_s)
    }

    /// Plain dot product, 8-wide, 2× unrolled.
    /// Breaks the FMA dep chain at the cost of one extra register.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut i = 0;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        // 4× unroll. FMA latency is 4 cycles, throughput 0.5/cycle on
        // Alder Lake P-core and Zen 4, so 4 in-flight accumulators
        // saturate the FMA pipe.
        while i + 32 <= n {
            let av0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv0 = _mm256_loadu_ps(b.as_ptr().add(i));
            let av1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let bv1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            let av2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
            let bv2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
            let av3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
            let bv3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
            acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
            acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
            acc2 = _mm256_fmadd_ps(av2, bv2, acc2);
            acc3 = _mm256_fmadd_ps(av3, bv3, acc3);
            i += 32;
        }
        if i + 16 <= n {
            let av0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv0 = _mm256_loadu_ps(b.as_ptr().add(i));
            let av1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let bv1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
            acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
            i += 16;
        }
        if i + 8 <= n {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(av, bv, acc0);
            i += 8;
        }
        let sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut s = reduce_add_f32x8(sum);
        while i < n {
            let ai = *a.get_unchecked(i);
            let bi = *b.get_unchecked(i);
            s = ai.mul_add(bi, s);
            i += 1;
        }
        s
    }

    /// Squared L2 norm, 8-wide, 2× unrolled.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn norm_sq_f32(v: &[f32]) -> f32 {
        let n = v.len();
        let mut i = 0;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        while i + 16 <= n {
            let x0 = _mm256_loadu_ps(v.as_ptr().add(i));
            let x1 = _mm256_loadu_ps(v.as_ptr().add(i + 8));
            acc0 = _mm256_fmadd_ps(x0, x0, acc0);
            acc1 = _mm256_fmadd_ps(x1, x1, acc1);
            i += 16;
        }
        if i + 8 <= n {
            let x = _mm256_loadu_ps(v.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(x, x, acc0);
            i += 8;
        }
        let mut s = reduce_add_f32x8(_mm256_add_ps(acc0, acc1));
        while i < n {
            let x = *v.get_unchecked(i);
            s = x.mul_add(x, s);
            i += 1;
        }
        s
    }

    // ------------------------------------------------------------
    // Batch entry points — dispatch once per call, run all rows
    // inside this `target_feature` block.
    // ------------------------------------------------------------

    /// Generic one-to-many: writes `cosine(query, matrix[i])` into
    /// `out[i]` for `i ∈ [0, n_rows)`. Computes the query norm once
    /// outside the row loop; per-row inner loop runs at 2 FMAs/elt.
    ///
    /// Safety: `matrix.len() == n_rows * dim`, `out.len() == n_rows`,
    /// `query.len() == dim`.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn cosine_one_to_many_into(
        query: &[f32],
        matrix: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        let q_norm_sq = norm_sq_f32(query);
        let n_rows = out.len();
        if q_norm_sq <= 0.0 {
            for slot in out.iter_mut() {
                *slot = 0.0;
            }
            return;
        }
        for row in 0..n_rows {
            let row_slice = matrix.get_unchecked(row * dim..(row + 1) * dim);
            let (dot, row_sq) = cosine_partial_f32(query, row_slice);
            *out.get_unchecked_mut(row) = super::finalize_cosine_f32(dot, q_norm_sq, row_sq);
        }
    }

    /// Pre-normalised one-to-many: writes `clip(dot(query, matrix[i]), -1, 1)`
    /// into `out[i]`. One FMA per element.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn cosine_one_to_many_normalized_into(
        query: &[f32],
        matrix: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        let n_rows = out.len();
        for row in 0..n_rows {
            let row_slice = matrix.get_unchecked(row * dim..(row + 1) * dim);
            *out.get_unchecked_mut(row) = dot_f32(query, row_slice).clamp(-1.0, 1.0);
        }
    }

    /// Generic adjacent: writes `cosine(matrix[i], matrix[i+1])` into
    /// `out[i]` for `i ∈ [0, n_rows - 1)`. Each pair runs the 3-FMA
    /// fused kernel (norms aren't shared across pairs).
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn cosine_adjacent_into(matrix: &[f32], dim: usize, out: &mut [f32]) {
        let n_pairs = out.len();
        for i in 0..n_pairs {
            let a = matrix.get_unchecked(i * dim..(i + 1) * dim);
            let b = matrix.get_unchecked((i + 1) * dim..(i + 2) * dim);
            let (dot, na, nb) = cosine_f32_fused(a, b);
            *out.get_unchecked_mut(i) = super::finalize_cosine_f32(dot, na, nb);
        }
    }

    /// Pre-normalised adjacent: `clip(dot(matrix[i], matrix[i+1]), -1, 1)`.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn cosine_adjacent_normalized_into(
        matrix: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        let n_pairs = out.len();
        for i in 0..n_pairs {
            let a = matrix.get_unchecked(i * dim..(i + 1) * dim);
            let b = matrix.get_unchecked((i + 1) * dim..(i + 2) * dim);
            *out.get_unchecked_mut(i) = dot_f32(a, b).clamp(-1.0, 1.0);
        }
    }
}

// ============================================================================
// AVX-512F (x86_64)
// ============================================================================
//
// 16-wide f32 FMA. Same pattern as AVX2, just twice the lanes.
// Lifted from NumKong's `nk_angular_f16_skylake`
// (spatial/skylake.h:317). We use `_mm512_maskz_loadu_ps` for the
// tail so we don't need a scalar epilogue.

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use std::arch::x86_64::*;

    /// Horizontal sum of an `__m512` via the AVX-512 reduce intrinsic.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn reduce_add_f32x16(v: __m512) -> f32 {
        _mm512_reduce_add_ps(v)
    }

    /// Fused `(dot, a², b²)`, 16-wide, 2× unrolled. Mask-loads the tail.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn cosine_f32_fused(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut i = 0;
        let mut dot0 = _mm512_setzero_ps();
        let mut dot1 = _mm512_setzero_ps();
        let mut na0 = _mm512_setzero_ps();
        let mut na1 = _mm512_setzero_ps();
        let mut nb0 = _mm512_setzero_ps();
        let mut nb1 = _mm512_setzero_ps();
        while i + 32 <= n {
            let av0 = _mm512_loadu_ps(a.as_ptr().add(i));
            let bv0 = _mm512_loadu_ps(b.as_ptr().add(i));
            let av1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
            let bv1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
            dot0 = _mm512_fmadd_ps(av0, bv0, dot0);
            dot1 = _mm512_fmadd_ps(av1, bv1, dot1);
            na0 = _mm512_fmadd_ps(av0, av0, na0);
            na1 = _mm512_fmadd_ps(av1, av1, na1);
            nb0 = _mm512_fmadd_ps(bv0, bv0, nb0);
            nb1 = _mm512_fmadd_ps(bv1, bv1, nb1);
            i += 32;
        }
        if i + 16 <= n {
            let av = _mm512_loadu_ps(a.as_ptr().add(i));
            let bv = _mm512_loadu_ps(b.as_ptr().add(i));
            dot0 = _mm512_fmadd_ps(av, bv, dot0);
            na0 = _mm512_fmadd_ps(av, av, na0);
            nb0 = _mm512_fmadd_ps(bv, bv, nb0);
            i += 16;
        }
        if i < n {
            let rem = (n - i) as u32;
            // `(1 << rem) - 1` is safe when rem < 16.
            let mask = ((1u32 << rem) - 1) as u16;
            let av = _mm512_maskz_loadu_ps(mask, a.as_ptr().add(i));
            let bv = _mm512_maskz_loadu_ps(mask, b.as_ptr().add(i));
            dot0 = _mm512_fmadd_ps(av, bv, dot0);
            na0 = _mm512_fmadd_ps(av, av, na0);
            nb0 = _mm512_fmadd_ps(bv, bv, nb0);
        }
        (
            reduce_add_f32x16(_mm512_add_ps(dot0, dot1)),
            reduce_add_f32x16(_mm512_add_ps(na0, na1)),
            reduce_add_f32x16(_mm512_add_ps(nb0, nb1)),
        )
    }

    /// Partial fused pass: `(dot, b²)`, 16-wide, 2× unrolled.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn cosine_partial_f32(a: &[f32], b: &[f32]) -> (f32, f32) {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut i = 0;
        let mut dot0 = _mm512_setzero_ps();
        let mut dot1 = _mm512_setzero_ps();
        let mut nb0 = _mm512_setzero_ps();
        let mut nb1 = _mm512_setzero_ps();
        while i + 32 <= n {
            let av0 = _mm512_loadu_ps(a.as_ptr().add(i));
            let bv0 = _mm512_loadu_ps(b.as_ptr().add(i));
            let av1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
            let bv1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
            dot0 = _mm512_fmadd_ps(av0, bv0, dot0);
            dot1 = _mm512_fmadd_ps(av1, bv1, dot1);
            nb0 = _mm512_fmadd_ps(bv0, bv0, nb0);
            nb1 = _mm512_fmadd_ps(bv1, bv1, nb1);
            i += 32;
        }
        if i + 16 <= n {
            let av = _mm512_loadu_ps(a.as_ptr().add(i));
            let bv = _mm512_loadu_ps(b.as_ptr().add(i));
            dot0 = _mm512_fmadd_ps(av, bv, dot0);
            nb0 = _mm512_fmadd_ps(bv, bv, nb0);
            i += 16;
        }
        if i < n {
            let rem = (n - i) as u32;
            let mask = ((1u32 << rem) - 1) as u16;
            let av = _mm512_maskz_loadu_ps(mask, a.as_ptr().add(i));
            let bv = _mm512_maskz_loadu_ps(mask, b.as_ptr().add(i));
            dot0 = _mm512_fmadd_ps(av, bv, dot0);
            nb0 = _mm512_fmadd_ps(bv, bv, nb0);
        }
        (
            reduce_add_f32x16(_mm512_add_ps(dot0, dot1)),
            reduce_add_f32x16(_mm512_add_ps(nb0, nb1)),
        )
    }

    /// Plain dot product, 16-wide, 2× unrolled.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut i = 0;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        while i + 32 <= n {
            let av0 = _mm512_loadu_ps(a.as_ptr().add(i));
            let bv0 = _mm512_loadu_ps(b.as_ptr().add(i));
            let av1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
            let bv1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
            acc0 = _mm512_fmadd_ps(av0, bv0, acc0);
            acc1 = _mm512_fmadd_ps(av1, bv1, acc1);
            i += 32;
        }
        if i + 16 <= n {
            let av = _mm512_loadu_ps(a.as_ptr().add(i));
            let bv = _mm512_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm512_fmadd_ps(av, bv, acc0);
            i += 16;
        }
        if i < n {
            let rem = (n - i) as u32;
            let mask = ((1u32 << rem) - 1) as u16;
            let av = _mm512_maskz_loadu_ps(mask, a.as_ptr().add(i));
            let bv = _mm512_maskz_loadu_ps(mask, b.as_ptr().add(i));
            acc0 = _mm512_fmadd_ps(av, bv, acc0);
        }
        reduce_add_f32x16(_mm512_add_ps(acc0, acc1))
    }

    /// Squared L2 norm, 16-wide.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn norm_sq_f32(v: &[f32]) -> f32 {
        let n = v.len();
        let mut i = 0;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        while i + 32 <= n {
            let x0 = _mm512_loadu_ps(v.as_ptr().add(i));
            let x1 = _mm512_loadu_ps(v.as_ptr().add(i + 16));
            acc0 = _mm512_fmadd_ps(x0, x0, acc0);
            acc1 = _mm512_fmadd_ps(x1, x1, acc1);
            i += 32;
        }
        if i + 16 <= n {
            let x = _mm512_loadu_ps(v.as_ptr().add(i));
            acc0 = _mm512_fmadd_ps(x, x, acc0);
            i += 16;
        }
        if i < n {
            let rem = (n - i) as u32;
            let mask = ((1u32 << rem) - 1) as u16;
            let x = _mm512_maskz_loadu_ps(mask, v.as_ptr().add(i));
            acc0 = _mm512_fmadd_ps(x, x, acc0);
        }
        reduce_add_f32x16(_mm512_add_ps(acc0, acc1))
    }

    // ------------------------------------------------------------
    // Batch entry points
    // ------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn cosine_one_to_many_into(
        query: &[f32],
        matrix: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        let q_norm_sq = norm_sq_f32(query);
        let n_rows = out.len();
        if q_norm_sq <= 0.0 {
            for slot in out.iter_mut() {
                *slot = 0.0;
            }
            return;
        }
        for row in 0..n_rows {
            let row_slice = matrix.get_unchecked(row * dim..(row + 1) * dim);
            let (dot, row_sq) = cosine_partial_f32(query, row_slice);
            *out.get_unchecked_mut(row) = super::finalize_cosine_f32(dot, q_norm_sq, row_sq);
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn cosine_one_to_many_normalized_into(
        query: &[f32],
        matrix: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        let n_rows = out.len();
        for row in 0..n_rows {
            let row_slice = matrix.get_unchecked(row * dim..(row + 1) * dim);
            *out.get_unchecked_mut(row) = dot_f32(query, row_slice).clamp(-1.0, 1.0);
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn cosine_adjacent_into(matrix: &[f32], dim: usize, out: &mut [f32]) {
        let n_pairs = out.len();
        for i in 0..n_pairs {
            let a = matrix.get_unchecked(i * dim..(i + 1) * dim);
            let b = matrix.get_unchecked((i + 1) * dim..(i + 2) * dim);
            let (dot, na, nb) = cosine_f32_fused(a, b);
            *out.get_unchecked_mut(i) = super::finalize_cosine_f32(dot, na, nb);
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn cosine_adjacent_normalized_into(
        matrix: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        let n_pairs = out.len();
        for i in 0..n_pairs {
            let a = matrix.get_unchecked(i * dim..(i + 1) * dim);
            let b = matrix.get_unchecked((i + 1) * dim..(i + 2) * dim);
            *out.get_unchecked_mut(i) = dot_f32(a, b).clamp(-1.0, 1.0);
        }
    }
}

// ============================================================================
// NEON (aarch64)
// ============================================================================
//
// 4-wide f32 FMA via `vfmaq_f32`. Pattern from NumKong
// `nk_angular_f16_neon` (spatial/neon.h:354): three
// `vdupq_n_f32(0)` accumulators, one `vld1q_f32` per operand per
// iteration, three `vfmaq_f32` per iteration, `vaddvq_f32` to fold
// to scalar at the end. We unroll 2× because Apple M-series and
// Graviton 3+ have 4 SIMD FP pipes — the extra accumulator hides
// the FMA latency.

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Fused `(dot, a², b²)`. 4-wide FMA, 2× unrolled.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn cosine_f32_fused(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut i = 0;
        let mut dot0 = vdupq_n_f32(0.0);
        let mut dot1 = vdupq_n_f32(0.0);
        let mut na0 = vdupq_n_f32(0.0);
        let mut na1 = vdupq_n_f32(0.0);
        let mut nb0 = vdupq_n_f32(0.0);
        let mut nb1 = vdupq_n_f32(0.0);
        while i + 8 <= n {
            let av0 = vld1q_f32(a.as_ptr().add(i));
            let bv0 = vld1q_f32(b.as_ptr().add(i));
            let av1 = vld1q_f32(a.as_ptr().add(i + 4));
            let bv1 = vld1q_f32(b.as_ptr().add(i + 4));
            dot0 = vfmaq_f32(dot0, av0, bv0);
            dot1 = vfmaq_f32(dot1, av1, bv1);
            na0 = vfmaq_f32(na0, av0, av0);
            na1 = vfmaq_f32(na1, av1, av1);
            nb0 = vfmaq_f32(nb0, bv0, bv0);
            nb1 = vfmaq_f32(nb1, bv1, bv1);
            i += 8;
        }
        if i + 4 <= n {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            dot0 = vfmaq_f32(dot0, av, bv);
            na0 = vfmaq_f32(na0, av, av);
            nb0 = vfmaq_f32(nb0, bv, bv);
            i += 4;
        }
        let mut dot_s = vaddvq_f32(vaddq_f32(dot0, dot1));
        let mut na_s = vaddvq_f32(vaddq_f32(na0, na1));
        let mut nb_s = vaddvq_f32(vaddq_f32(nb0, nb1));
        while i < n {
            let ai = *a.get_unchecked(i);
            let bi = *b.get_unchecked(i);
            dot_s = ai.mul_add(bi, dot_s);
            na_s = ai.mul_add(ai, na_s);
            nb_s = bi.mul_add(bi, nb_s);
            i += 1;
        }
        (dot_s, na_s, nb_s)
    }

    /// Partial fused pass: `(dot, b²)`. 4-wide FMA, 2× unrolled.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn cosine_partial_f32(a: &[f32], b: &[f32]) -> (f32, f32) {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut i = 0;
        let mut dot0 = vdupq_n_f32(0.0);
        let mut dot1 = vdupq_n_f32(0.0);
        let mut nb0 = vdupq_n_f32(0.0);
        let mut nb1 = vdupq_n_f32(0.0);
        while i + 8 <= n {
            let av0 = vld1q_f32(a.as_ptr().add(i));
            let bv0 = vld1q_f32(b.as_ptr().add(i));
            let av1 = vld1q_f32(a.as_ptr().add(i + 4));
            let bv1 = vld1q_f32(b.as_ptr().add(i + 4));
            dot0 = vfmaq_f32(dot0, av0, bv0);
            dot1 = vfmaq_f32(dot1, av1, bv1);
            nb0 = vfmaq_f32(nb0, bv0, bv0);
            nb1 = vfmaq_f32(nb1, bv1, bv1);
            i += 8;
        }
        if i + 4 <= n {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            dot0 = vfmaq_f32(dot0, av, bv);
            nb0 = vfmaq_f32(nb0, bv, bv);
            i += 4;
        }
        let mut dot_s = vaddvq_f32(vaddq_f32(dot0, dot1));
        let mut nb_s = vaddvq_f32(vaddq_f32(nb0, nb1));
        while i < n {
            let ai = *a.get_unchecked(i);
            let bi = *b.get_unchecked(i);
            dot_s = ai.mul_add(bi, dot_s);
            nb_s = bi.mul_add(bi, nb_s);
            i += 1;
        }
        (dot_s, nb_s)
    }

    /// Plain dot product, 4-wide, 4× unrolled.
    /// (Apple M-series & Graviton 3+ have 4 FP pipes; 4 accumulators
    /// saturate the FMA throughput at ~4 cycles latency.)
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut i = 0;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        while i + 16 <= n {
            let av0 = vld1q_f32(a.as_ptr().add(i));
            let bv0 = vld1q_f32(b.as_ptr().add(i));
            let av1 = vld1q_f32(a.as_ptr().add(i + 4));
            let bv1 = vld1q_f32(b.as_ptr().add(i + 4));
            let av2 = vld1q_f32(a.as_ptr().add(i + 8));
            let bv2 = vld1q_f32(b.as_ptr().add(i + 8));
            let av3 = vld1q_f32(a.as_ptr().add(i + 12));
            let bv3 = vld1q_f32(b.as_ptr().add(i + 12));
            acc0 = vfmaq_f32(acc0, av0, bv0);
            acc1 = vfmaq_f32(acc1, av1, bv1);
            acc2 = vfmaq_f32(acc2, av2, bv2);
            acc3 = vfmaq_f32(acc3, av3, bv3);
            i += 16;
        }
        if i + 8 <= n {
            let av0 = vld1q_f32(a.as_ptr().add(i));
            let bv0 = vld1q_f32(b.as_ptr().add(i));
            let av1 = vld1q_f32(a.as_ptr().add(i + 4));
            let bv1 = vld1q_f32(b.as_ptr().add(i + 4));
            acc0 = vfmaq_f32(acc0, av0, bv0);
            acc1 = vfmaq_f32(acc1, av1, bv1);
            i += 8;
        }
        if i + 4 <= n {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            acc0 = vfmaq_f32(acc0, av, bv);
            i += 4;
        }
        let sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut s = vaddvq_f32(sum);
        while i < n {
            let ai = *a.get_unchecked(i);
            let bi = *b.get_unchecked(i);
            s = ai.mul_add(bi, s);
            i += 1;
        }
        s
    }

    /// Squared L2 norm, 4-wide, 2× unrolled.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn norm_sq_f32(v: &[f32]) -> f32 {
        let n = v.len();
        let mut i = 0;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        while i + 8 <= n {
            let x0 = vld1q_f32(v.as_ptr().add(i));
            let x1 = vld1q_f32(v.as_ptr().add(i + 4));
            acc0 = vfmaq_f32(acc0, x0, x0);
            acc1 = vfmaq_f32(acc1, x1, x1);
            i += 8;
        }
        if i + 4 <= n {
            let x = vld1q_f32(v.as_ptr().add(i));
            acc0 = vfmaq_f32(acc0, x, x);
            i += 4;
        }
        let mut s = vaddvq_f32(vaddq_f32(acc0, acc1));
        while i < n {
            let x = *v.get_unchecked(i);
            s = x.mul_add(x, s);
            i += 1;
        }
        s
    }

    // ------------------------------------------------------------
    // Batch entry points
    // ------------------------------------------------------------

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn cosine_one_to_many_into(
        query: &[f32],
        matrix: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        let q_norm_sq = norm_sq_f32(query);
        let n_rows = out.len();
        if q_norm_sq <= 0.0 {
            for slot in out.iter_mut() {
                *slot = 0.0;
            }
            return;
        }
        for row in 0..n_rows {
            let row_slice = matrix.get_unchecked(row * dim..(row + 1) * dim);
            let (dot, row_sq) = cosine_partial_f32(query, row_slice);
            *out.get_unchecked_mut(row) = super::finalize_cosine_f32(dot, q_norm_sq, row_sq);
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn cosine_one_to_many_normalized_into(
        query: &[f32],
        matrix: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        let n_rows = out.len();
        for row in 0..n_rows {
            let row_slice = matrix.get_unchecked(row * dim..(row + 1) * dim);
            *out.get_unchecked_mut(row) = dot_f32(query, row_slice).clamp(-1.0, 1.0);
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn cosine_adjacent_into(matrix: &[f32], dim: usize, out: &mut [f32]) {
        let n_pairs = out.len();
        for i in 0..n_pairs {
            let a = matrix.get_unchecked(i * dim..(i + 1) * dim);
            let b = matrix.get_unchecked((i + 1) * dim..(i + 2) * dim);
            let (dot, na, nb) = cosine_f32_fused(a, b);
            *out.get_unchecked_mut(i) = super::finalize_cosine_f32(dot, na, nb);
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn cosine_adjacent_normalized_into(
        matrix: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        let n_pairs = out.len();
        for i in 0..n_pairs {
            let a = matrix.get_unchecked(i * dim..(i + 1) * dim);
            let b = matrix.get_unchecked((i + 1) * dim..(i + 2) * dim);
            *out.get_unchecked_mut(i) = dot_f32(a, b).clamp(-1.0, 1.0);
        }
    }
}

// ============================================================================
// Scalar batch entry points
// ============================================================================
//
// Used as the fallback when no SIMD ISA is detected. Same row-loop
// layout as the SIMD batch kernels, just calling `scalar::*` per row.

mod scalar_batch {
    use super::scalar;

    #[inline]
    pub(super) fn cosine_one_to_many_into(
        query: &[f32],
        matrix: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        let q_norm_sq = scalar::norm_sq_f32(query);
        let n_rows = out.len();
        if q_norm_sq <= 0.0 {
            for slot in out.iter_mut() {
                *slot = 0.0;
            }
            return;
        }
        for row in 0..n_rows {
            let row_slice = &matrix[row * dim..(row + 1) * dim];
            let (dot, row_sq) = scalar::cosine_partial_f32(query, row_slice);
            out[row] = super::finalize_cosine_f32(dot, q_norm_sq, row_sq);
        }
    }

    #[inline]
    pub(super) fn cosine_one_to_many_normalized_into(
        query: &[f32],
        matrix: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        let n_rows = out.len();
        for row in 0..n_rows {
            let row_slice = &matrix[row * dim..(row + 1) * dim];
            out[row] = scalar::dot_f32(query, row_slice).clamp(-1.0, 1.0);
        }
    }

    #[inline]
    pub(super) fn cosine_adjacent_into(matrix: &[f32], dim: usize, out: &mut [f32]) {
        let n_pairs = out.len();
        for i in 0..n_pairs {
            let a = &matrix[i * dim..(i + 1) * dim];
            let b = &matrix[(i + 1) * dim..(i + 2) * dim];
            let (dot, na, nb) = scalar::cosine_f32_fused(a, b);
            out[i] = super::finalize_cosine_f32(dot, na, nb);
        }
    }

    #[inline]
    pub(super) fn cosine_adjacent_normalized_into(matrix: &[f32], dim: usize, out: &mut [f32]) {
        let n_pairs = out.len();
        for i in 0..n_pairs {
            let a = &matrix[i * dim..(i + 1) * dim];
            let b = &matrix[(i + 1) * dim..(i + 2) * dim];
            out[i] = scalar::dot_f32(a, b).clamp(-1.0, 1.0);
        }
    }
}

// ============================================================================
// Public API — single-pair primitives (dispatched)
// ============================================================================
//
// Every public function:
//
//   1. Reads the cached dispatch flag (one atomic load on the hot path
//      — `OnceLock::get` is one `Acquire` after init; we then deref a
//      stack-local copy from the static).
//   2. `match`es on it and calls into the right `target_feature`
//      gated `unsafe fn`. The `unsafe` blocks are sound because the
//      dispatch flag is set only after the runtime feature check
//      passed.
//   3. Does the final scalar normalisation (rsqrt + clamp) outside the
//      SIMD path — it's `O(1)` and not on the inner loop.

/// Fused single-pass `(dot, ‖a‖², ‖b‖²)` over equal-length f32 slices.
/// One SIMD pass over each input pair — three FMAs per lane per
/// iteration.
#[inline]
pub fn cosine_components_f32(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    match dispatch() {
        Dispatch::Scalar => scalar::cosine_f32_fused(a, b),
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx2Fma => unsafe { avx2::cosine_f32_fused(a, b) },
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx512f => unsafe { avx512::cosine_f32_fused(a, b) },
        #[cfg(target_arch = "aarch64")]
        Dispatch::Neon => unsafe { neon::cosine_f32_fused(a, b) },
    }
}

/// Plain dot product over equal-length f32 slices. Used by the
/// pre-normalised fast path: cosine of two unit-norm vectors **is**
/// their dot product.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    match dispatch() {
        Dispatch::Scalar => scalar::dot_f32(a, b),
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx2Fma => unsafe { avx2::dot_f32(a, b) },
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx512f => unsafe { avx512::dot_f32(a, b) },
        #[cfg(target_arch = "aarch64")]
        Dispatch::Neon => unsafe { neon::dot_f32(a, b) },
    }
}

/// Squared L2 norm of an f32 slice. Used by `l2_normalize_in_place`
/// and by `cosine_one_to_many` for the once-per-call query norm.
#[inline]
pub fn norm_sq_f32(v: &[f32]) -> f32 {
    match dispatch() {
        Dispatch::Scalar => scalar::norm_sq_f32(v),
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx2Fma => unsafe { avx2::norm_sq_f32(v) },
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx512f => unsafe { avx512::norm_sq_f32(v) },
        #[cfg(target_arch = "aarch64")]
        Dispatch::Neon => unsafe { neon::norm_sq_f32(v) },
    }
}

/// Cosine similarity in `[-1, 1]` between two equal-length f32 slices.
/// Zero-norm vectors return `0.0`.
///
/// Generic path — computes both norms. If the caller knows both inputs
/// are already unit-norm, prefer [`cosine_f32_normalized`] which skips
/// ~30% of the FLOPs (two squared-norm reductions + a rsqrt).
#[inline]
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (dot, na_sq, nb_sq) = cosine_components_f32(a, b);
    finalize_cosine_f32(dot, na_sq, nb_sq)
}

/// Pre-normalised fast path: cosine of two **unit-norm** f32 vectors.
///
/// Computes `dot(a, b)` and clamps to `[-1, 1]`. **The caller is
/// responsible for guaranteeing unit-norm inputs** — this function
/// does not verify the norm (that would defeat the perf gain) and
/// will return an incorrect cosine if the contract is violated.
///
/// At our production shapes (`dim ∈ {256, 384, 768}`, unit-norm rows)
/// this path saves both squared-norm passes and the final scalar
/// `rsqrt`. SemanticChunker and ExtractiveRanker should call this.
#[inline]
pub fn cosine_f32_normalized(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    dot_f32(a, b).clamp(-1.0, 1.0)
}

/// Finalise a fused `(dot, ‖a‖², ‖b‖²)` triple into a cosine value
/// in `[-1, 1]`. Zero-norm vectors yield `0.0`.
///
/// Lifts the structure of NumKong's `nk_angular_normalize_f32_haswell_`
/// (spatial/haswell.h:167): branch on zero-norm, otherwise compute
/// `1 / √(a² × b²)` once via `rsqrt`. We use scalar `f64::sqrt` here
/// rather than `_mm_rsqrt_ps` because (a) it's executed exactly once
/// per call, not per SIMD lane, so a hardware rsqrt approximation
/// buys us nothing, and (b) the scalar path needs to be bit-stable
/// across the ISA dispatch, which `_mm_rsqrt_ps` is not (its
/// approximation differs between μarchs and from NEON's `vrsqrte_f32`).
#[inline]
pub fn finalize_cosine_f32(dot: f32, na_sq: f32, nb_sq: f32) -> f32 {
    if na_sq <= 0.0 || nb_sq <= 0.0 {
        return 0.0;
    }
    // Compute in f64 for the final divide — keeps the cosine bit-stable
    // across dispatch buckets, and the cost is one f64 sqrt + one f64
    // divide per call (not per row, not per lane).
    let denom = ((na_sq as f64) * (nb_sq as f64)).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    let cos = (dot as f64) / denom;
    cos.clamp(-1.0, 1.0) as f32
}

// ============================================================================
// Public API — batch primitives (dispatched once per call)
// ============================================================================
//
// These are the hot-path entry points for the consumer surface in
// `dense.rs`. Each one dispatches ONCE per call and runs the per-row
// inner loop inside a single `unsafe fn` marked
// `#[target_feature(enable = "...")]`, so:
//
//   * The dispatch decision is hoisted out of the row loop.
//   * SIMD registers and the cached query norm stay alive across rows.
//   * Per-row inner FMA count is 2 (one-to-many generic), 3 (adjacent
//     generic), or 1 (normalized fast paths).
//   * Output is written into a caller-provided slice via raw pointer
//     stores — no `Vec::push`, no realloc.
//
// Safety contract on the batch fns:
//
//   - `matrix.len() == n_rows * dim`
//   - `out.len() == n_rows` (or `n_rows - 1` for adjacent)
//   - `query.len() == dim`
//
// The public functions in `dense.rs` validate these *before* calling
// in; the kernels here trust the contract and use unchecked slicing.

/// Generic one-to-many cosine. Writes `cosine(query, matrix[i])` into
/// `out[i]`. Pre-normalised fast path lives below as
/// [`cosine_one_to_many_normalized_into`].
///
/// # Safety contract
///
/// * `matrix.len() == out.len() * dim`
/// * `query.len() == dim`
///
/// Violating the contract is undefined behaviour inside the SIMD paths.
#[inline]
pub fn cosine_one_to_many_into(query: &[f32], matrix: &[f32], dim: usize, out: &mut [f32]) {
    match dispatch() {
        Dispatch::Scalar => scalar_batch::cosine_one_to_many_into(query, matrix, dim, out),
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx2Fma => unsafe { avx2::cosine_one_to_many_into(query, matrix, dim, out) },
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx512f => unsafe { avx512::cosine_one_to_many_into(query, matrix, dim, out) },
        #[cfg(target_arch = "aarch64")]
        Dispatch::Neon => unsafe { neon::cosine_one_to_many_into(query, matrix, dim, out) },
    }
}

/// Pre-normalised one-to-many cosine. Writes `clamp(dot, -1, 1)` into
/// `out[i]`. Caller guarantees unit-norm inputs.
#[inline]
pub fn cosine_one_to_many_normalized_into(
    query: &[f32],
    matrix: &[f32],
    dim: usize,
    out: &mut [f32],
) {
    match dispatch() {
        Dispatch::Scalar => {
            scalar_batch::cosine_one_to_many_normalized_into(query, matrix, dim, out)
        }
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx2Fma => unsafe {
            avx2::cosine_one_to_many_normalized_into(query, matrix, dim, out)
        },
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx512f => unsafe {
            avx512::cosine_one_to_many_normalized_into(query, matrix, dim, out)
        },
        #[cfg(target_arch = "aarch64")]
        Dispatch::Neon => unsafe {
            neon::cosine_one_to_many_normalized_into(query, matrix, dim, out)
        },
    }
}

/// Generic adjacent-row cosine. Writes
/// `cosine(matrix[i], matrix[i + 1])` into `out[i]` for
/// `i ∈ [0, n_rows - 1)`.
#[inline]
pub fn cosine_adjacent_into(matrix: &[f32], dim: usize, out: &mut [f32]) {
    match dispatch() {
        Dispatch::Scalar => scalar_batch::cosine_adjacent_into(matrix, dim, out),
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx2Fma => unsafe { avx2::cosine_adjacent_into(matrix, dim, out) },
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx512f => unsafe { avx512::cosine_adjacent_into(matrix, dim, out) },
        #[cfg(target_arch = "aarch64")]
        Dispatch::Neon => unsafe { neon::cosine_adjacent_into(matrix, dim, out) },
    }
}

/// Pre-normalised adjacent-row cosine. Writes
/// `clamp(dot(matrix[i], matrix[i + 1]), -1, 1)` into `out[i]`.
#[inline]
pub fn cosine_adjacent_normalized_into(matrix: &[f32], dim: usize, out: &mut [f32]) {
    match dispatch() {
        Dispatch::Scalar => scalar_batch::cosine_adjacent_normalized_into(matrix, dim, out),
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx2Fma => unsafe { avx2::cosine_adjacent_normalized_into(matrix, dim, out) },
        #[cfg(target_arch = "x86_64")]
        Dispatch::Avx512f => unsafe { avx512::cosine_adjacent_normalized_into(matrix, dim, out) },
        #[cfg(target_arch = "aarch64")]
        Dispatch::Neon => unsafe { neon::cosine_adjacent_normalized_into(matrix, dim, out) },
    }
}

// ============================================================================
// Back-compat wrappers
// ============================================================================
//
// The pre-existing `dense.rs` callers used `dot_f32_to_f64` and
// `norm_sq_f32_to_f64`. We keep those names as thin shims so the
// `dense.rs` diff stays small — they now accumulate in f32 SIMD and
// widen to f64 at the end, which matches the public contract (the
// only observable difference is improved throughput).

#[inline]
pub fn dot_f32_to_f64(a: &[f32], b: &[f32]) -> f64 {
    dot_f32(a, b) as f64
}

#[inline]
pub fn norm_sq_f32_to_f64(v: &[f32]) -> f64 {
    norm_sq_f32(v) as f64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    #[test]
    fn dispatch_picks_a_bucket() {
        // Whatever bucket we get, the kernels must all answer.
        let _ = dispatch();
    }

    #[test]
    fn cosine_components_match_naive_at_small_size() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let (dot, na, nb) = cosine_components_f32(&a, &b);
        assert!((dot - 32.0).abs() < 1e-5);
        assert!((na - 14.0).abs() < 1e-5);
        assert!((nb - 77.0).abs() < 1e-5);
    }

    #[test]
    fn dot_matches_naive() {
        // 17 elements — exercises full chunk + tail on every dispatch.
        let a: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..17).map(|i| (17 - i) as f32).collect();
        let expected: f32 = (0..17).map(|i| (i * (17 - i)) as f32).sum();
        assert!((dot_f32(&a, &b) - expected).abs() < 1e-4);
    }

    #[test]
    fn dot_matches_naive_long() {
        // 1003 elements — exercises 4× unrolled main loop + 16-wide
        // tail + 8-wide tail + scalar tail on AVX2/AVX-512.
        let a: Vec<f32> = (0..1003).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..1003).map(|i| ((1003 - i) as f32) * 0.001).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let got = dot_f32(&a, &b);
        assert!((got - expected).abs() < 1e-3, "got {got} vs {expected}");
    }

    #[test]
    fn norm_sq_matches_naive() {
        let v: Vec<f32> = (0..50).map(|i| (i as f32) * 0.1).collect();
        let expected: f32 = v.iter().map(|x| x * x).sum();
        let got = norm_sq_f32(&v);
        assert!((got - expected).abs() < 1e-3);
    }

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let c = cosine_f32(&v, &v);
        assert!((c - 1.0).abs() < TOL);
    }

    #[test]
    fn cosine_opposite_is_minus_one() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![-1.0_f32, -2.0, -3.0];
        let c = cosine_f32(&a, &b);
        assert!((c + 1.0).abs() < TOL);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let c = cosine_f32(&a, &b);
        assert!(c.abs() < TOL);
    }

    #[test]
    fn cosine_zero_norm_returns_zero() {
        let z = vec![0.0_f32; 4];
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(cosine_f32(&z, &v), 0.0);
        assert_eq!(cosine_f32(&v, &z), 0.0);
    }

    #[test]
    fn cosine_clipped_to_unit_interval() {
        let v = vec![0.5773_f32; 256];
        let c = cosine_f32(&v, &v);
        assert!((-1.0..=1.0).contains(&c));
        assert!((c - 1.0).abs() < TOL);
    }

    #[test]
    fn cosine_known_value() {
        let a = [1.0_f32, 0.0, 0.0, 0.0];
        let b = [1.0_f32, 1.0, 0.0, 0.0];
        let c = cosine_f32(&a, &b);
        assert!((c - std::f32::consts::FRAC_1_SQRT_2).abs() < TOL);
    }

    #[test]
    fn cosine_normalized_fast_path_matches_generic_on_unit_norm_input() {
        // Construct a 384-d unit-norm pair, check the fast path
        // matches the generic path to within f32 epsilon * sqrt(dim).
        let a: Vec<f32> = (0..384).map(|i| ((i as f32) - 192.0) / 384.0).collect();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let a_u: Vec<f32> = a.iter().map(|x| x / na).collect();
        let b: Vec<f32> = (0..384)
            .map(|i| (((i as f32) - 100.0).sin()).abs() + 0.1)
            .collect();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let b_u: Vec<f32> = b.iter().map(|x| x / nb).collect();

        let generic = cosine_f32(&a_u, &b_u);
        let fast = cosine_f32_normalized(&a_u, &b_u);
        assert!(
            (generic - fast).abs() < 1e-5,
            "generic={generic}, fast={fast}, delta={}",
            (generic - fast).abs()
        );
    }

    #[test]
    fn cosine_normalized_clamps_to_unit_interval() {
        // Unit vector dotted with itself → 1.0; if a caller violates
        // the contract slightly, we still clamp.
        let v = vec![0.5_f32; 4]; // norm = 1.0
        let c = cosine_f32_normalized(&v, &v);
        assert!((-1.0..=1.0).contains(&c));
        assert!((c - 1.0).abs() < TOL);
    }

    #[test]
    fn finalize_handles_zero_norm() {
        assert_eq!(finalize_cosine_f32(1.0, 0.0, 1.0), 0.0);
        assert_eq!(finalize_cosine_f32(1.0, 1.0, 0.0), 0.0);
        assert_eq!(finalize_cosine_f32(0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn cross_isa_bit_stability_smoke() {
        // The same input pair, run through the dispatched cosine, must
        // match the scalar reference within f32 tolerance even if a
        // wider ISA is selected — the f64 finalize keeps the divide
        // bit-stable across buckets.
        let n = 384;
        let a: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) / (n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| ((n - i) as f32) / (n as f32)).collect();
        let dispatched = cosine_f32(&a, &b);
        let scalar_ref = {
            let (d, na, nb) = scalar::cosine_f32_fused(&a, &b);
            finalize_cosine_f32(d, na, nb)
        };
        assert!((dispatched - scalar_ref).abs() < 1e-5);
    }

    #[test]
    fn long_dot_matches_naive_sum() {
        // 2000 elements — exercises the production "chunker" size at
        // dim=384 → 2000 × 384 = 768k floats per matrix. Verify the
        // FMA accumulators don't drift.
        let n = 2000;
        let a: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001).cos()).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001).sin()).collect();
        // Reference: f64 accumulation to keep the ground truth high-precision.
        let ref_dot: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum();
        let got = dot_f32(&a, &b);
        assert!(
            ((got as f64) - ref_dot).abs() < 1e-3,
            "got {got} vs {ref_dot}"
        );
    }

    // ------------------------------------------------------------------
    // Batch kernel tests — exercise the dispatched-once batch entry
    // points. Each test checks both (a) bit-tolerance against a naive
    // reference and (b) that the dispatched batch path matches the
    // single-pair primitives row-by-row.
    // ------------------------------------------------------------------

    #[test]
    fn batch_one_to_many_matches_per_row_dispatch() {
        let dim = 384;
        let n_rows = 50;
        let q: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.013).cos()).collect();
        let mut matrix: Vec<f32> = Vec::with_capacity(n_rows * dim);
        for r in 0..n_rows {
            for c in 0..dim {
                matrix.push(((r * dim + c) as f32 * 0.0007).sin() + 0.1);
            }
        }
        let mut batch_out = vec![0.0_f32; n_rows];
        cosine_one_to_many_into(&q, &matrix, dim, &mut batch_out);
        // Reference: call the single-pair cosine row by row.
        for row in 0..n_rows {
            let row_slice = &matrix[row * dim..(row + 1) * dim];
            let single = cosine_f32(&q, row_slice);
            assert!(
                (batch_out[row] - single).abs() < 1e-5,
                "row {row}: batch={} single={}",
                batch_out[row],
                single
            );
        }
    }

    #[test]
    fn batch_one_to_many_normalized_matches_clamped_dot() {
        let dim = 256;
        let n_rows = 30;
        // Unit-norm random vectors.
        let mut q: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).cos()).collect();
        let qn = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        q.iter_mut().for_each(|x| *x /= qn);

        let mut matrix: Vec<f32> = Vec::with_capacity(n_rows * dim);
        for r in 0..n_rows {
            let row: Vec<f32> = (0..dim)
                .map(|c| ((r as f32) * 0.1 + (c as f32) * 0.003).sin() + 0.5)
                .collect();
            let n = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            matrix.extend(row.iter().map(|x| x / n));
        }
        let mut batch_out = vec![0.0_f32; n_rows];
        cosine_one_to_many_normalized_into(&q, &matrix, dim, &mut batch_out);
        // Reference: dot product row by row, clamped.
        for row in 0..n_rows {
            let row_slice = &matrix[row * dim..(row + 1) * dim];
            let single = cosine_f32_normalized(&q, row_slice);
            assert!(
                (batch_out[row] - single).abs() < 1e-5,
                "row {row}: batch={} single={}",
                batch_out[row],
                single
            );
        }
    }

    #[test]
    fn batch_one_to_many_zero_query_yields_zeros() {
        let dim = 64;
        let n_rows = 5;
        let q = vec![0.0_f32; dim];
        let matrix: Vec<f32> = (0..n_rows * dim).map(|i| (i as f32) * 0.001).collect();
        let mut batch_out = vec![1.0_f32; n_rows]; // sentinel
        cosine_one_to_many_into(&q, &matrix, dim, &mut batch_out);
        for x in &batch_out {
            assert_eq!(*x, 0.0);
        }
    }

    #[test]
    fn batch_adjacent_matches_per_row_dispatch() {
        let dim = 768;
        let n_rows = 20;
        let mut matrix: Vec<f32> = Vec::with_capacity(n_rows * dim);
        for r in 0..n_rows {
            for c in 0..dim {
                matrix.push(((r * dim + c) as f32 * 0.0011).cos() + 0.2);
            }
        }
        let mut batch_out = vec![0.0_f32; n_rows - 1];
        cosine_adjacent_into(&matrix, dim, &mut batch_out);
        for i in 0..(n_rows - 1) {
            let a = &matrix[i * dim..(i + 1) * dim];
            let b = &matrix[(i + 1) * dim..(i + 2) * dim];
            let single = cosine_f32(a, b);
            assert!(
                (batch_out[i] - single).abs() < 1e-5,
                "pair {i}: batch={} single={}",
                batch_out[i],
                single
            );
        }
    }

    #[test]
    fn batch_adjacent_normalized_matches_clamped_dot() {
        let dim = 384;
        let n_rows = 15;
        let mut matrix: Vec<f32> = Vec::with_capacity(n_rows * dim);
        for r in 0..n_rows {
            let row: Vec<f32> = (0..dim)
                .map(|c| ((r as f32) * 0.05 + (c as f32) * 0.002).cos() + 0.3)
                .collect();
            let n = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            matrix.extend(row.iter().map(|x| x / n));
        }
        let mut batch_out = vec![0.0_f32; n_rows - 1];
        cosine_adjacent_normalized_into(&matrix, dim, &mut batch_out);
        for i in 0..(n_rows - 1) {
            let a = &matrix[i * dim..(i + 1) * dim];
            let b = &matrix[(i + 1) * dim..(i + 2) * dim];
            let single = cosine_f32_normalized(a, b);
            assert!(
                (batch_out[i] - single).abs() < 1e-5,
                "pair {i}: batch={} single={}",
                batch_out[i],
                single
            );
        }
    }
}
