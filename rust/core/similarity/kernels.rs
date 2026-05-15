//! Portable f32 dot / cosine kernels with f64 accumulation.
//!
//! The numerical contract here is the one we borrowed from NumKong:
//!
//! * f32 inputs, f64 accumulators.
//! * Pairwise-style summation via 8 parallel f64 lanes -- LLVM
//!   auto-vectorises the inner loop to AVX2 / NEON on platforms that
//!   support it, and the 8 lanes give a `O(log N)` error bound for
//!   the accumulator drift that motivates Kahan / Neumaier
//!   compensation. For the dimensionalities we serve (256-1536) the
//!   accumulator drift sits well below f32 epsilon without explicit
//!   compensation.
//! * Cosine clipped to `[-1.0, 1.0]` to absorb f64 round-off near
//!   `+/- 1`.
//! * Zero-norm vectors yield cosine = 0.0 by convention.
//!
//! No external SIMD crate dependency: the only requirement on the
//! compiler is that it auto-vectorises a chunked loop over 8
//! parallel `f64` accumulators with `(x as f64) * (y as f64)` lanes,
//! which both rustc/LLVM on x86_64 (AVX2) and aarch64 (NEON) do at
//! `-C opt-level=3` (we ship `--release`). This unblocks Windows
//! builds that previously broke on numkong's vendored C (immintrin.h
//! ARM64 reject + DllMain symbol conflict with stringzilla).
//!
//! Why we did not depend on a third-party SIMD crate
//! ------------------------------------------------
//!
//! `wide`, `pulp`, `std::simd` and friends all add either a
//! non-trivial dependency surface or require nightly. The chunked
//! f64 accumulator pattern auto-vectorises today on every target we
//! ship a wheel for, with zero new deps. If a future workload
//! demands ISA-specific intrinsics (AVX-512 mask + reduce, SVE
//! gather), we can introduce a targeted SIMD crate then -- but not
//! across the entire similarity surface.

/// Dot product of two equal-length `f32` slices, accumulating in
/// `f64`.
///
/// Uses 8 parallel `f64` accumulators so LLVM can auto-vectorise the
/// hot loop and the partial sums give a pairwise-style error bound.
/// Tail elements that don't fit a chunk of 8 are folded into the
/// first accumulator. Callers must ensure `a.len() == b.len()`
/// (debug-asserted; production callers in this module check first).
#[inline]
pub fn dot_f32_to_f64(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    // 8 parallel accumulators; rustc auto-vectorises this to one AVX2
    // ymm-shaped reduction on x86_64 / one NEON 4-lane on aarch64.
    let mut acc = [0.0_f64; 8];
    let chunks = n / 8;
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 8;
        // Manually unroll so each `acc[j]` accumulates independently
        // -- this is what licences the autovec.
        acc[0] += (a[base] as f64) * (b[base] as f64);
        acc[1] += (a[base + 1] as f64) * (b[base + 1] as f64);
        acc[2] += (a[base + 2] as f64) * (b[base + 2] as f64);
        acc[3] += (a[base + 3] as f64) * (b[base + 3] as f64);
        acc[4] += (a[base + 4] as f64) * (b[base + 4] as f64);
        acc[5] += (a[base + 5] as f64) * (b[base + 5] as f64);
        acc[6] += (a[base + 6] as f64) * (b[base + 6] as f64);
        acc[7] += (a[base + 7] as f64) * (b[base + 7] as f64);
    }
    // Tail
    let tail_start = chunks * 8;
    for i in tail_start..n {
        acc[0] += (a[i] as f64) * (b[i] as f64);
    }
    // Pairwise reduction of the 8 lanes -- minimises further error
    // vs. a naive left-to-right sum.
    let s0 = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    let s1 = (acc[4] + acc[5]) + (acc[6] + acc[7]);
    s0 + s1
}

/// Squared L2 norm of an `f32` slice, accumulated in `f64`.
#[inline]
pub fn norm_sq_f32_to_f64(v: &[f32]) -> f64 {
    dot_f32_to_f64(v, v)
}

/// Cosine similarity in `[-1, 1]`. Zero-norm vectors return `0.0`.
#[inline]
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot = dot_f32_to_f64(a, b);
    let na = norm_sq_f32_to_f64(a).sqrt();
    let nb = norm_sq_f32_to_f64(b).sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    let cos = dot / (na * nb);
    cos.clamp(-1.0, 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    #[test]
    fn dot_simple() {
        // (1,2,3) . (4,5,6) = 4 + 10 + 18 = 32
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        assert!((dot_f32_to_f64(&a, &b) - 32.0).abs() < 1e-9);
    }

    #[test]
    fn dot_chunk_boundary() {
        // 17 elements -- exercises one full chunk-of-8 + 9 tail elems.
        let a: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..17).map(|i| (17 - i) as f32).collect();
        // expected = sum_{i=0..17} i * (17-i)
        let expected: f64 = (0..17).map(|i| (i * (17 - i)) as f64).sum();
        assert!((dot_f32_to_f64(&a, &b) - expected).abs() < 1e-6);
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
        // Repeated identical f32 can produce naive cosine slightly
        // above 1.0; the clamp keeps us in range.
        let v = vec![0.5773_f32; 256];
        let c = cosine_f32(&v, &v);
        assert!((-1.0..=1.0).contains(&c));
        assert!((c - 1.0).abs() < TOL);
    }

    #[test]
    fn cosine_known_value() {
        // (1,0,0,0) . (1,1,0,0) / (1 * sqrt(2)) = 1/sqrt(2)
        let a = [1.0_f32, 0.0, 0.0, 0.0];
        let b = [1.0_f32, 1.0, 0.0, 0.0];
        let c = cosine_f32(&a, &b);
        assert!((c - std::f32::consts::FRAC_1_SQRT_2).abs() < TOL);
    }
}
