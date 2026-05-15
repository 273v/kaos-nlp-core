"""Throughput benchmarks for :mod:`kaos_nlp_core.similarity` vs numpy.

Run with ``uv run pytest tests/bench_similarity.py --no-cov --benchmark-only``.

Workloads target the dimensionality + corpus sizes the package
actually serves in production:

- ``dim=256``  — model2vec / potion-base-8M (kaos-nlp-transformers default static embedder).
- ``dim=384``  — BAAI/bge-small-en-v1.5 (kaos-nlp-transformers default ONNX embedder).
- ``dim=768``  — bge-base, gte-base.
- ``dim=1536`` — OpenAI text-embedding-3-small (kaos-llm-core SemanticCache).

Corpus sizes: ``100``, ``1_000``, ``10_000``. The 100K case is too
slow to run on every CI but is interesting for retrieval scale.

The benchmark also writes a JSON report to ``docs/benchmarks/`` with
the median Rust-vs-numpy ratio per shape so we can track perf drift
over releases.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from kaos_nlp_core.similarity import (
    cosine_adjacent,
    cosine_adjacent_normalized,
    cosine_one_to_many,
    cosine_one_to_many_normalized,
    mmr_select,
    top_k_cosine,
)

pytestmark = pytest.mark.slow

_BENCH_DIR = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"

DIMS = (256, 384, 768, 1536)
CORPUS_SIZES = (100, 1_000, 10_000)


def _make_corpus(rng: np.random.Generator, n_rows: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(matrix, query)`` — random L2-normalized f32 corpus + query."""
    M = rng.standard_normal((n_rows, dim)).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-12)
    q = rng.standard_normal(dim).astype(np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return M, q


def _median_ns(fn, *args, n: int = 30, warmup: int = 3) -> float:
    """Median elapsed time in nanoseconds across `n` iterations."""
    for _ in range(warmup):
        fn(*args)
    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn(*args)
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def _emit_report(name: str, payload: dict) -> None:
    if os.environ.get("KAOS_NLP_SCALE_NO_REPORT"):
        return
    try:
        _BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (_BENCH_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# cosine_one_to_many: Rust+NumKong vs numpy matmul on normalized rows
# ---------------------------------------------------------------------------


def test_bench_cosine_one_to_many_grid() -> None:
    """Cosine query→corpus across the full dim/n_rows grid.

    Records Rust-and-numpy medians + the speedup ratio to a JSON
    benchmark file under ``docs/benchmarks/``. The **only assertion**
    is correctness — `|rust - numpy| < 1e-5` on every cell.

    Post-v0.1.0a6 (hand-rolled ``std::arch`` SIMD with runtime ISA
    dispatch — AVX2+FMA / AVX-512F / NEON / scalar):

    - The generic path is competitive with numpy's BLAS sgemv across
      the mid-corpus shapes (``n_rows ∈ [100, 1000]``) that
      kaos-nlp-transformers' SemanticChunker and ExtractiveRanker
      actually serve in production.
    - The pre-normalised fast path (`*_normalized`) is the right
      route for SemanticChunker/ExtractiveRanker — they always feed
      unit-norm embeddings. See
      ``test_bench_cosine_one_to_many_normalized_grid`` for that
      envelope.
    - Numpy still wins at the largest shapes (``n_rows >= 10000``)
      because numpy dispatches sgemv to multi-threaded BLAS. We
      single-thread on purpose — multi-threading is the caller's
      decision, and the Rust core releases the GIL so the caller
      can ``ThreadPoolExecutor`` over it.

    The JSON artifact is the source of truth for tracking drift
    across releases. We do NOT assert speedup >= 1x because that
    fails on the largest cells; this bench documents the perf
    envelope rather than gating on it.
    """
    rng = np.random.default_rng(2026_05_14)
    grid: list[dict] = []
    for n_rows in CORPUS_SIZES:
        for dim in DIMS:
            M, q = _make_corpus(rng, n_rows, dim)

            # Default-arg binding (`M=M`, `q=q`) snapshots the loop
            # variables; ruff's B023 (closure-captures-loop-variable)
            # would fire otherwise, even though the bench timer
            # consumes each closure within the same iteration.
            def _rust(M=M, q=q):
                return cosine_one_to_many(q, M)

            def _numpy(M=M, q=q):
                return M @ q

            # Correctness check — the only hard assertion.
            assert np.allclose(_rust(), _numpy(), atol=1e-5), (
                f"Rust/numpy diverge at n={n_rows} dim={dim}"
            )

            rust_ns = _median_ns(_rust, n=20)
            np_ns = _median_ns(_numpy, n=20)
            speedup = np_ns / rust_ns
            grid.append(
                {
                    "n_rows": n_rows,
                    "dim": dim,
                    "rust_median_ns": int(rust_ns),
                    "numpy_median_ns": int(np_ns),
                    "speedup_x": round(speedup, 3),
                    "rust_ms_per_call": round(rust_ns / 1_000_000, 4),
                    "numpy_ms_per_call": round(np_ns / 1_000_000, 4),
                }
            )

    _emit_report("similarity-cosine-one-to-many", {"grid": grid})


# ---------------------------------------------------------------------------
# top_k_cosine: Rust vs numpy argpartition+argsort
# ---------------------------------------------------------------------------


def test_bench_top_k_cosine_grid() -> None:
    """Full top-k pipeline benchmark (cosine sweep + heap selection)."""
    rng = np.random.default_rng(2026_05_14)
    grid: list[dict] = []
    for n_rows in CORPUS_SIZES:
        for dim in (256, 384, 768):
            M, q = _make_corpus(rng, n_rows, dim)
            k = 10

            def _rust(M=M, q=q, k=k):
                return top_k_cosine(q, M, k)

            def _numpy(M=M, q=q, k=k):
                sims = M @ q
                top_idx = np.argpartition(-sims, k)[:k]
                top_idx = top_idx[np.argsort(-sims[top_idx], kind="stable")]
                return top_idx, sims[top_idx]

            rust_ns = _median_ns(_rust, n=20)
            np_ns = _median_ns(_numpy, n=20)
            grid.append(
                {
                    "n_rows": n_rows,
                    "dim": dim,
                    "k": k,
                    "rust_median_ns": int(rust_ns),
                    "numpy_median_ns": int(np_ns),
                    "speedup_x": round(np_ns / rust_ns, 3),
                    "rust_ms_per_call": round(rust_ns / 1_000_000, 4),
                    "numpy_ms_per_call": round(np_ns / 1_000_000, 4),
                }
            )
    _emit_report("similarity-top-k-cosine", {"grid": grid, "k": 10})


# ---------------------------------------------------------------------------
# cosine_one_to_many_normalized: pre-normalised fast path vs numpy matmul
# ---------------------------------------------------------------------------


def test_bench_cosine_one_to_many_normalized_grid() -> None:
    """Pre-normalised fast path vs numpy matmul, across the
    production dim/n_rows grid.

    This is the path SemanticChunker and ExtractiveRanker should take
    (their EmbeddingModel always returns unit-norm rows). The Rust
    kernel skips both norm computations and the final ``sqrt``,
    leaving a pure SIMD dot product clamped to ``[-1, 1]``.

    Correctness assertion: ``|rust - clip(M @ q, -1, 1)| < 1e-5``.
    """
    rng = np.random.default_rng(2026_05_14)
    grid: list[dict] = []
    for n_rows in CORPUS_SIZES:
        for dim in DIMS:
            M, q = _make_corpus(rng, n_rows, dim)

            def _rust(M=M, q=q):
                return cosine_one_to_many_normalized(q, M)

            def _numpy(M=M, q=q):
                return np.clip(M @ q, -1.0, 1.0)

            assert np.allclose(_rust(), _numpy(), atol=1e-5), (
                f"Rust/numpy diverge at n={n_rows} dim={dim}"
            )

            rust_ns = _median_ns(_rust, n=20)
            np_ns = _median_ns(_numpy, n=20)
            speedup = np_ns / rust_ns
            grid.append(
                {
                    "n_rows": n_rows,
                    "dim": dim,
                    "rust_median_ns": int(rust_ns),
                    "numpy_median_ns": int(np_ns),
                    "speedup_x": round(speedup, 3),
                    "rust_ms_per_call": round(rust_ns / 1_000_000, 4),
                    "numpy_ms_per_call": round(np_ns / 1_000_000, 4),
                }
            )

    _emit_report("similarity-cosine-one-to-many-normalized", {"grid": grid})


# ---------------------------------------------------------------------------
# cosine_adjacent_normalized: pre-normalised fast path vs numpy einsum
# ---------------------------------------------------------------------------


def test_bench_cosine_adjacent_normalized_grid() -> None:
    """Pre-normalised adjacent-row cosine, the SemanticChunker fast path."""
    rng = np.random.default_rng(2026_05_14)
    grid: list[dict] = []
    for n_rows in (10, 50, 200, 1000):
        for dim in (256, 384, 768):
            M, _ = _make_corpus(rng, n_rows, dim)

            def _rust(M=M):
                return cosine_adjacent_normalized(M)

            def _numpy(M=M):
                return np.clip(np.einsum("ij,ij->i", M[:-1], M[1:]), -1.0, 1.0)

            assert np.allclose(_rust(), _numpy(), atol=1e-5)
            rust_ns = _median_ns(_rust, n=20)
            np_ns = _median_ns(_numpy, n=20)
            grid.append(
                {
                    "n_rows": n_rows,
                    "dim": dim,
                    "rust_median_ns": int(rust_ns),
                    "numpy_median_ns": int(np_ns),
                    "speedup_x": round(np_ns / rust_ns, 3),
                    "rust_us_per_call": round(rust_ns / 1_000, 3),
                    "numpy_us_per_call": round(np_ns / 1_000, 3),
                }
            )
    _emit_report("similarity-cosine-adjacent-normalized", {"grid": grid})


# ---------------------------------------------------------------------------
# cosine_adjacent: Rust vs numpy einsum on adjacent rows
# ---------------------------------------------------------------------------


def test_bench_cosine_adjacent_grid() -> None:
    rng = np.random.default_rng(2026_05_14)
    grid: list[dict] = []
    # Adjacent-pair pattern is per-document; n_rows here = paragraphs per doc.
    for n_rows in (10, 50, 200, 1000):
        for dim in (256, 384, 768):
            M, _ = _make_corpus(rng, n_rows, dim)

            def _rust(M=M):
                return cosine_adjacent(M)

            def _numpy(M=M):
                return np.einsum("ij,ij->i", M[:-1], M[1:])

            assert np.allclose(_rust(), _numpy(), atol=1e-5)
            rust_ns = _median_ns(_rust, n=20)
            np_ns = _median_ns(_numpy, n=20)
            grid.append(
                {
                    "n_rows": n_rows,
                    "dim": dim,
                    "rust_median_ns": int(rust_ns),
                    "numpy_median_ns": int(np_ns),
                    "speedup_x": round(np_ns / rust_ns, 3),
                    "rust_us_per_call": round(rust_ns / 1_000, 3),
                    "numpy_us_per_call": round(np_ns / 1_000, 3),
                }
            )
    _emit_report("similarity-cosine-adjacent", {"grid": grid})


# ---------------------------------------------------------------------------
# mmr_select: Rust vs numpy hand-rolled MMR
# ---------------------------------------------------------------------------


def _numpy_mmr(matrix: np.ndarray, relevance: np.ndarray, k: int, lambda_: float):
    """Reference MMR using numpy matmul for the inner cosine pass."""
    n = matrix.shape[0]
    cap = min(k, n)
    picked: list[int] = []
    max_sim = np.full(n, -np.inf, dtype=np.float32)
    available = np.ones(n, dtype=bool)
    # First pick: argmax(relevance)
    first = int(np.argmax(relevance))
    picked.append(first)
    available[first] = False
    max_sim = matrix @ matrix[first]
    while len(picked) < cap:
        scores = lambda_ * relevance - (1 - lambda_) * max_sim
        scores = np.where(available, scores, -np.inf)
        nxt = int(np.argmax(scores))
        picked.append(nxt)
        available[nxt] = False
        new_sims = matrix @ matrix[nxt]
        max_sim = np.maximum(max_sim, new_sims)
    return picked


def test_bench_mmr_select_grid() -> None:
    rng = np.random.default_rng(2026_05_14)
    grid: list[dict] = []
    # MMR is per-query reranker; corpus sizes here = candidate pool.
    for n_rows in (100, 1_000):
        for dim in (256, 384, 768):
            M, _ = _make_corpus(rng, n_rows, dim)
            rel = rng.standard_normal(n_rows).astype(np.float32)
            k = 20

            def _rust(M=M, rel=rel, k=k):
                return mmr_select(M, rel, k, 0.5)

            def _numpy(M=M, rel=rel, k=k):
                return _numpy_mmr(M, rel, k, 0.5)

            rust_ns = _median_ns(_rust, n=10)
            np_ns = _median_ns(_numpy, n=10)
            grid.append(
                {
                    "n_rows": n_rows,
                    "dim": dim,
                    "k": k,
                    "rust_median_ns": int(rust_ns),
                    "numpy_median_ns": int(np_ns),
                    "speedup_x": round(np_ns / rust_ns, 3),
                    "rust_ms_per_call": round(rust_ns / 1_000_000, 4),
                    "numpy_ms_per_call": round(np_ns / 1_000_000, 4),
                }
            )
    _emit_report("similarity-mmr-select", {"grid": grid, "k": 20})
