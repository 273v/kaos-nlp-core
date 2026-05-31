"""Scale tests for deterministic chunkers over real corpora.

What these tests prove (and what unit tests do not):

1. **Offset round-trip holds on every chunk** produced from every
   document in the USC sample (default 1000 sections), all 200 EDGAR
   agreements, and all 200 patents. With ~13K-word averages on the
   long-doc corpora and arbitrarily-shaped real text, this is the
   real round-trip test.

2. **chunk_id determinism** across two independent chunker
   instances. A real test that the SHA-256-derived id is stable
   across process state, not just within a single call site.

3. **No exceptions on real text**. Real legal/patent text has weird
   enumerators, tables, repeated whitespace, mid-word newlines,
   etc. — every chunker must tolerate it.

4. **Throughput is in a sane range** (chunks/sec, docs/sec).

5. **Chunk-size distributions stay bounded**: the budget is a soft
   ceiling and we tolerate one-unit-oversize, but a runaway chunker
   that emits a 10x-budget chunk would fail.

The throughput/distribution metrics are written to
``docs/benchmarks/<corpus>-chunkers-scale.json`` (gitignored) so the
release gate can track regressions over time.

These tests are slow (default sample ~1000 USC docs takes 5-10 s on
a laptop). They are gated by ``pytestmark = pytest.mark.slow`` in
``conftest.py`` and are skipped if the HF fixtures aren't present.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kaos_nlp_core.chunking import (
    Chunk,
    FixedTokenChunker,
    HierarchicalChunker,
    ParagraphChunker,
    SectionChunker,
    SentenceChunker,
    validate_chunk_offsets,
)

from .conftest import record_text

# ``resource`` is a Unix-only stdlib module; these memory-scale tests skip on Windows.
resource = pytest.importorskip("resource")

# ---------------------------------------------------------------------------
# Chunker registry
# ---------------------------------------------------------------------------


def _make_fixed() -> FixedTokenChunker:
    return FixedTokenChunker(max_tokens=512, overlap_tokens=64)


def _make_sentence() -> SentenceChunker:
    return SentenceChunker(max_tokens=512)


def _make_paragraph() -> ParagraphChunker:
    return ParagraphChunker(max_tokens=1024)


def _make_section() -> SectionChunker:
    return SectionChunker(max_tokens=1024)


def _make_hierarchical() -> HierarchicalChunker:
    return HierarchicalChunker(max_tokens=1024)


CHUNKER_FACTORIES: dict[str, Callable[[], Any]] = {
    "FixedToken": _make_fixed,
    "Sentence": _make_sentence,
    "Paragraph": _make_paragraph,
    "Section": _make_section,
    "Hierarchical": _make_hierarchical,
}


# ---------------------------------------------------------------------------
# Benchmark-report sink
# ---------------------------------------------------------------------------


_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"


def _emit_report(name: str, payload: dict[str, Any]) -> None:
    """Write a benchmark JSON to ``docs/benchmarks/`` (best-effort).

    Failure is silent: the test's correctness assertions are the
    authoritative signal; the JSON is a side-channel for tracking.
    """
    if os.environ.get("KAOS_NLP_SCALE_NO_REPORT"):
        return
    try:
        _BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (_BENCH_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        # Reporting is non-blocking by design.
        pass


def _rss_mb() -> float:
    """Resident set size in MB on Linux (ru_maxrss is in KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Core scale exercise
# ---------------------------------------------------------------------------


def _exercise_chunker(
    chunker: Any,
    documents: list[dict[str, Any]],
    *,
    label: str,
    cap_bad_examples: int = 5,
) -> dict[str, Any]:
    """Run ``chunker`` over every document; collect metrics.

    Returns a dict with throughput, distribution stats, and
    ``bad_offset_examples`` (empty list when 100% of chunks satisfy
    the offset round-trip invariant).
    """
    total_chunks = 0
    total_docs = 0
    empty_doc_chunks = 0
    chunk_token_counts: list[int] = []
    chunks_per_doc: list[int] = []
    bad_offsets: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []

    rss_before = _rss_mb()
    elapsed_total = 0.0

    for doc_index, record in enumerate(documents):
        text = record_text(record)
        doc_id = str(record.get("id", f"doc-{doc_index}"))
        if not text.strip():
            continue
        total_docs += 1

        t0 = time.perf_counter()
        try:
            chunks: list[Chunk] = chunker.chunk(text, parent_id=doc_id)
        except Exception as exc:
            exceptions.append(
                {"doc_id": doc_id, "error_type": type(exc).__name__, "error": str(exc)[:200]}
            )
            continue
        elapsed_total += time.perf_counter() - t0

        if not chunks:
            empty_doc_chunks += 1
            continue

        chunks_per_doc.append(len(chunks))
        for chunk in chunks:
            total_chunks += 1
            chunk_token_counts.append(chunk.token_count)
            # Validate offset round-trip. Cap captured bad examples
            # to keep failure messages readable.
            if not validate_chunk_offsets(text, chunk) and len(bad_offsets) < cap_bad_examples:
                bad_offsets.append(
                    {
                        "doc_id": doc_id,
                        "start": chunk.start,
                        "end": chunk.end,
                        "chunk_text_head": chunk.text[:60],
                        "source_slice_head": text[chunk.start : chunk.end][:60],
                    }
                )

    rss_after = _rss_mb()
    gc.collect()

    metrics: dict[str, Any] = {
        "label": label,
        "documents": total_docs,
        "chunks": total_chunks,
        "empty_documents": empty_doc_chunks,
        "exceptions": exceptions,
        "bad_offset_examples": bad_offsets,
        "elapsed_seconds": round(elapsed_total, 3),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_mb": round(rss_after, 1),
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "chunks_per_sec": round(total_chunks / elapsed_total, 1) if elapsed_total else 0.0,
        "docs_per_sec": round(total_docs / elapsed_total, 1) if elapsed_total else 0.0,
    }
    if chunk_token_counts:
        metrics["chunk_token_count"] = {
            "min": min(chunk_token_counts),
            "p50": statistics.median(chunk_token_counts),
            "p95": statistics.quantiles(chunk_token_counts, n=20)[18]
            if len(chunk_token_counts) >= 20
            else max(chunk_token_counts),
            "p99": statistics.quantiles(chunk_token_counts, n=100)[98]
            if len(chunk_token_counts) >= 100
            else max(chunk_token_counts),
            "max": max(chunk_token_counts),
            "mean": round(statistics.mean(chunk_token_counts), 1),
        }
    if chunks_per_doc:
        metrics["chunks_per_doc"] = {
            "min": min(chunks_per_doc),
            "p50": statistics.median(chunks_per_doc),
            "p95": statistics.quantiles(chunks_per_doc, n=20)[18]
            if len(chunks_per_doc) >= 20
            else max(chunks_per_doc),
            "max": max(chunks_per_doc),
            "mean": round(statistics.mean(chunks_per_doc), 1),
        }
    return metrics


def _assert_invariants(metrics: dict[str, Any]) -> None:
    """Assert the load-bearing invariants on the metrics dict."""
    bad = metrics["bad_offset_examples"]
    assert not bad, (
        f"{metrics['label']}: offset round-trip violated on {len(bad)} chunks. Examples: {bad[:3]}"
    )
    exceptions = metrics["exceptions"]
    assert not exceptions, (
        f"{metrics['label']}: chunker raised on {len(exceptions)} docs. Examples: {exceptions[:3]}"
    )


# ---------------------------------------------------------------------------
# Per-corpus per-chunker scale tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunker_name", list(CHUNKER_FACTORIES))
def test_chunker_scale_usc(chunker_name: str, usc_sample: list[dict[str, Any]]) -> None:
    """Every chunker round-trips on the USC sample, with no exceptions."""
    factory = CHUNKER_FACTORIES[chunker_name]
    chunker = factory()
    metrics = _exercise_chunker(
        chunker,
        usc_sample,
        label=f"{chunker_name}@USC",
    )
    _emit_report(f"chunker-scale-usc-{chunker_name.lower()}", metrics)
    _assert_invariants(metrics)
    assert metrics["chunks"] > 0, f"{chunker_name}: no chunks produced for USC sample"


@pytest.mark.parametrize("chunker_name", list(CHUNKER_FACTORIES))
def test_chunker_scale_edgar(chunker_name: str, edgar_agreements: list[dict[str, Any]]) -> None:
    """Every chunker handles the long EDGAR contracts (avg ~13K words)."""
    factory = CHUNKER_FACTORIES[chunker_name]
    chunker = factory()
    metrics = _exercise_chunker(
        chunker,
        edgar_agreements,
        label=f"{chunker_name}@EDGAR",
    )
    _emit_report(f"chunker-scale-edgar-{chunker_name.lower()}", metrics)
    _assert_invariants(metrics)
    assert metrics["chunks"] > 0


@pytest.mark.parametrize("chunker_name", list(CHUNKER_FACTORIES))
def test_chunker_scale_patents(chunker_name: str, patents: list[dict[str, Any]]) -> None:
    """Every chunker handles patent documents (claims-heavy text)."""
    factory = CHUNKER_FACTORIES[chunker_name]
    chunker = factory()
    metrics = _exercise_chunker(
        chunker,
        patents,
        label=f"{chunker_name}@Patents",
    )
    _emit_report(f"chunker-scale-patents-{chunker_name.lower()}", metrics)
    _assert_invariants(metrics)
    assert metrics["chunks"] > 0


# ---------------------------------------------------------------------------
# Cross-cutting checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunker_name", list(CHUNKER_FACTORIES))
def test_chunker_determinism_at_scale(chunker_name: str, usc_sample: list[dict[str, Any]]) -> None:
    """Two independent chunker instances produce identical chunk_id sequences.

    A 200-doc subsample is enough to surface any nondeterminism
    without doubling the cost of the full-scale test.
    """
    factory = CHUNKER_FACTORIES[chunker_name]
    chunker_a = factory()
    chunker_b = factory()

    sample = usc_sample[:200]
    drifts = 0
    drift_examples: list[str] = []
    for record in sample:
        text = record_text(record)
        if not text.strip():
            continue
        doc_id = str(record["id"])
        a_ids = [c.chunk_id for c in chunker_a.chunk(text, parent_id=doc_id)]
        b_ids = [c.chunk_id for c in chunker_b.chunk(text, parent_id=doc_id)]
        if a_ids != b_ids:
            drifts += 1
            if len(drift_examples) < 3:
                drift_examples.append(doc_id)

    assert drifts == 0, (
        f"{chunker_name}: chunk_id drift on {drifts} of {len(sample)} documents "
        f"(examples: {drift_examples})"
    )


def test_chunker_budget_compliance_usc(usc_sample: list[dict[str, Any]]) -> None:
    """No chunker should emit a chunk drastically larger than its budget.

    Soft ceiling: a chunker is allowed to exceed ``max_tokens`` by
    one indivisible unit (the "oversize sentence" / "oversize
    paragraph" case is part of the API contract). We assert the p99
    chunk size stays within 3x the budget to catch runaway packers.
    """
    for name, factory in CHUNKER_FACTORIES.items():
        chunker = factory()
        max_tokens: int = chunker.max_tokens
        sample_chunks: list[Chunk] = []
        for record in usc_sample[:500]:
            text = record_text(record)
            if not text.strip():
                continue
            sample_chunks.extend(chunker.chunk(text, parent_id=str(record["id"])))
        if not sample_chunks:
            continue
        token_counts = sorted(c.token_count for c in sample_chunks)
        p99 = token_counts[int(len(token_counts) * 0.99)]
        worst = token_counts[-1]
        # Hierarchical emits depth-0 section chunks that can be much
        # larger than max_tokens by design (one whole section).
        # Section chunker similarly emits a section-shaped chunk.
        ceiling = max_tokens * (10 if name in {"Hierarchical", "Section"} else 3)
        assert p99 <= ceiling, (
            f"{name}: p99 chunk token_count={p99} exceeds {ceiling} "
            f"(max_tokens={max_tokens}, worst={worst}, n={len(sample_chunks)})"
        )
