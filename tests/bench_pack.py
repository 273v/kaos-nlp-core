"""Throughput benchmarks for :mod:`kaos_nlp_core.chunking._pack`.

Compares the Rust packer (``kaos_nlp_core._rust.chunking.pack_units``,
wired through the production
:func:`kaos_nlp_core.chunking._pack.pack_units` wrapper) against the
pre-Rust pure-Python greedy loop. The Python baseline is reproduced
inline below so we can diff the two implementations on identical
inputs after the production code has migrated to Rust.

Run with::

    uv run --no-sync pytest tests/bench_pack.py --no-cov --benchmark-only

The benchmark writes a JSON report to ``docs/benchmarks/`` capturing
the median Rust-vs-Python speedup per workload shape so we can track
perf drift over releases.

Workload shapes target the unit counts a long-document chunker
actually faces:

* ``n_units=100``    — a typical contract or filing section.
* ``n_units=1_000``  — a small statute title (USC) or full 10-K.
* ``n_units=10_000`` — multi-document corpus pack (long-form ingest).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest

from kaos_nlp_core.chunking._pack import _Unit, default_token_counter, pack_units


def _python_pack_units(
    units: list[_Unit],
    *,
    source: str,
    parent_id: str | None,
    max_tokens: int,
    overlap_units: int = 0,
    token_counter=default_token_counter,
    depth: int = 0,
    chunk_metadata: dict[str, object] | None = None,
):
    """Pre-Rust pure-Python pack loop, frozen for benchmark comparison.

    Mirrors the algorithm that lived in
    :mod:`kaos_nlp_core.chunking._pack` before the Rust kernel landed.
    Kept here (and ONLY here) so we can compare throughput on identical
    inputs; production code must call :func:`pack_units` from the
    Rust-backed wrapper, never this fallback.
    """
    from kaos_nlp_core.chunking import Chunk

    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
    if overlap_units < 0:
        raise ValueError(f"overlap_units must be >= 0, got {overlap_units}")

    filtered = [u for u in units if u.end > u.start]
    if not filtered:
        return []

    chunks: list[Chunk] = []
    current: list[_Unit] = []
    current_tokens = 0
    base_metadata = chunk_metadata or {}

    def _flush():
        nonlocal current, current_tokens
        if not current:
            return
        first = current[0]
        last = current[-1]
        chunk_text = source[first.start : last.end]
        merged_metadata: dict[str, object] = dict(base_metadata)
        merged_metadata["units"] = len(current)
        for unit in current:
            for key, value in unit.metadata.items():
                merged_metadata.setdefault(key, value)
        chunks.append(
            Chunk(
                text=chunk_text,
                start=first.start,
                end=last.end,
                parent_id=parent_id,
                token_count=token_counter(chunk_text),
                depth=depth,
                metadata=merged_metadata,
            )
        )
        if overlap_units > 0:
            tail = current[-overlap_units:]
            current = list(tail)
            current_tokens = sum(u.token_count for u in current)
        else:
            current = []
            current_tokens = 0

    for unit in filtered:
        if current and current_tokens + unit.token_count > max_tokens:
            _flush()
        current.append(unit)
        current_tokens += unit.token_count
        if len(current) == 1 and current_tokens > max_tokens:
            _flush()

    _flush()
    return chunks


def _build_corpus(n_units: int, unit_chars: int = 80, seed: int = 0):
    """Build a synthetic corpus + matching ``_Unit`` records.

    Each unit is ``unit_chars`` long. Token counts are
    ``ceil(unit_chars / 4)`` to mirror the default counter, with mild
    jitter so the packer occasionally has to flush early.
    """
    rng = np.random.default_rng(seed)
    source_parts = []
    units: list[_Unit] = []
    pos = 0
    for _ in range(n_units):
        # 10-char jitter so every chunk does real work
        length = unit_chars + int(rng.integers(-5, 6))
        length = max(8, length)
        text = "a" * length + " "
        start = pos
        end = pos + len(text)
        token_count = -(-length // 4)
        source_parts.append(text)
        units.append(
            _Unit(
                text=text,
                start=start,
                end=end,
                token_count=token_count,
            )
        )
        pos = end
    return "".join(source_parts), units


def _equal_chunks(a, b) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if x.start != y.start or x.end != y.end:
            return False
        if x.text != y.text:
            return False
        if x.token_count != y.token_count:
            return False
    return True


SHAPES = [
    (100, 512, 0),
    (1_000, 512, 0),
    (10_000, 512, 0),
    (1_000, 512, 2),
    (10_000, 2048, 4),
]


def _bench_one(fn, *args, warmup: int = 1, iters: int = 5) -> float:
    """Return median wall-clock seconds across ``iters`` calls."""
    for _ in range(warmup):
        fn(*args)
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


@pytest.mark.parametrize(("n_units", "max_tokens", "overlap_units"), SHAPES)
def test_pack_speedup(n_units: int, max_tokens: int, overlap_units: int, request) -> None:
    source, units = _build_corpus(n_units)

    def _rust_call():
        return pack_units(
            units,
            source=source,
            parent_id=None,
            max_tokens=max_tokens,
            overlap_units=overlap_units,
        )

    def _py_call():
        return _python_pack_units(
            units,
            source=source,
            parent_id=None,
            max_tokens=max_tokens,
            overlap_units=overlap_units,
        )

    # Sanity: outputs must match exactly.
    assert _equal_chunks(_rust_call(), _py_call())

    rust_t = _bench_one(_rust_call)
    py_t = _bench_one(_py_call)
    speedup = py_t / rust_t if rust_t > 0 else float("inf")

    report_path = (
        Path(__file__).parents[1] / "docs" / "benchmarks" / "chunker-pack-rust-vs-python.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "n_units": n_units,
        "max_tokens": max_tokens,
        "overlap_units": overlap_units,
        "rust_seconds": rust_t,
        "python_seconds": py_t,
        "speedup": speedup,
    }
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []
    if not isinstance(existing, list):
        existing = []
    existing = [
        r
        for r in existing
        if not (
            r.get("n_units") == n_units
            and r.get("max_tokens") == max_tokens
            and r.get("overlap_units") == overlap_units
        )
    ]
    existing.append(record)
    report_path.write_text(json.dumps(existing, indent=2, sort_keys=True))

    if os.environ.get("KNC_BENCH_PRINT"):
        print(
            f"\nn={n_units:>6} max={max_tokens:>4} ov={overlap_units} "
            f"rust={rust_t * 1e3:8.3f}ms python={py_t * 1e3:8.3f}ms speedup={speedup:5.2f}x"
        )

    # Asserting only correctness (above), NOT minimum speedup, because
    # the boundary cost dominates at very small workloads.
