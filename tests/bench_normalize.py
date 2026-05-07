"""Python-side normalizer benchmarks (pytest-benchmark).

Two purposes:

1. Establish FFI overhead — same workloads as the criterion bench so the
   binding cost is visible per release.
2. Compare the binding to a pure-Python baseline using ``str.translate`` +
   ``str.lower`` + manual whitespace collapse. The baseline is intentionally
   simple so the speed-up is meaningful.

Run with:
    uv run pytest tests/bench_normalize.py --benchmark-only
"""

from __future__ import annotations

import re

import pytest

from kaos_nlp_core.segmentation import normalize

# ─── Fixture corpora ────────────────────────────────────────────────────────


def _synthetic_ascii(target_bytes: int) -> str:
    pool = [
        "ARTICLE I. PURPOSES",
        "  1. The Borrower hereby agrees to repay all sums advanced.",
        "  (a) Notwithstanding the foregoing, no waiver shall be effective.",
        "    (i) The interest rate shall accrue daily from disbursement.",
        "Section 5. Severability.",
        "If any provision is held invalid, the remainder shall remain.",
    ]
    parts: list[str] = []
    total = 0
    while total < target_bytes:
        for line in pool:
            parts.append(line)
            parts.append("\n")
            total += len(line.encode("utf-8")) + 1
            if total >= target_bytes:
                break
    return "".join(parts)


def _synthetic_unicode_legal(target_bytes: int) -> str:
    pool = [
        "“The Parties” agree as follows:",
        "  • Item 1—the obligor shall…",
        "  • Item 2—the obligee shall…",
        "Section 5. Severability—see Annex A.",
        "Whereas, the foregoing recital is incorporated…",
    ]
    parts: list[str] = []
    total = 0
    while total < target_bytes:
        for line in pool:
            parts.append(line)
            parts.append("\n")
            total += len(line.encode("utf-8")) + 1
            if total >= target_bytes:
                break
    return "".join(parts)


# ─── Pure-Python baseline (for "aggressive" config) ─────────────────────────


_PUNCT_TABLE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "‹": "'",
        "›": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
        "«": '"',
        "»": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "…": "...",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        "　": " ",
        "­": "",
        "​": "",
        "‌": "",
        "‍": "",
        "﻿": "",
        "•": "",
        "●": "",
        "◦": "",
    }
)
_WS_RUN = re.compile(r"\s+")


def _python_aggressive(text: str) -> str:
    """Pure-Python implementation of NormalizeOptions::aggressive()."""
    # Unicode-punct map → ASCII.
    out = text.translate(_PUNCT_TABLE)
    # Lower-case (ASCII-only is hard in Python; .lower() touches Unicode too,
    # so this baseline is *more* aggressive than ours; we keep it for
    # speed-comparison purposes only).
    out = out.lower()
    # Whitespace collapse.
    out = _WS_RUN.sub(" ", out)
    return out


# ─── Benchmarks ─────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="normalize/rust")
@pytest.mark.parametrize("size_kib", [16, 100, 1024])
def test_bench_rust_ascii_fast_path(benchmark, size_kib: int) -> None:
    text = _synthetic_ascii(size_kib * 1024)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    result = benchmark(normalize, text, normalize_unicode_punct=True)
    assert result.text == text  # fast path returns input unchanged


@pytest.mark.benchmark(group="normalize/rust")
@pytest.mark.parametrize("size_kib", [16, 100])
def test_bench_rust_ascii_aggressive(benchmark, size_kib: int) -> None:
    text = _synthetic_ascii(size_kib * 1024)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    result = benchmark(
        normalize,
        text,
        collapse_whitespace=True,
        fold_case=True,
        normalize_unicode_punct=True,
    )
    assert result.text  # not empty


@pytest.mark.benchmark(group="normalize/rust")
@pytest.mark.parametrize("size_kib", [16, 100])
def test_bench_rust_unicode_legal_aggressive(benchmark, size_kib: int) -> None:
    text = _synthetic_unicode_legal(size_kib * 1024)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    result = benchmark(
        normalize,
        text,
        collapse_whitespace=True,
        fold_case=True,
        normalize_unicode_punct=True,
    )
    assert result.text


@pytest.mark.benchmark(group="normalize/python")
@pytest.mark.parametrize("size_kib", [16, 100])
def test_bench_python_ascii_aggressive(benchmark, size_kib: int) -> None:
    text = _synthetic_ascii(size_kib * 1024)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    result = benchmark(_python_aggressive, text)
    assert result


@pytest.mark.benchmark(group="normalize/python")
@pytest.mark.parametrize("size_kib", [16, 100])
def test_bench_python_unicode_aggressive(benchmark, size_kib: int) -> None:
    text = _synthetic_unicode_legal(size_kib * 1024)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    result = benchmark(_python_aggressive, text)
    assert result
