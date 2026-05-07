"""Python-side LineRecord benchmarks (pytest-benchmark).

Two purposes:

1. Establish FFI overhead — same workloads as the Rust criterion bench so
   we can compare bytes/sec at both layers.
2. Compare against a pure-Python baseline using ``str.splitlines()`` plus a
   manual feature loop so the speed-up is visible per release.

Run with:
    uv run pytest tests/bench_lines.py --benchmark-only
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaos_nlp_core.segmentation import extract_line_records

# ─── Fixture corpora ──────────────────────────────────────────────────────────


def _synthetic_ascii(target_bytes: int) -> str:
    pool = [
        "ARTICLE I. PURPOSES",
        "  1. The Borrower hereby agrees to repay all sums advanced.",
        "  (a) Notwithstanding the foregoing, no waiver shall be effective unless made in writing.",
        "    (i) The interest rate shall accrue daily from the date of disbursement.",
        "",
        "Section 5. Severability.",
        "If any provision is held invalid, the remainder shall remain in full force and effect.",
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


def _synthetic_unicode(target_bytes: int) -> str:
    pool = [
        "第一条 目的（こうもく）。",
        "  一、本契約の目的は債権の返済である。",
        "    （イ）貸付日より日々利息が発生する。",
        "",
        "Article II — Définitions et applications 😀",
        "Article III — Стороны соглашаются на следующих условиях.",
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


def _shakespeare() -> str | None:
    p = Path(__file__).parent / "fixtures" / "shakespeare.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


# ─── Pure-Python baseline ────────────────────────────────────────────────────


def _python_baseline(text: str) -> list[dict]:
    """Reasonable splitlines + per-line feature loop in pure Python.

    Produces a list of dicts with the same shape (modulo offsets) as
    `extract_line_records`. Used only as a speed reference — it does not
    track byte/char offsets correctly with multi-byte input.
    """
    if not text:
        return []
    out: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        upper = sum(1 for c in stripped if c.isalpha() and c.isupper())
        lower = sum(1 for c in stripped if c.isalpha() and c.islower())
        out.append(
            {
                "char_len": len(line),
                "upper": upper,
                "lower": lower,
                "blank": not bool(stripped),
                "indent": len(line) - len(line.lstrip()),
            }
        )
    return out


# ─── Benchmarks ──────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="line_record/rust")
@pytest.mark.parametrize("size_kib", [16, 100, 1024])
def test_bench_rust_ascii(benchmark, size_kib: int) -> None:
    text = _synthetic_ascii(size_kib * 1024)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    benchmark.extra_info["lines_est"] = text.count("\n")
    result = benchmark(extract_line_records, text)
    assert result  # not empty


@pytest.mark.benchmark(group="line_record/rust")
@pytest.mark.parametrize("size_kib", [16, 100])
def test_bench_rust_unicode(benchmark, size_kib: int) -> None:
    text = _synthetic_unicode(size_kib * 1024)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    result = benchmark(extract_line_records, text)
    assert result


@pytest.mark.benchmark(group="line_record/rust")
def test_bench_rust_real_shakespeare(benchmark) -> None:
    text = _shakespeare()
    if text is None:
        pytest.skip("shakespeare.txt fixture missing")
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    result = benchmark(extract_line_records, text)
    assert result


@pytest.mark.benchmark(group="line_record/python")
@pytest.mark.parametrize("size_kib", [16, 100])
def test_bench_python_baseline_ascii(benchmark, size_kib: int) -> None:
    text = _synthetic_ascii(size_kib * 1024)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    result = benchmark(_python_baseline, text)
    assert result


@pytest.mark.benchmark(group="line_record/python")
def test_bench_python_baseline_real_shakespeare(benchmark) -> None:
    text = _shakespeare()
    if text is None:
        pytest.skip("shakespeare.txt fixture missing")
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    result = benchmark(_python_baseline, text)
    assert result
