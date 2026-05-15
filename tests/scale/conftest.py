"""Shared fixtures for scale tests over real corpora.

These tests exercise chunkers / aggregators against the HuggingFace
corpora downloaded by
``kaos-nlp-core/tests/fixtures/download_hf_fixtures.py``:

- ``usc.jsonl`` (68,759 sections of the US Code, ~50M words)
- ``edgar_agreements.jsonl`` (200 SEC EDGAR contracts, ~13K words each)
- ``patents.jsonl`` (200 US patents, ~13K words each)

The fixtures themselves are not committed (they're ~360 MB total and
covered by a separate license / provenance manifest). The conftest
resolves their location in this order:

1. ``KAOS_NLP_SCALE_FIXTURES`` env var (absolute directory path).
2. ``../kaos-modules/kaos-nlp-core/tests/fixtures/`` relative to the
   repo root (the canonical local layout when working out of the
   ``273v/`` workspace).
3. ``./tests/fixtures/`` if the user has run ``download_hf_fixtures.py``
   in-tree.

If none resolve, every test in this directory is skipped with a
single message — never with import errors.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# Tests in this directory are slow by default.
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixture-directory resolution
# ---------------------------------------------------------------------------


_FIXTURE_FILES = ("usc.jsonl", "edgar_agreements.jsonl", "patents.jsonl")


def _resolve_fixtures_dir() -> Path | None:
    """Return the directory containing the HF JSONL fixtures, or None."""
    env = os.environ.get("KAOS_NLP_SCALE_FIXTURES")
    if env:
        path = Path(env).expanduser().resolve()
        if all((path / name).exists() for name in _FIXTURE_FILES):
            return path

    # Walk up from this file to find the workspace root, then look in
    # the ``kaos-modules`` sibling.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "kaos-modules" / "kaos-nlp-core" / "tests" / "fixtures"
        if all((candidate / name).exists() for name in _FIXTURE_FILES):
            return candidate

    # Fall back to in-tree fixtures (download_hf_fixtures.py output).
    in_tree = here.parent.parent / "fixtures"
    if all((in_tree / name).exists() for name in _FIXTURE_FILES):
        return in_tree

    return None


@pytest.fixture(scope="session")
def scale_fixtures_dir() -> Path:
    path = _resolve_fixtures_dir()
    if path is None:
        pytest.skip(
            "Scale fixtures not available. "
            "Set KAOS_NLP_SCALE_FIXTURES or run "
            "tests/fixtures/download_hf_fixtures.py to populate "
            "usc.jsonl / edgar_agreements.jsonl / patents.jsonl."
        )
    return path


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path, *, max_records: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            if max_records is not None and index >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Streaming variant for the largest corpus."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _record_text(record: dict[str, Any]) -> str:
    """Extract the canonical text payload from a HF fixture record.

    The three corpora use different field names — USC has ``text``;
    patents have ``title``/``abstract``/``claims``/``text``; EDGAR
    has ``text``. We always return the ``text`` field and let the
    caller add extras if they need them.
    """
    text = record.get("text") or record.get("content") or record.get("body") or ""
    return str(text)


# ---------------------------------------------------------------------------
# Sample sizes (tunable via env vars for ad-hoc deeper runs)
# ---------------------------------------------------------------------------


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Defaults sized for a few-minute run on a laptop. Override via env
# for the nightly / release-gate run.
USC_SAMPLE_SIZE = _int_env("KAOS_NLP_SCALE_USC_SAMPLE", 1000)
EDGAR_SAMPLE_SIZE = _int_env("KAOS_NLP_SCALE_EDGAR_SAMPLE", 200)
PATENTS_SAMPLE_SIZE = _int_env("KAOS_NLP_SCALE_PATENTS_SAMPLE", 200)


# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def usc_sample(scale_fixtures_dir: Path) -> list[dict[str, Any]]:
    """USC sample of size :data:`USC_SAMPLE_SIZE`."""
    return _load_jsonl(scale_fixtures_dir / "usc.jsonl", max_records=USC_SAMPLE_SIZE)


@pytest.fixture(scope="session")
def edgar_agreements(scale_fixtures_dir: Path) -> list[dict[str, Any]]:
    """All EDGAR agreements (capped at :data:`EDGAR_SAMPLE_SIZE`)."""
    return _load_jsonl(
        scale_fixtures_dir / "edgar_agreements.jsonl",
        max_records=EDGAR_SAMPLE_SIZE,
    )


@pytest.fixture(scope="session")
def patents(scale_fixtures_dir: Path) -> list[dict[str, Any]]:
    """All patents (capped at :data:`PATENTS_SAMPLE_SIZE`)."""
    return _load_jsonl(
        scale_fixtures_dir / "patents.jsonl",
        max_records=PATENTS_SAMPLE_SIZE,
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


class Stopwatch:
    """Trivial wall-clock timer for in-test throughput reporting."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start


@pytest.fixture
def stopwatch() -> Stopwatch:
    return Stopwatch()


# Re-export helpers so test modules can ``from .conftest import _record_text``.
record_text = _record_text
iter_jsonl = _iter_jsonl
load_jsonl = _load_jsonl
