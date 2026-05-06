"""Tests for :func:`kaos_nlp_core.segmentation.detect_boilerplate` at the
PyO3 boundary.

The detector takes a raw ``str`` (not a list of LineRecords) — the binding
extracts records internally — so these tests exercise the full Python-facing
contract.
"""

from __future__ import annotations

import pickle

import pytest

from kaos_nlp_core.segmentation import BoilerplateRun, detect_boilerplate


def _build_pages(header: str, footer: str, n_pages: int, *, body: str = "Body") -> str:
    """Form-feed-separated synthetic document with stable header/footer."""
    parts: list[str] = []
    for i in range(n_pages):
        parts.append(header + "\n")
        parts.append(body + "\n")
        parts.append(f"{body} unique {i}\n")  # vary so body is not boilerplate
        parts.append(footer + "\n")
        if i + 1 < n_pages:
            parts.append("\f")
    return "".join(parts)


# ─── Basic detection ────────────────────────────────────────────────────────


def test_empty_input_returns_empty_list() -> None:
    assert detect_boilerplate("") == []


def test_detects_repeated_header_with_form_feed() -> None:
    src = _build_pages("FILED 5/5/2026 SMITH V JONES", "End of page", 5)
    runs = detect_boilerplate(src)
    headers = [r for r in runs if "filed" in r.canonical_text]
    assert headers, "expected a header run"
    assert headers[0].kind == "header"
    assert headers[0].occurrences == 5


def test_detects_repeated_footer() -> None:
    src = _build_pages("HEADER LINE", "Confidential — Internal Only", 4)
    runs = detect_boilerplate(src)
    footers = [r for r in runs if "confidential" in r.canonical_text]
    assert footers
    assert footers[0].kind == "footer"
    assert footers[0].occurrences == 4


def test_detects_monotonic_page_number_sequence() -> None:
    """Real page numbers — values 1, 2, 3, 4, 5 across the bottom of 5 pages —
    are detected as a single PageNumber sequence (P5.6)."""
    parts: list[str] = []
    for i in range(1, 6):
        parts.append("HEADER\n")
        parts.append("Body content one\n")
        parts.append(f"Body content two on page {i}\n")
        parts.append("Body content three\n")
        parts.append(f"{i}\n")
        if i < 5:
            parts.append("\f")
    src = "".join(parts)
    runs = detect_boilerplate(src)
    pn = [r for r in runs if r.kind == "page_number"]
    assert pn, "expected a page-number sequence run"
    assert pn[0].occurrences == 5
    # Canonical text is a synthesized label, not any one occurrence's text.
    assert "page numbers" in pn[0].canonical_text


def test_constant_digit_footer_is_not_page_number() -> None:
    """P5.6: a footer that is the literal '1' on every page is NOT a
    page-number sequence (no monotonicity). It surfaces as a Footer instead."""
    src = _build_pages("HEADER", "1", 5)
    runs = detect_boilerplate(src)
    # No PageNumber should be emitted.
    assert not [r for r in runs if r.kind == "page_number"]
    # The constant '1' cluster does still pass the cluster threshold and is
    # classified as a Footer (it is in the bottom zone).
    footer = [r for r in runs if r.canonical_text == "1"]
    assert footer
    assert footer[0].kind == "footer"


# ─── Threshold gates ────────────────────────────────────────────────────────


def test_min_occurrences_filters_small_clusters() -> None:
    src = _build_pages("HEADER", "FOOTER", 2)  # 2 occurrences, below default 3
    runs = detect_boilerplate(src)
    assert not [r for r in runs if r.canonical_text == "header"]


def test_min_occurrences_kwarg_lowers_threshold() -> None:
    src = _build_pages("HEADER", "FOOTER", 2)
    runs = detect_boilerplate(src, min_occurrences=2, min_rate=0.0)
    headers = [r for r in runs if r.canonical_text == "header"]
    assert headers, "lowering min_occurrences should detect 2-page clusters"


# ─── OCR drift via MinHash ─────────────────────────────────────────────────


def test_near_dup_groups_ocr_drift() -> None:
    """One OCR-corrupted character per occurrence on a realistic 35-char header.

    Per the design reference, threshold 0.75 is one-typo-tolerant on 30-char
    strings (Jaccard floor ≈ 0.85). Three near-duplicate occurrences should
    cluster into a single run.
    """
    page = "FILED IN COURT 5/5/2026 SMITH V JONES\nbody one\nbody two\nfooter\n"
    page_typo1 = "FlLED IN COURT 5/5/2026 SMITH V JONES\nbody one\nbody two\nfooter\n"
    page_typo2 = "F1LED IN COURT 5/5/2026 SMITH V JONES\nbody one\nbody two\nfooter\n"
    src = page + "\f" + page_typo1 + "\f" + page_typo2
    runs = detect_boilerplate(src)
    near_dup = [r for r in runs if "court" in r.canonical_text]
    assert near_dup
    assert near_dup[0].occurrences == 3


def test_skip_near_dup_disables_ocr_clustering() -> None:
    page = "FILED IN COURT 5/5/2026 SMITH V JONES\nbody\n"
    page_typo1 = "FlLED IN COURT 5/5/2026 SMITH V JONES\nbody\n"
    page_typo2 = "F1LED IN COURT 5/5/2026 SMITH V JONES\nbody\n"
    src = page + "\f" + page_typo1 + "\f" + page_typo2
    runs = detect_boilerplate(src, skip_near_dup=True)
    assert not [r for r in runs if "court" in r.canonical_text]


# ─── Determinism + invariants ──────────────────────────────────────────────


def test_byte_identical_repeat_runs() -> None:
    src = _build_pages("HEADER", "FOOTER", 5)
    r1 = detect_boilerplate(src)
    r2 = detect_boilerplate(src)
    # Compare canonical text + occurrences; pyo3 wrappers don't implement __eq__.
    assert [(r.canonical_text, r.occurrences, r.kind) for r in r1] == [
        (r.canonical_text, r.occurrences, r.kind) for r in r2
    ]


def test_fingerprints_are_unique_per_run() -> None:
    src = _build_pages("HEADER A", "FOOTER B", 5)
    runs = detect_boilerplate(src)
    fps = [r.fingerprint for r in runs]
    assert len(fps) == len(set(fps))


def test_line_indices_within_record_count() -> None:
    src = _build_pages("HEADER", "FOOTER", 4)
    runs = detect_boilerplate(src)
    # Every line_indices entry must be a non-negative u32; we cannot verify
    # the upper bound from Python (no records list exposed) but the binding
    # validates internally and detect_boilerplate panicking would surface.
    for r in runs:
        assert all(idx >= 0 for idx in r.line_indices)
        assert r.occurrences == len(r.line_indices)


# ─── Windowed fall-back when \f is absent ──────────────────────────────────


def test_windowed_bucketing_when_no_form_feed() -> None:
    # 4 pages × 50 lines, NO form feed — header sits at line 0 of each window.
    parts = []
    for page in range(4):
        for line in range(50):
            parts.append("WINDOWED HEADER" if line == 0 else f"body {page}-{line}")
            parts.append("\n")
    src = "".join(parts)
    runs = detect_boilerplate(src)
    headers = [r for r in runs if r.canonical_text == "windowed header"]
    assert headers
    assert headers[0].kind == "header"
    assert headers[0].occurrences == 4


# ─── Pickle round-trip ─────────────────────────────────────────────────────


def test_run_pickle_round_trip() -> None:
    src = _build_pages("HEADER", "FOOTER", 5)
    runs = detect_boilerplate(src)
    for r in runs:
        copy = pickle.loads(pickle.dumps(r))
        assert copy.canonical_text == r.canonical_text
        assert copy.occurrences == r.occurrences
        assert copy.kind == r.kind


def test_run_repr_is_informative() -> None:
    src = _build_pages("HEADER", "FOOTER", 5)
    runs = detect_boilerplate(src)
    assert all("BoilerplateRun" in repr(r) for r in runs)


# ─── Type re-export ────────────────────────────────────────────────────────


def test_boilerplate_run_class_is_re_exported() -> None:
    src = _build_pages("HEADER", "FOOTER", 5)
    runs = detect_boilerplate(src)
    assert all(isinstance(r, BoilerplateRun) for r in runs)


# ─── Smoke: aggressive options should not panic ─────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lines_per_page": 25},
        {"header_zone_lines": 1, "footer_zone_lines": 1},
        {"min_occurrences": 2, "min_rate": 0.0},
        {"near_dup_threshold": 0.6, "num_perm": 32, "shingle_size": 3},
        {"zone_dominance": 0.99},
        {"drop_empty": False},
    ],
)
def test_aggressive_kwargs_smoke(kwargs: dict) -> None:
    src = _build_pages("HEADER", "FOOTER", 5)
    detect_boilerplate(src, **kwargs)  # must not raise


# ─── Multi-language caption hints (P7.0f) ──────────────────────────────────


def _build_pages_with_caption(caption: str, n_pages: int) -> str:
    parts: list[str] = []
    for i in range(n_pages):
        parts.append("HEADER\n")
        parts.append("Body line one\n")
        parts.append("Body line two\n")
        parts.append(caption + "\n")
        parts.append("Body line four\n")
        parts.append(f"Body unique {i + 1}\n")
        parts.append("FOOTER\n")
        if i + 1 < n_pages:
            parts.append("\f")
    return "".join(parts)


@pytest.mark.parametrize(
    "caption,language",
    [
        ("Figure 5: Architecture diagram", "english"),
        ("Abbildung 3: Diagramm der Komponenten", "german"),
        ("Tableau 2 — Résumé des résultats", "french"),
        ("Tabla 4: Comparación de modelos", "spanish"),
        ("Tabella 1: Riepilogo dei dati", "italian"),
        ("Quadro 2: Resumo das medidas", "portuguese"),
    ],
)
def test_caption_language_hint_per_lexicon(caption: str, language: str) -> None:
    """Each Western-language caption prefix must surface as a Caption run
    with the matching ``language_hint``. P7.0f generality contract."""
    src = _build_pages_with_caption(caption, 4)
    runs = detect_boilerplate(src)
    captions = [r for r in runs if r.kind == "caption"]
    assert captions, f"expected Caption run for {caption!r}"
    assert captions[0].language_hint == language


def test_non_caption_runs_have_no_language_hint() -> None:
    """Header/Footer/PageNumber runs must not carry a language_hint."""
    src = _build_pages("FILED IN COURT", "End of page", 4)
    runs = detect_boilerplate(src)
    for r in runs:
        if r.kind != "caption":
            assert r.language_hint is None
