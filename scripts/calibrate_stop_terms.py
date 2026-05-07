"""Calibrate ``DEFAULT_STOP_HYPERNYMS`` / ``DEFAULT_STOP_HYPONYMS``.

Iterates the bundled fixtures (USC, EDGAR agreements, Project Gutenberg)
and computes the document-frequency of every hypernym and hyponym that
``extract_concepts`` would surface with no stop-list. Terms that appear
in nearly every document are the abstract roots ("determiner",
"preposition", "thing") that drown the signal.

The script emits Python ``frozenset[str]`` literals ready to paste into
``python/kaos_nlp_core/concepts/__init__.py``.

Usage::

    uv run python scripts/calibrate_stop_terms.py
    uv run python scripts/calibrate_stop_terms.py --threshold 0.6 --sample 200
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from kaos_nlp_core.concepts import extract_concepts
from kaos_nlp_core.lexicon import default_opengloss_lexicon

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"

CORPORA: tuple[Path, ...] = (
    FIXTURES / "usc.jsonl",
    FIXTURES / "edgar_agreements.jsonl",
    FIXTURES / "shakespeare.txt",
    FIXTURES / "war_and_peace.txt",
)


def iter_documents(path: Path) -> Iterable[str]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or ""
                if text:
                    yield text
    elif path.suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
        # For Project Gutenberg files (~5-10 MB), chunk into ~5k-char windows
        # so we get more "documents" for the document-frequency statistic.
        # Without chunking, df=1.0 for any term appearing once in 5 MB.
        chunk_size = 5_000
        if len(text) <= chunk_size:
            yield text
        else:
            for start in range(0, len(text), chunk_size):
                chunk = text[start : start + chunk_size]
                if len(chunk.strip()) >= 200:
                    yield chunk
    else:
        raise ValueError(f"Unsupported fixture format: {path}")


def collect_doc_freqs(
    paths: Iterable[Path],
    *,
    direction: str,
    sample_per_corpus: int,
    top_k_per_doc: int,
) -> tuple[Counter[str], int]:
    """Return ``(doc_frequency_counter, total_docs)`` for the given direction."""
    df: Counter[str] = Counter()
    total_docs = 0
    for path in paths:
        if not path.exists():
            print(f"  warn: missing fixture {path}, skipping")
            continue
        n_local = 0
        for text in iter_documents(path):
            if not text or len(text) < 200:
                continue
            try:
                # Big top_k to capture the long tail; only the high-df ones
                # become stop terms.
                concepts = extract_concepts(
                    text,
                    direction=direction,
                    top_k=top_k_per_doc,
                    extra_stop_terms=set(),  # no stop-list at calibration time
                )
            except Exception as exc:
                print(f"  warn: extract failed on {path.name}#{n_local}: {exc}")
                continue
            seen_in_doc: set[str] = set()
            for c in concepts:
                if c.term not in seen_in_doc:
                    df[c.term] += 1
                    seen_in_doc.add(c.term)
            total_docs += 1
            n_local += 1
            if sample_per_corpus and n_local >= sample_per_corpus:
                break
        print(f"  {path.name}: {n_local} documents")
    return df, total_docs


def render_frozenset(name: str, terms: list[tuple[str, float]]) -> str:
    """Render a stable, alphabetically-sorted ``frozenset[str]`` literal."""
    sorted_terms = sorted({t for t, _ in terms})
    if not sorted_terms:
        return f"{name}: frozenset[str] = frozenset()"
    lines = [f"{name}: frozenset[str] = frozenset({{"]
    for t in sorted_terms:
        lines.append(f"    {t!r},")
    lines.append("})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Document-frequency threshold; terms appearing in ≥ this fraction "
        "of documents become stop terms (default 0.50).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=300,
        help="Max documents per corpus (default 300). 0 = all.",
    )
    parser.add_argument(
        "--top-k-per-doc",
        type=int,
        default=50,
        help="Concepts to extract per doc before measuring df (default 50).",
    )
    parser.add_argument(
        "--direction",
        choices=("hypernym", "hyponym", "both"),
        default="both",
    )
    parser.add_argument("--show-top", type=int, default=30)
    args = parser.parse_args(argv)

    # Force lexicon load before iterating, so the per-doc loop is fast.
    print("Warming default OpenGloss lexicon ...")
    default_opengloss_lexicon()
    print("  lexicon loaded.")

    directions = ("hypernym", "hyponym") if args.direction == "both" else (args.direction,)

    for direction in directions:
        print(f"\nCalibrating stop-{direction}s (df ≥ {args.threshold:.2f}) ...")
        df, total = collect_doc_freqs(
            CORPORA,
            direction=direction,
            sample_per_corpus=args.sample,
            top_k_per_doc=args.top_k_per_doc,
        )
        print(f"  total documents scored: {total}")
        if total == 0:
            print("  (no docs — skipping)")
            continue
        threshold = args.threshold
        rated = [(term, count / total) for term, count in df.items()]
        rated.sort(key=lambda kv: -kv[1])
        stop = [(t, r) for t, r in rated if r >= threshold]
        print(f"\n  Top {args.show_top} most-frequent {direction}s:")
        for term, ratio in rated[: args.show_top]:
            marker = "★" if ratio >= threshold else " "
            print(f"    {marker} {ratio:.3f}  {term}")

        const_name = f"DEFAULT_STOP_{direction.upper()}S"
        print()
        print(f"  → {len(stop)} stop-{direction}s at df ≥ {threshold:.2f}")
        print()
        print(render_frozenset(const_name, stop))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
