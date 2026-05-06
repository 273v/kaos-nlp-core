"""Build a kaos-nlp-core Lexicon binary from OpenGloss v1.3 on HuggingFace.

Downloads mjbommar/opengloss-v1.3-dictionary (205,988 entries), converts
each row to the Lexicon entry format, and saves as a bincode binary file
via Lexicon.save().

Usage::

    uv run python scripts/build_opengloss_lexicon.py
    uv run python scripts/build_opengloss_lexicon.py --output data/opengloss-v1.3.lexicon.bin
    uv run python scripts/build_opengloss_lexicon.py --verify

Output: ~30-80MB binary file loadable via Lexicon.load().
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

_DEFAULT_BATCH_SIZE = 5000
_LOG_PROGRESS_INTERVAL = 50_000
_VERIFY_DISPLAY_TERMS = 15


def build_lexicon(
    output_path: Path,
    *,
    dataset_id: str = "mjbommar/opengloss-v1.3-dictionary",
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> None:
    """Download an OpenGloss dictionary from HuggingFace and build a Lexicon binary."""
    from datasets import load_dataset

    from kaos_nlp_core.lexicon import Lexicon

    print(f"Downloading {dataset_id} from HuggingFace...")
    t0 = time.perf_counter()
    ds = load_dataset(dataset_id, split="train")
    t1 = time.perf_counter()
    print(f"  Downloaded {len(ds)} entries in {t1 - t0:.1f}s")

    print("Building Lexicon...")
    lex = Lexicon()
    batch: list[dict] = []
    total_added = 0

    for i, row in enumerate(ds):
        entry = _row_to_entry(row)
        batch.append(entry)

        if len(batch) >= batch_size:
            lex.add_entries(batch)
            total_added += len(batch)
            batch = []
            if (i + 1) % _LOG_PROGRESS_INTERVAL == 0:
                print(f"  {i + 1:,}/{len(ds):,} entries processed...")

    if batch:
        lex.add_entries(batch)
        total_added += len(batch)

    t2 = time.perf_counter()
    print(f"  Built {len(lex):,} entries in {t2 - t1:.1f}s")

    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lex.save(str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    t3 = time.perf_counter()
    print(f"  Saved {size_mb:.1f} MB in {t3 - t2:.1f}s")
    print(f"  Total time: {t3 - t0:.1f}s")


def _row_to_entry(row: dict) -> dict:
    """Convert an OpenGloss v1.3 dictionary row to a Lexicon entry dict."""
    entry: dict = {"word": row["word"]}

    for field in (
        "all_synonyms",
        "all_antonyms",
        "all_hypernyms",
        "all_hyponyms",
        "all_inflections",
        "all_derivations",
        "all_collocations",
    ):
        val = row.get(field)
        if val:
            entry[field] = list(val) if not isinstance(val, list) else val

    senses_raw = row.get("senses")
    if senses_raw:
        senses = []
        for s in senses_raw:
            if isinstance(s, str):
                try:
                    s = json.loads(s)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(s, dict):
                continue
            sense = {
                "part_of_speech": s.get("part_of_speech", ""),
                "sense_index": s.get("sense_index", 0),
                "definition": s.get("definition", ""),
            }
            for rel in ("synonyms", "antonyms", "hypernyms", "hyponyms"):
                val = s.get(rel)
                if val:
                    sense[rel] = list(val) if not isinstance(val, list) else val
            senses.append(sense)
        if senses:
            entry["senses"] = senses

    edges_raw = row.get("edges")
    if edges_raw:
        edges = []
        for e in edges_raw:
            if isinstance(e, str):
                try:
                    e = json.loads(e)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(e, dict):
                continue
            if "relationship_type" in e and "target" in e:
                edges.append(
                    {
                        "relationship_type": e["relationship_type"],
                        "target": e["target"],
                        "source_pos": e.get("source_pos"),
                        "sense_index": e.get("sense_index"),
                    }
                )
        if edges:
            entry["edges"] = edges

    return entry


def verify_lexicon(path: Path) -> None:
    """Load and verify a built Lexicon binary."""
    from kaos_nlp_core.lexicon import Lexicon

    print(f"Loading {path}...")
    t0 = time.perf_counter()
    lex = Lexicon.load(str(path))
    t1 = time.perf_counter()
    print(f"  Loaded {len(lex):,} entries in {t1 - t0:.1f}s")

    # Sample words from diverse domains to verify cross-domain coverage.
    # The lexicon should NOT be validated against only one domain.
    test_words = [
        "contract",  # legal
        "diagnosis",  # medical
        "hypothesis",  # scientific
        "revenue",  # financial
        "algorithm",  # technical
        "democracy",  # political
        "photosynthesis",  # biology
        "velocity",  # physics
    ]

    print("\nSpot-check synonyms (cross-domain):")
    for word in test_words:
        if lex.contains(word):
            syns = lex.synonyms(word)
            hyps = lex.hypernyms(word)
            infl = lex.inflections(word)
            print(f"  {word}:")
            print(f"    synonyms({len(syns)}): {syns[:5]}")
            print(f"    hypernyms({len(hyps)}): {hyps[:5]}")
            print(f"    inflections({len(infl)}): {infl[:5]}")
        else:
            print(f"  {word}: NOT FOUND")

    print("\nQuery expansion test:")
    expanded = lex.expand_query(
        ["non-compete", "agreement"],
        ["synonym", "hypernym", "inflection"],
        2,
    )
    print(f"  expand_query(['non-compete', 'agreement']): {len(expanded)} terms")
    print(f"    {sorted(expanded)[:_VERIFY_DISPLAY_TERMS]}")

    expanded2 = lex.expand_query(
        ["employment", "salary"],
        ["synonym", "hypernym"],
        1,
    )
    print(f"  expand_query(['employment', 'salary']): {len(expanded2)} terms")
    print(f"    {sorted(expanded2)[:_VERIFY_DISPLAY_TERMS]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a kaos-nlp-core Lexicon binary from any OpenGloss HuggingFace dataset."
        ),
        epilog="Examples:\n"
        "  uv run python scripts/build_opengloss_lexicon.py\n"
        "  uv run python scripts/build_opengloss_lexicon.py "
        "--dataset mjbommar/opengloss-v1.4-dictionary\n"
        "  uv run python scripts/build_opengloss_lexicon.py "
        "--verify --output data/opengloss-v1.3.lexicon.bin\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="mjbommar/opengloss-v1.3-dictionary",
        help="HuggingFace dataset ID (default: mjbommar/opengloss-v1.3-dictionary)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to data/<dataset-name>.lexicon.bin",
    )
    parser.add_argument("--verify", action="store_true", help="Verify an existing binary")
    parser.add_argument(
        "--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, help="Batch size for add_entries"
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=19,
        help=(
            "zstd compression level used by Lexicon.save() (default: 19 — "
            "maintainer-grade, slow compress / fast decompress, smallest wire "
            "size). Set to 3 for fast iteration during local development."
        ),
    )
    args = parser.parse_args(argv)

    if args.output is None:
        dataset_name = args.dataset.split("/")[-1]
        args.output = f"data/{dataset_name}.lexicon.bin"

    output_path = Path(args.output)

    if args.verify:
        verify_lexicon(output_path)
    else:
        # The Rust save path reads KAOS_NLP_ZSTD_LEVEL at write time.
        # Override here so the canonical asset ships at the smaller wire size
        # without forcing user-side runtime saves to pay the slow level-19 cost.
        import os

        os.environ["KAOS_NLP_ZSTD_LEVEL"] = str(args.compression_level)
        build_lexicon(output_path, dataset_id=args.dataset, batch_size=args.batch_size)
        print()
        verify_lexicon(output_path)


if __name__ == "__main__":
    main()
