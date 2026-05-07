"""Build the default English wordset FST shipped with kaos-nlp-core.

Loads `data/opengloss-v1.3.lexicon.bin`, extracts every headword and
inflection, lowercases + dedupes them, and writes a compact FST to
`data/english_wordset.fst`. The result is the lexicon used by the
`ratio_in_lexicon` quality metric.

Usage::

    uv run python scripts/build_default_wordset.py
    uv run python scripts/build_default_wordset.py --no-inflections
    uv run python scripts/build_default_wordset.py \
        --lexicon data/opengloss-v1.3.lexicon.bin \
        --output data/english_wordset.fst
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from kaos_nlp_core.lexicon import Lexicon

DEFAULT_LEXICON = Path("data/opengloss-v1.3.lexicon.bin")
DEFAULT_OUTPUT = Path("python/kaos_nlp_core/data/english_wordset.fst")


def build_wordset(
    lexicon_path: Path,
    output_path: Path,
    *,
    include_inflections: bool = True,
) -> None:
    if not lexicon_path.exists():
        raise SystemExit(
            f"Lexicon not found at {lexicon_path}. Run scripts/build_opengloss_lexicon.py first."
        )

    print(f"Loading lexicon from {lexicon_path} ...")
    t0 = time.perf_counter()
    lex = Lexicon.load(str(lexicon_path))
    t1 = time.perf_counter()
    print(f"  Loaded {len(lex)} entries in {t1 - t0:.1f}s")

    print(f"Building FST wordset (include_inflections={include_inflections}) ...")
    t0 = time.perf_counter()
    fst = lex.to_fst_set(include_inflections)
    t1 = time.perf_counter()
    print(f"  Built {len(fst)} keys in {t1 - t0:.1f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving FST to {output_path} ...")
    t0 = time.perf_counter()
    fst.save(str(output_path))
    t1 = time.perf_counter()
    size = output_path.stat().st_size
    print(f"  Wrote {size:,} bytes in {t1 - t0:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-inflections",
        action="store_true",
        help="Skip inflected forms (smaller FST, fewer hits).",
    )
    args = parser.parse_args(argv)

    build_wordset(
        args.lexicon,
        args.output,
        include_inflections=not args.no_inflections,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
