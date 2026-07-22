"""Build a familiar-word FST for Dale-Chall readability scoring.

Converts a plain-text word list (one word per line, ``#`` comments and
blank lines ignored) into the compact FST format accepted by
``kaos_nlp_core.readability`` via ``--familiar-words`` / the
``familiar_words=`` argument. Words are lowercased and deduplicated.

The 1995 revised Dale-Chall list (~3,000 words) comes from a
copyrighted book (Chall & Dale, *Readability Revisited*, 1995) and is
deliberately **not** bundled with this package; supply your own copy or
any other familiar-word list.

Usage::

    uv run python scripts/build_familiar_wordset.py words.txt familiar.fst
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kaos_nlp_core.matching import FstSet


def build(source: Path, output: Path) -> int:
    words: set[str] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        word = line.split("#", 1)[0].strip().lower()
        if word:
            words.add(word)
    if not words:
        raise SystemExit(f"No words found in {source}")
    fst = FstSet(sorted(words))
    output.parent.mkdir(parents=True, exist_ok=True)
    fst.save(str(output))
    return len(words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Plain-text word list (one word per line)")
    parser.add_argument("output", type=Path, help="Output FST path")
    args = parser.parse_args()
    count = build(args.source, args.output)
    size = args.output.stat().st_size
    print(f"Wrote {count} words ({size:,} bytes) to {args.output}")


if __name__ == "__main__":
    main()
