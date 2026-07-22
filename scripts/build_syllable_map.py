"""Build the bundled CMUdict-derived syllable map shipped with kaos-nlp-core.

Parses a CMU Pronouncing Dictionary file (``cmudict.dict`` from
https://github.com/cmusphinx/cmudict, 2-clause BSD), takes the FIRST
pronunciation of each word, counts its syllables as the number of
phonemes carrying a stress digit, and writes a compact word → count
``fst::Map`` consumed by ``kaos_nlp_core.readability``.

Only entries whose headword is pure letters/apostrophes are kept, which
matches the analyzer's normalized lookup keys. Alternate pronunciations
(``word(2)``) are skipped so loading is deterministic.

Also writes an optional evaluation fixture: a deterministic 1-in-N
sample of (word, count) pairs used by the syllable-accuracy regression
test.

Usage::

    uv run python scripts/build_syllable_map.py --source cmudict.dict
    uv run python scripts/build_syllable_map.py \
        --source cmudict.dict \
        --output python/kaos_nlp_core/data/cmudict_syllables.fst \
        --fixture tests/fixtures/syllables_cmudict_sample.tsv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from kaos_nlp_core._rust.readability import SyllableMap

DEFAULT_OUTPUT = Path("python/kaos_nlp_core/data/cmudict_syllables.fst")
DEFAULT_FIXTURE = Path("tests/fixtures/syllables_cmudict_sample.tsv")
FIXTURE_STRIDE = 250  # deterministic 1-in-250 sample (~500 words)


def parse_cmudict(path: Path) -> dict[str, int]:
    """Return word -> syllable count from first pronunciations only."""
    out: dict[str, int] = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line or line.startswith(";;;"):
            continue
        word, _, phones = line.partition(" ")
        if "(" in word:  # alternate pronunciation
            continue
        w = word.lower()
        if not w or not all((c.isalpha() and c.isascii()) or c == "'" for c in w):
            continue
        if not any(c.isalpha() for c in w):
            continue
        phones = phones.split("#")[0]  # strip trailing comments
        n = sum(1 for p in phones.split() if p and p[-1].isdigit())
        if n > 0:
            out[w] = n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Path to cmudict.dict")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help=f"Also write a 1-in-{FIXTURE_STRIDE} TSV sample (word<TAB>count)",
    )
    args = parser.parse_args()

    print(f"Parsing {args.source} ...")
    t0 = time.perf_counter()
    entries = parse_cmudict(args.source)
    print(f"  {len(entries)} entries in {time.perf_counter() - t0:.1f}s")

    print("Building FST map ...")
    t0 = time.perf_counter()
    fst = SyllableMap(sorted(entries.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fst.save(str(args.output))
    size = args.output.stat().st_size
    print(f"  Wrote {size:,} bytes to {args.output} in {time.perf_counter() - t0:.1f}s")

    if args.fixture is not None:
        sample = sorted(entries.items())[::FIXTURE_STRIDE]
        lines = [f"# word\tsyllables — deterministic 1-in-{FIXTURE_STRIDE} sample of CMUdict"]
        lines += [f"{w}\t{n}" for w, n in sample]
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        args.fixture.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  Wrote {len(sample)} fixture rows to {args.fixture}")


if __name__ == "__main__":
    main()
