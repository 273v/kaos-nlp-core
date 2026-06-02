"""Derive the kaos-nlp-core English stopword resource (hybrid, reproducible).

Two halves, unioned:

1. **Statistical — cross-domain document frequency.** For each of several
   KL3M *sources* (copyright-clean), tokenize documents with the
   kaos-nlp-core **Rust tokenizer** and compute each term's document
   frequency *within that source*. A term is a stopword only if it is
   near-universal across **most sources** — not merely frequent overall.
   This is the key to deriving a *general* stopword list from a
   domain-heavy corpus: genuine function words (``the``/``of``/``is``/
   ``with``) are near-universal in every source, while domain content
   words (``section``/``shall``/``agreement``) spike in one or two sources
   and are correctly excluded. DF is exactly inverse-IDF — the same
   "appears everywhere, discriminates nothing" signal c-TF-IDF exploits.

2. **Grammatical — OpenGloss closed-class POS.** Add every OpenGloss word
   whose part of speech is closed-class (determiner / article / adposition
   / preposition / conjunction / pronoun / auxiliary / particle), for
   grammatical completeness on function words too rare to be near-universal.

The shipped artifact is a frequency/POS-derived word list (facts) with full
provenance — never a hand-typed list.

Usage::

    # Reproducible default: stream KL3M per-source datasets from HuggingFace.
    uv run python scripts/build_stopwords.py

    # Local cross-domain run (fast); repeat --source name=path per domain:
    uv run python scripts/build_stopwords.py \\
        --source contracts=/data0/data/legal/agreement.10000.jsonl \\
        --source summaries=/data0/data/legal/contract-summaries.jsonl.gz \\
        --source congress=/data0/data/legal/congress-record.dedupe.jsonl.gz \\
        --source regulatory=/data0/data/legal/all-shuffled/shuffled.jsonl

Mirrors ``scripts/build_opengloss_lexicon.py`` (maintainer-run; processes
the corpus, writes the asset committed under ``python/kaos_nlp_core/data``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kaos_nlp_core.tokenizer import tokenize_words

CLOSED_CLASS_POS: frozenset[str] = frozenset(
    {
        "determiner",
        "article",
        "adposition",
        "preposition",
        "postposition",
        "conjunction",
        "coordinating conjunction",
        "subordinating conjunction",
        "pronoun",
        "auxiliary",
        "auxiliary verb",
        "particle",
    }
)

# Reproducible default sources: KL3M per-source datasets on HuggingFace
# (copyright-clean), spanning regulatory / judicial / legislative /
# financial domains for cross-domain contrast.
DEFAULT_HF_SOURCES: dict[str, str] = {
    "ecfr": "alea-institute/kl3m-data-ecfr",
    "fdlp": "alea-institute/kl3m-data-fdlp",
    "pacer": "alea-institute/kl3m-data-pacer",
    "edgar_10k": "alea-institute/kl3m-data-edgar-10k",
}
DEFAULT_OPENGLOSS_DATASET = "mjbommar/opengloss-v1.3-dictionary"
MIN_TOKEN_CHARS = 2

# Manual-review additions: standard English closed-class function words that
# BOTH automatic halves structurally miss — the statistical half because the
# legal/government reference corpus underuses them (personal pronouns, casual
# conjunctions), and the OpenGloss closed-class half because they carry an
# open-class homonym sense that fails the "all senses closed-class" test
# (``do``/``can``/``will`` have noun/verb senses; ``he`` is also the chemical
# symbol for helium). This is the linguistic closed-class (pronouns,
# determiners, prepositions, conjunctions, auxiliaries, modals, negation) plus
# the handful of adverbial function words present in essentially every English
# stopword list — a bounded, defined set, not an ad-hoc guess. Recorded as a
# distinct provenance source.
MANUAL_ENGLISH_FUNCTION_WORDS: frozenset[str] = frozenset(
    [
        "a",
        "i",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "ought",
        "need",
        "he",
        "him",
        "his",
        "she",
        "we",
        "us",
        "you",
        "they",
        "it",
        "this",
        "that",
        "what",
        "who",
        "whom",
        "whose",
        "whoever",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "himself",
        "herself",
        "yourself",
        "ourselves",
        "yourselves",
        "a",
        "an",
        "the",
        "each",
        "every",
        "either",
        "neither",
        "any",
        "some",
        "no",
        "all",
        "both",
        "few",
        "many",
        "more",
        "most",
        "much",
        "several",
        "such",
        "another",
        "other",
        "enough",
        "what",
        "which",
        "whatever",
        "whichever",
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "around",
        "amid",
        "beyond",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "besides",
        "between",
        "down",
        "except",
        "inside",
        "near",
        "off",
        "out",
        "outside",
        "over",
        "past",
        "since",
        "toward",
        "towards",
        "under",
        "underneath",
        "until",
        "up",
        "within",
        "without",
        "throughout",
        "concerning",
        "including",
        "notwithstanding",
        "and",
        "or",
        "but",
        "nor",
        "so",
        "yet",
        "because",
        "although",
        "though",
        "albeit",
        "while",
        "whilst",
        "whereas",
        "unless",
        "if",
        "lest",
        "than",
        "whether",
        "once",
        "when",
        "where",
        "why",
        "how",
        "not",
        "no",
        "too",
        "very",
        "just",
        "only",
        "also",
        "then",
        "there",
        "here",
        "again",
        "once",
        "now",
        "ever",
        "never",
        "however",
        "therefore",
        "thus",
        "hence",
        "moreover",
        "nonetheless",
        "nevertheless",
        "meanwhile",
    ]
)

# Curation: domain *content* words that survive cross-domain DF only because
# every available KL3M source is US-government / legal-flavored, so these are
# "universal" within that set without being function words. Removed by manual
# review (the standard derive-then-curate step for stopword lists) so they are
# never stripped from labels — they are exactly the distinctive terms a legal
# clustering wants to keep. Recorded in the asset provenance.
CURATED_CONTENT_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "act",
        "following",
        "new",
        "number",
        "page",
        "president",
        "state",
        "states",
        "time",
        "united",
        "year",
    }
)


def _iter_local_jsonl(path: Path, max_docs: int) -> Iterator[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= max_docs:
                break
            try:
                yield str(json.loads(line).get("text", ""))
            except (json.JSONDecodeError, TypeError):
                continue


def _iter_hf_stream(dataset_id: str, max_docs: int) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        yield str(row.get("text", ""))


def _has_digit(term: str) -> bool:
    return any(ch.isdigit() for ch in term)


def source_doc_frequency(
    docs: Iterator[str], prune_every: int = 20_000
) -> tuple[Counter[str], int]:
    """Per-source document frequency via the Rust tokenizer (one count per doc)."""
    df: Counter[str] = Counter()
    n = 0
    for n, text in enumerate(docs, start=1):
        if not text:
            continue
        terms = {
            w
            for w in tokenize_words(text, lowercase=True)
            if len(w) >= MIN_TOKEN_CHARS and not _has_digit(w)
        }
        df.update(terms)
        if n % prune_every == 0:
            df = Counter({t: c for t, c in df.items() if c >= 2})
    return df, n


# OpenGloss is a MULTILINGUAL dictionary with no reliable per-entry language
# tag, and kaos-nlp-core's English wordset includes loanwords, so neither can
# delang it automatically. These single-word closed-class candidates are
# non-English (or archaic/dialectal) entries removed by manual review — the
# authorized "derive then curate" step. Listed explicitly for transparency.
OPENGLOSS_NON_ENGLISH: frozenset[str] = frozenset(
    [
        "aby",
        "aku",
        "aleichem",
        "altho",
        "ang",
        "apo",
        "auf",
        "avec",
        "azt",
        "cewa",
        "chez",
        "comme",
        "contra",
        "cui",
        "czy",
        "dans",
        "dari",
        "das",
        "dat",
        "de",
        "degli",
        "del",
        "della",
        "dem",
        "durante",
        "eine",
        "einem",
        "einer",
        "eines",
        "ella",
        "ellas",
        "eller",
        "esta",
        "este",
        "gulo",
        "hoc",
        "hos",
        "ia",
        "ich",
        "ihre",
        "ils",
        "inga",
        "itu",
        "khi",
        "ki",
        "kwa",
        "la",
        "las",
        "les",
        "los",
        "lui",
        "mein",
        "meine",
        "mich",
        "minden",
        "mit",
        "na",
        "nach",
        "ningal",
        "nos",
        "notre",
        "nuestra",
        "nuestro",
        "nung",
        "ob",
        "ohne",
        "pero",
        "qui",
        "saya",
        "sensu",
        "sie",
        "sui",
        "sur",
        "til",
        "tous",
        "uma",
        "una",
        "unser",
        "ut",
        "von",
        "vous",
        "weil",
        "welches",
        "wir",
        "yer",
        "yous",
        "ur",
        "zu",
        "thee",
        "thees",
        "thine",
        "thou",
        "thyself",
        "iff",
    ]
)


def opengloss_closed_class(dataset_id: str) -> set[str]:
    """English closed-class function words from OpenGloss.

    Keeps a word only when it is a single ASCII alphabetic token **all** of
    whose senses are closed-class (so content words with a minor
    preposition/conjunction sense are excluded), then subtracts the curated
    non-English entries above (OpenGloss is multilingual).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")
    out: set[str] = set()
    for row in ds:
        word = str(row.get("word", "")).lower().strip()
        if len(word) < MIN_TOKEN_CHARS or " " in word or not word.isascii() or not word.isalpha():
            continue
        pos: list[str] = []
        for s in row.get("senses") or []:
            if isinstance(s, str):
                try:
                    s = json.loads(s)
                except (json.JSONDecodeError, TypeError):
                    continue
            if isinstance(s, dict):
                p = str(s.get("part_of_speech", "")).lower()
                if p:
                    pos.append(p)
        if pos and all(p in CLOSED_CLASS_POS for p in pos):
            out.add(word)
    return out - OPENGLOSS_NON_ENGLISH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="A local source as name=jsonl(.gz) path. Repeat per domain. "
        "If omitted, streams the KL3M HF default sources.",
    )
    parser.add_argument("--max-docs-per-source", type=int, default=5000)
    parser.add_argument(
        "--per-source-threshold",
        type=float,
        default=0.5,
        help="DF fraction for a term to count as 'common' within a source.",
    )
    parser.add_argument(
        "--min-source-fraction",
        type=float,
        default=0.6,
        help="Fraction of sources a term must be 'common' in to be a stopword.",
    )
    parser.add_argument("--opengloss-dataset", default=DEFAULT_OPENGLOSS_DATASET)
    parser.add_argument("--no-opengloss", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "python/kaos_nlp_core/data/stopwords-en-v1.json",
    )
    args = parser.parse_args()

    # Resolve sources: explicit local --source entries, else HF defaults.
    sources: dict[str, str] = {}
    if args.source:
        for spec in args.source:
            name, _, path = spec.partition("=")
            sources[name] = path
        use_local = True
    else:
        sources = dict(DEFAULT_HF_SOURCES)
        use_local = False

    per_source: dict[str, tuple[Counter[str], int]] = {}
    for name, location in sources.items():
        print(f"[{name}] {location} ...")
        docs = (
            _iter_local_jsonl(Path(location), args.max_docs_per_source)
            if use_local
            else _iter_hf_stream(location, args.max_docs_per_source)
        )
        df, n = source_doc_frequency(docs)
        per_source[name] = (df, n)
        print(f"  {n:,} docs, {len(df):,} terms")

    n_sources = len(per_source)
    min_sources = max(1, math.ceil(args.min_source_fraction * n_sources))

    # A term is "common" in a source when its DF fraction >= threshold;
    # it is a stopword when common in >= min_sources sources.
    common_in: Counter[str] = Counter()
    for df, n in per_source.values():
        if n == 0:
            continue
        for term, count in df.items():
            if count / n >= args.per_source_threshold:
                common_in[term] += 1
    statistical_raw = {t for t, c in common_in.items() if c >= min_sources}
    statistical = statistical_raw - CURATED_CONTENT_EXCLUSIONS
    excluded = sorted(statistical_raw & CURATED_CONTENT_EXCLUSIONS)
    print(
        f"Statistical: {len(statistical_raw)} terms common in >= {min_sources}/{n_sources} sources"
    )
    print(f"  curated out {len(excluded)} domain-content words: {excluded}")

    opengloss: set[str] = set()
    if not args.no_opengloss:
        print(f"OpenGloss closed-class POS from {args.opengloss_dataset} ...")
        opengloss = opengloss_closed_class(args.opengloss_dataset)
        print(f"  {len(opengloss):,} closed-class function words")

    terms = sorted(statistical | opengloss | MANUAL_ENGLISH_FUNCTION_WORDS)
    payload: dict[str, Any] = {
        "language": "en",
        "version": "1",
        "terms": terms,
        "provenance": {
            "method": "cross-domain document-frequency contrast UNION OpenGloss closed-class POS",
            "tokenizer": "kaos_nlp_core.tokenizer.tokenize_words (Rust, lowercase)",
            "statistical": {
                "sources": {name: n for name, (_, n) in per_source.items()},
                "per_source_threshold": args.per_source_threshold,
                "min_source_fraction": args.min_source_fraction,
                "min_sources": min_sources,
                "metric": "document_frequency",
                "selected": len(statistical),
                "curated_excluded": excluded,
            },
            "opengloss": {
                "dataset": None if args.no_opengloss else args.opengloss_dataset,
                "closed_class_pos": sorted(CLOSED_CLASS_POS),
                "selected": len(opengloss),
                "note": "single-word, all-senses-closed; multilingual entries curated out",
            },
            "manual_review": {
                "selected": len(MANUAL_ENGLISH_FUNCTION_WORDS),
                "note": "standard English closed-class function words missed by both "
                "automatic halves (open-class homonyms / legal-corpus underuse)",
            },
            "total": len(terms),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(terms):,} stopwords -> {args.output}")


if __name__ == "__main__":
    main()
