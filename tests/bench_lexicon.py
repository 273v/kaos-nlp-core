"""Benchmarks for Lexicon query expansion + BM25 retrieval pipeline.

Measures:
  - Lexicon build time and memory
  - Query expansion latency at various depths
  - Expanded vs unexpanded BM25 query latency
  - Full pipeline: expand + search
"""

import pickle

import pytest

from kaos_nlp_core.lexicon import Lexicon
from kaos_nlp_core.structures import InvertedIndex


def tokenize_doc(text: str) -> list[str]:
    """Tokenize using the shared ICU-based tokenizer."""
    from conftest import DEFAULT_TOKENIZER

    return DEFAULT_TOKENIZER.tokenize_words(text)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def regulatory_lexicon():
    """Multi-domain lexicon (~15 entries)."""
    lex = Lexicon()
    entries = [
        {
            "word": "alcohol",
            "all_synonyms": ["liquor", "spirits", "ethanol", "intoxicant"],
            "all_hypernyms": ["beverage", "substance"],
            "all_hyponyms": ["beer", "wine", "whiskey", "vodka", "rum"],
            "all_inflections": [],
            "all_collocations": ["blood alcohol", "alcohol abuse"],
        },
        {
            "word": "beverage",
            "all_synonyms": ["drink", "refreshment"],
            "all_hyponyms": ["alcohol", "juice", "water", "soda", "tea", "coffee"],
            "all_inflections": ["beverages"],
        },
        {
            "word": "liquor",
            "all_synonyms": ["alcohol", "spirits", "booze"],
            "all_hyponyms": ["whiskey", "vodka", "rum", "gin"],
            "all_inflections": ["liquors"],
        },
        {
            "word": "firearm",
            "all_synonyms": ["gun", "weapon"],
            "all_hypernyms": ["weapon", "armament"],
            "all_hyponyms": ["rifle", "pistol", "shotgun", "revolver", "handgun"],
            "all_inflections": ["firearms"],
        },
        {
            "word": "weapon",
            "all_synonyms": ["arm", "armament"],
            "all_hyponyms": ["firearm", "knife", "explosive", "missile"],
            "all_inflections": ["weapons"],
        },
        {
            "word": "ammunition",
            "all_synonyms": ["ammo", "munitions", "rounds"],
            "all_hyponyms": ["bullet", "cartridge", "shell"],
        },
        {
            "word": "drug",
            "all_synonyms": ["narcotic", "controlled substance", "medication", "pharmaceutical"],
            "all_hyponyms": ["opioid", "cocaine", "marijuana", "heroin"],
            "all_inflections": ["drugs"],
        },
        {
            "word": "narcotic",
            "all_synonyms": ["drug", "opiate", "controlled substance"],
            "all_hyponyms": ["heroin", "morphine", "fentanyl"],
            "all_inflections": ["narcotics"],
        },
        {
            "word": "vehicle",
            "all_synonyms": ["automobile", "conveyance"],
            "all_hyponyms": ["automobile", "truck", "bus", "motorcycle"],
            "all_inflections": ["vehicles"],
        },
        {
            "word": "automobile",
            "all_synonyms": ["car", "vehicle", "motorcar"],
            "all_hyponyms": ["sedan", "coupe", "SUV"],
            "all_inflections": ["automobiles"],
        },
        {
            "word": "pollution",
            "all_synonyms": ["contamination", "degradation"],
            "all_hyponyms": ["air pollution", "water pollution"],
            "all_inflections": [],
        },
        {
            "word": "emission",
            "all_synonyms": ["discharge", "release"],
            "all_hyponyms": ["carbon emission", "exhaust", "greenhouse gas"],
            "all_inflections": ["emissions"],
        },
        {"word": "hazardous", "all_synonyms": ["dangerous", "toxic", "harmful"]},
        {
            "word": "tax",
            "all_synonyms": ["levy", "duty", "tariff", "assessment"],
            "all_hyponyms": ["income tax", "sales tax", "estate tax"],
            "all_inflections": ["taxes", "taxed", "taxing"],
        },
        {
            "word": "securities",
            "all_synonyms": ["stocks", "bonds", "investments"],
            "all_hyponyms": ["stock", "bond", "option", "future"],
            "all_inflections": [],
        },
    ]
    for e in entries:
        lex.add_entry(e)
    return lex


@pytest.fixture(scope="module")
def usc_index(usc_docs):
    idx = InvertedIndex()
    for d in usc_docs:
        idx.add_document(d["id"], tokenize_doc(d["text"]))
    return idx


# ── Lexicon operation benchmarks ─────────────────────────────────────────────


@pytest.mark.benchmark(group="lexicon_ops")
def test_lexicon_build(benchmark):
    """Build a 15-entry lexicon."""
    entries = [
        {
            "word": f"term_{i}",
            "all_synonyms": [f"syn_{i}_{j}" for j in range(5)],
            "all_hyponyms": [f"hypo_{i}_{j}" for j in range(3)],
        }
        for i in range(15)
    ]

    def build():
        lex = Lexicon()
        for e in entries:
            lex.add_entry(e)
        return len(lex)

    benchmark(build)


@pytest.mark.benchmark(group="lexicon_ops")
def test_lexicon_build_1000(benchmark):
    """Build a 1000-entry lexicon."""
    entries = [
        {
            "word": f"term_{i}",
            "all_synonyms": [f"syn_{i}_{j}" for j in range(10)],
            "all_hyponyms": [f"hypo_{i}_{j}" for j in range(5)],
            "all_inflections": [f"infl_{i}_{j}" for j in range(3)],
        }
        for i in range(1000)
    ]

    def build():
        lex = Lexicon()
        for e in entries:
            lex.add_entry(e)
        return len(lex)

    result = benchmark.pedantic(build, rounds=10, iterations=1)
    assert result == 1000


@pytest.mark.benchmark(group="lexicon_ops")
def test_lexicon_synonym_lookup(benchmark, regulatory_lexicon):
    """Synonym lookup latency."""
    benchmark(regulatory_lexicon.synonyms, "alcohol")


@pytest.mark.benchmark(group="lexicon_ops")
def test_lexicon_expand_depth1(benchmark, regulatory_lexicon):
    """Single-hop expansion (synonym + inflection)."""
    benchmark(regulatory_lexicon.expand_query, ["alcohol"], ["synonym", "inflection"], 1)


@pytest.mark.benchmark(group="lexicon_ops")
def test_lexicon_expand_depth2(benchmark, regulatory_lexicon):
    """Two-hop expansion (synonym + hyponym)."""
    benchmark(regulatory_lexicon.expand_query, ["beverage"], ["synonym", "hyponym"], 2)


@pytest.mark.benchmark(group="lexicon_ops")
def test_lexicon_expand_multi_relation(benchmark, regulatory_lexicon):
    """Multi-relation expansion."""
    benchmark(
        regulatory_lexicon.expand_query,
        ["firearm"],
        ["synonym", "hyponym", "inflection", "collocation"],
        1,
    )


@pytest.mark.benchmark(group="lexicon_ops")
def test_lexicon_pickle_roundtrip(benchmark, regulatory_lexicon):
    """Pickle serialize + deserialize."""

    def roundtrip():
        data = pickle.dumps(regulatory_lexicon)
        return pickle.loads(data)

    result = benchmark(roundtrip)
    assert len(result) == len(regulatory_lexicon)


# ── Expansion + BM25 pipeline benchmarks ─────────────────────────────────────


@pytest.mark.benchmark(group="expand_search")
def test_unexpanded_bm25(benchmark, usc_index):
    """Baseline: BM25 with single term."""
    benchmark(usc_index.query_bm25, ["alcohol"], 20)


@pytest.mark.benchmark(group="expand_search")
def test_expanded_bm25_alcohol(benchmark, usc_index, regulatory_lexicon):
    """Expanded: BM25 with synonym-expanded 'alcohol'."""
    expanded = list(
        regulatory_lexicon.expand_query(["alcohol"], ["synonym", "hyponym"], max_depth=1)
    )

    benchmark(usc_index.query_bm25, expanded, 20)


@pytest.mark.benchmark(group="expand_search")
def test_full_pipeline_alcohol(benchmark, usc_index, regulatory_lexicon):
    """Full pipeline: expand + BM25 in one call."""

    def pipeline():
        expanded = list(
            regulatory_lexicon.expand_query(
                ["alcohol"], ["synonym", "hyponym", "inflection"], max_depth=1
            )
        )
        return usc_index.query_bm25(expanded, 20)

    benchmark(pipeline)


@pytest.mark.benchmark(group="expand_search")
def test_full_pipeline_firearm(benchmark, usc_index, regulatory_lexicon):
    """Full pipeline: expand + BM25 for firearms."""

    def pipeline():
        expanded = list(
            regulatory_lexicon.expand_query(
                ["firearm"], ["synonym", "hyponym", "inflection"], max_depth=1
            )
        )
        return usc_index.query_bm25(expanded, 20)

    benchmark(pipeline)


@pytest.mark.benchmark(group="expand_search")
def test_full_pipeline_multi_term(benchmark, usc_index, regulatory_lexicon):
    """Full pipeline: expand multi-term query."""

    def pipeline():
        expanded = list(
            regulatory_lexicon.expand_query(
                ["drug", "narcotic"],
                ["synonym", "hyponym", "inflection"],
                max_depth=1,
            )
        )
        return usc_index.query_bm25(expanded, 20)

    benchmark(pipeline)


# ── Memory measurement ──────────────────────────────────────────────────────


@pytest.mark.benchmark(group="lexicon_memory")
def test_lexicon_pickle_size(benchmark, regulatory_lexicon):
    """Measure serialized size of the lexicon."""

    def measure():
        data = pickle.dumps(regulatory_lexicon)
        return len(data)

    _size = benchmark(measure)
    # Just report — no assertion needed
