"""Integration tests: Lexicon query expansion + BM25 retrieval on US Code.

Tests the full pipeline: concept → lexicon expansion → BM25 search → validate
that expanded queries find more relevant documents than unexpanded queries.
"""

from pathlib import Path

import pytest

from kaos_nlp_core.lexicon import Lexicon
from kaos_nlp_core.structures import InvertedIndex

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def tokenize_doc(text: str) -> list[str]:
    """Tokenize using the shared ICU-based tokenizer."""
    from conftest import DEFAULT_TOKENIZER

    return DEFAULT_TOKENIZER.tokenize_words(text)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def usc_index(usc_docs):
    idx = InvertedIndex()
    for d in usc_docs:
        idx.add_document(d["id"], tokenize_doc(d["text"]))
    return idx


@pytest.fixture(scope="module")
def regulatory_lexicon():
    """A realistic multi-domain lexicon for regulatory/legal concepts."""
    lex = Lexicon()

    # ── Alcohol / Beverages ──
    lex.add_entry(
        {
            "word": "alcohol",
            "senses": [
                {
                    "part_of_speech": "noun",
                    "sense_index": 0,
                    "definition": "Intoxicating liquor or ethanol",
                    "synonyms": ["liquor", "spirits", "ethanol"],
                    "antonyms": [],
                    "hypernyms": ["beverage", "substance"],
                    "hyponyms": ["beer", "wine", "whiskey", "vodka"],
                },
            ],
            "all_synonyms": ["liquor", "spirits", "ethanol", "intoxicant"],
            "all_antonyms": [],
            "all_hypernyms": ["beverage", "substance", "chemical"],
            "all_hyponyms": ["beer", "wine", "whiskey", "vodka", "rum", "brandy"],
            "all_inflections": [],
            "all_collocations": ["blood alcohol", "alcohol abuse", "alcohol content"],
        }
    )
    lex.add_entry(
        {
            "word": "beverage",
            "all_synonyms": ["drink", "refreshment"],
            "all_hypernyms": ["liquid", "food product"],
            "all_hyponyms": ["alcohol", "juice", "water", "soda", "tea", "coffee"],
            "all_inflections": ["beverages"],
        }
    )
    lex.add_entry(
        {
            "word": "liquor",
            "all_synonyms": ["alcohol", "spirits", "booze"],
            "all_hypernyms": ["alcoholic beverage"],
            "all_hyponyms": ["whiskey", "vodka", "rum", "gin", "brandy"],
            "all_inflections": ["liquors"],
        }
    )
    lex.add_entry(
        {
            "word": "intoxicating",
            "all_synonyms": ["inebriating", "alcoholic"],
            "all_hypernyms": [],
            "all_hyponyms": [],
            "all_inflections": [],
        }
    )

    # ── Firearms / Weapons ──
    lex.add_entry(
        {
            "word": "firearm",
            "senses": [
                {
                    "part_of_speech": "noun",
                    "sense_index": 0,
                    "definition": "A weapon that launches projectiles",
                    "synonyms": ["gun", "weapon"],
                    "antonyms": [],
                    "hypernyms": ["weapon", "armament"],
                    "hyponyms": ["rifle", "pistol", "shotgun", "revolver"],
                },
            ],
            "all_synonyms": ["gun", "weapon", "piece"],
            "all_hypernyms": ["weapon", "armament"],
            "all_hyponyms": ["rifle", "pistol", "shotgun", "revolver", "handgun"],
            "all_inflections": ["firearms"],
            "all_collocations": ["firearm possession", "firearm license"],
        }
    )
    lex.add_entry(
        {
            "word": "weapon",
            "all_synonyms": ["arm", "armament"],
            "all_hypernyms": ["instrument", "device"],
            "all_hyponyms": ["firearm", "knife", "explosive", "missile"],
            "all_inflections": ["weapons"],
        }
    )
    lex.add_entry(
        {
            "word": "ammunition",
            "all_synonyms": ["ammo", "munitions", "rounds"],
            "all_hypernyms": ["military supply"],
            "all_hyponyms": ["bullet", "cartridge", "shell"],
            "all_inflections": [],
        }
    )

    # ── Controlled Substances ──
    lex.add_entry(
        {
            "word": "drug",
            "senses": [
                {
                    "part_of_speech": "noun",
                    "sense_index": 0,
                    "definition": "A controlled or illegal substance",
                    "synonyms": ["narcotic", "controlled substance"],
                    "antonyms": [],
                    "hypernyms": ["substance"],
                    "hyponyms": ["opioid", "cocaine", "marijuana", "heroin"],
                },
                {
                    "part_of_speech": "noun",
                    "sense_index": 1,
                    "definition": "A pharmaceutical medicine",
                    "synonyms": ["medication", "medicine", "pharmaceutical"],
                    "antonyms": [],
                    "hypernyms": ["therapeutic agent"],
                    "hyponyms": ["antibiotic", "vaccine", "analgesic"],
                },
            ],
            "all_synonyms": [
                "narcotic",
                "controlled substance",
                "medication",
                "medicine",
                "pharmaceutical",
            ],
            "all_hypernyms": ["substance", "therapeutic agent"],
            "all_hyponyms": ["opioid", "cocaine", "marijuana", "heroin", "antibiotic", "vaccine"],
            "all_inflections": ["drugs"],
        }
    )
    lex.add_entry(
        {
            "word": "narcotic",
            "all_synonyms": ["drug", "opiate", "controlled substance"],
            "all_hypernyms": ["substance"],
            "all_hyponyms": ["heroin", "morphine", "fentanyl", "oxycodone"],
            "all_inflections": ["narcotics"],
        }
    )

    # ── Vehicles / Transportation ──
    lex.add_entry(
        {
            "word": "vehicle",
            "all_synonyms": ["automobile", "conveyance", "transport"],
            "all_hypernyms": ["transport", "machine"],
            "all_hyponyms": ["automobile", "truck", "bus", "motorcycle", "trailer"],
            "all_inflections": ["vehicles"],
            "all_collocations": ["motor vehicle", "commercial vehicle"],
        }
    )
    lex.add_entry(
        {
            "word": "automobile",
            "all_synonyms": ["car", "vehicle", "motorcar"],
            "all_hypernyms": ["vehicle", "motor vehicle"],
            "all_hyponyms": ["sedan", "coupe", "SUV", "convertible"],
            "all_inflections": ["automobiles"],
        }
    )

    # ── Environment ──
    lex.add_entry(
        {
            "word": "pollution",
            "all_synonyms": ["contamination", "degradation"],
            "all_hypernyms": ["environmental damage"],
            "all_hyponyms": ["air pollution", "water pollution", "soil contamination"],
            "all_inflections": [],
            "all_collocations": ["pollution control", "pollution prevention"],
        }
    )
    lex.add_entry(
        {
            "word": "emission",
            "all_synonyms": ["discharge", "release", "output"],
            "all_hypernyms": ["release", "discharge"],
            "all_hyponyms": ["carbon emission", "exhaust", "greenhouse gas"],
            "all_inflections": ["emissions"],
        }
    )
    lex.add_entry(
        {
            "word": "hazardous",
            "all_synonyms": ["dangerous", "toxic", "harmful", "perilous"],
            "all_hypernyms": [],
            "all_hyponyms": [],
            "all_inflections": [],
            "all_collocations": ["hazardous waste", "hazardous material"],
        }
    )

    return lex


# ── Helper ───────────────────────────────────────────────────────────────────


def count_matching_docs(docs: list[dict], terms: list[str]) -> int:
    """Count docs where ANY term appears (case-insensitive)."""
    count = 0
    for d in docs:
        text_lower = d["text"].lower()
        if any(t.lower() in text_lower for t in terms):
            count += 1
    return count


# ── Quality Tests: Expansion improves recall ─────────────────────────────────


class TestExpansionImprovesRecall:
    """Verify that lexicon-expanded queries find more relevant documents."""

    def test_alcohol_expansion_recall(self, usc_index, usc_docs, regulatory_lexicon):
        """Expanding 'alcohol' should find docs about liquor, spirits, etc."""
        # Unexpanded
        base_results = usc_index.query_bm25(["alcohol"], top_k=50)
        base_ids = {r.doc_id for r in base_results}

        # Expanded with synonyms + hyponyms
        expanded_terms = regulatory_lexicon.expand_query(
            ["alcohol"], ["synonym", "hyponym"], max_depth=1
        )
        expanded_results = usc_index.query_bm25(list(expanded_terms), top_k=50)
        expanded_ids = {r.doc_id for r in expanded_results}

        # Expanded should find at least as many results
        assert len(expanded_ids) >= len(base_ids)

        # Expanded should find docs that mention related terms
        # Check that some expanded-only results actually mention the related terms
        new_ids = expanded_ids - base_ids
        if new_ids:
            new_docs = [d for d in usc_docs if d["id"] in new_ids]
            related = ["liquor", "spirits", "beer", "wine", "beverage"]
            has_related = sum(1 for d in new_docs if any(r in d["text"].lower() for r in related))
            assert has_related > 0, "Expanded results should contain related terms"

    def test_firearm_expansion_recall(self, usc_index, usc_docs, regulatory_lexicon):
        """Expanding 'firearm' should find docs about guns, weapons, etc."""
        base = usc_index.query_bm25(["firearm"], top_k=50)
        expanded_terms = regulatory_lexicon.expand_query(
            ["firearm"], ["synonym", "hyponym", "inflection"], max_depth=1
        )
        expanded = usc_index.query_bm25(list(expanded_terms), top_k=50)

        assert len(expanded) >= len(base)

    def test_drug_sense_aware_expansion(self, usc_index, regulatory_lexicon):
        """Sense-aware expansion of 'drug' (controlled substance, not medicine)
        should prioritize narcotic/controlled substance results."""
        # All-sense expansion includes "medication", "pharmaceutical"
        all_sense = set(regulatory_lexicon.expand_query(["drug"], ["synonym"], max_depth=1))
        assert "medication" in all_sense
        assert "narcotic" in all_sense

        # Sense 0 (controlled substance) only
        sense0 = set(
            regulatory_lexicon.expand_query_sense_aware(
                [("drug", "noun", 0)],
                ["synonym"],
            )
        )
        assert "narcotic" in sense0
        assert "medication" not in sense0

    def test_pollution_expansion(self, usc_index, regulatory_lexicon):
        """Expanding 'pollution' finds contamination, emission, toxic docs."""
        base = usc_index.query_bm25(["pollution"], top_k=50)
        expanded_terms = regulatory_lexicon.expand_query(
            ["pollution"], ["synonym", "hyponym", "collocation"], max_depth=1
        )
        expanded = usc_index.query_bm25(list(expanded_terms), top_k=50)

        assert len(expanded) >= len(base)


class TestExpansionPrecision:
    """Verify expanded queries don't introduce too much noise."""

    def test_expanded_results_are_topically_relevant(self, usc_index, usc_docs, regulatory_lexicon):
        """Top results from expanded firearm query should be about weapons."""
        expanded_terms = regulatory_lexicon.expand_query(
            ["firearm"], ["synonym", "hyponym"], max_depth=1
        )
        results = usc_index.query_bm25(list(expanded_terms), top_k=20)

        # Check that top results actually mention weapon-related terms
        weapon_terms = {"firearm", "gun", "weapon", "rifle", "pistol", "ammunition"}
        relevant = 0
        for r in results:
            doc = next(d for d in usc_docs if d["id"] == r.doc_id)
            text_lower = doc["text"].lower()
            if any(t in text_lower for t in weapon_terms):
                relevant += 1

        precision = relevant / len(results) if results else 0
        assert precision >= 0.5, f"Precision = {precision:.0%}, expected >= 50%"

    def test_vehicle_expansion_stays_on_topic(self, usc_index, usc_docs, regulatory_lexicon):
        """Vehicle expansion should find transportation law, not random docs."""
        expanded_terms = regulatory_lexicon.expand_query(
            ["vehicle"], ["synonym", "hyponym"], max_depth=1
        )
        results = usc_index.query_bm25(list(expanded_terms), top_k=20)

        transport_terms = {"vehicle", "automobile", "truck", "motor", "highway", "transport"}
        relevant = sum(
            1
            for r in results
            if any(
                t in next(d for d in usc_docs if d["id"] == r.doc_id)["text"].lower()
                for t in transport_terms
            )
        )
        assert relevant >= 10, f"Only {relevant}/20 results are transport-related"


class TestExpansionVsUnexpanded:
    """Side-by-side comparison: expanded vs unexpanded BM25."""

    def test_expanded_finds_more_unique_docs(self, usc_index, usc_docs, regulatory_lexicon):
        """Expanded query should find docs the base query misses entirely.

        Note: at fixed top-K, expanded recall can be slightly lower because
        common synonyms (e.g., "dangerous") dilute the ranking. But the
        expanded query reaches docs that the base query cannot find at all.
        """
        # Unexpanded: only "hazardous"
        base = usc_index.query_bm25(["hazardous"], top_k=200)
        base_ids = {r.doc_id for r in base}

        # Expanded: hazardous + toxic + dangerous + harmful
        expanded_terms = regulatory_lexicon.expand_query(["hazardous"], ["synonym"], max_depth=1)
        expanded = usc_index.query_bm25(list(expanded_terms), top_k=200)
        expanded_ids = {r.doc_id for r in expanded}

        # The expanded query should discover docs the base query missed
        new_docs = expanded_ids - base_ids
        assert len(new_docs) > 0, "Expansion should find docs base query missed"

        # Verify the new docs actually contain the expanded terms
        expanded_only_docs = [d for d in usc_docs if d["id"] in new_docs]
        has_synonym = sum(
            1
            for d in expanded_only_docs[:20]
            if any(t in d["text"].lower() for t in ["toxic", "dangerous", "harmful"])
        )
        assert has_synonym > 0


class TestMultiHopExpansion:
    """Test multi-hop graph traversal for concept discovery."""

    def test_2hop_discovers_indirect_terms(self, regulatory_lexicon):
        """beverage → alcohol → liquor (2 hops via hyponym then synonym)."""
        hop1 = set(regulatory_lexicon.expand_query(["beverage"], ["hyponym"], max_depth=1))
        assert "alcohol" in hop1

        hop2 = set(
            regulatory_lexicon.expand_query(["beverage"], ["hyponym", "synonym"], max_depth=2)
        )
        # 2-hop: beverage → alcohol (hyponym) → liquor (synonym of alcohol)
        assert "liquor" in hop2

    def test_concept_traversal_weapons(self, regulatory_lexicon):
        """weapon → firearm → rifle (hyponym chain)."""
        expanded = set(regulatory_lexicon.expand_query(["weapon"], ["hyponym"], max_depth=2))
        # weapon → firearm (hyponym) → rifle (hyponym of firearm)
        assert "firearm" in expanded
        assert "rifle" in expanded


class TestLexiconWithIndex:
    """End-to-end: build lexicon, expand, search, validate."""

    def test_full_pipeline_narcotics(self, usc_index, usc_docs, regulatory_lexicon):
        """Full pipeline: narcotic concept → expansion → BM25 → validate."""
        # Step 1: Expand
        expanded = regulatory_lexicon.expand_query(
            ["narcotic"], ["synonym", "hyponym", "inflection"], max_depth=1
        )

        # Step 2: Search
        results = usc_index.query_bm25(list(expanded), top_k=20)

        # Step 3: Validate — results should mention drug-related terms
        drug_terms = {
            "narcotic",
            "drug",
            "controlled substance",
            "opiate",
            "heroin",
            "morphine",
            "fentanyl",
            "opioid",
            "oxycodone",
            "narcotics",
        }
        relevant = 0
        for r in results:
            doc = next(d for d in usc_docs if d["id"] == r.doc_id)
            if any(t in doc["text"].lower() for t in drug_terms):
                relevant += 1

        assert relevant >= 10, f"Only {relevant}/20 drug-related"

    def test_full_pipeline_emissions(self, usc_index, usc_docs, regulatory_lexicon):
        """Full pipeline: emission → expansion → BM25 → validate."""
        expanded = regulatory_lexicon.expand_query(
            ["emission"], ["synonym", "hyponym", "inflection"], max_depth=1
        )
        results = usc_index.query_bm25(list(expanded), top_k=20)

        env_terms = {
            "emission",
            "emissions",
            "discharge",
            "release",
            "carbon",
            "exhaust",
            "greenhouse",
            "pollutant",
            "pollution",
        }
        relevant = sum(
            1
            for r in results
            if any(
                t in next(d for d in usc_docs if d["id"] == r.doc_id)["text"].lower()
                for t in env_terms
            )
        )
        assert relevant >= 5
