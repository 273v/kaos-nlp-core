"""Tests for the Lexicon semantic knowledge graph."""

import pickle
from pathlib import Path

import pytest

from kaos_nlp_core.lexicon import (
    OPENGLOSS_FILENAME,
    Lexicon,
    default_opengloss_lexicon,
)
from kaos_nlp_core.types import RelatedTerm


@pytest.fixture
def legal_lexicon():
    """A small legal-domain lexicon for testing."""
    lex = Lexicon()

    lex.add_entry(
        {
            "word": "contract",
            "senses": [
                {
                    "part_of_speech": "noun",
                    "sense_index": 0,
                    "definition": "A legally binding agreement",
                    "synonyms": ["agreement", "pact", "covenant"],
                    "antonyms": ["breach"],
                    "hypernyms": ["legal document"],
                    "hyponyms": ["employment contract", "service contract", "lease"],
                },
                {
                    "part_of_speech": "verb",
                    "sense_index": 0,
                    "definition": "To shrink or become smaller",
                    "synonyms": ["shrink", "compress"],
                    "antonyms": ["expand"],
                    "hypernyms": ["size change"],
                    "hyponyms": [],
                },
            ],
            "edges": [
                {
                    "relationship_type": "synonym",
                    "target": "agreement",
                    "source_pos": "noun",
                    "sense_index": 0,
                },
            ],
            "all_synonyms": ["agreement", "pact", "covenant", "shrink", "compress"],
            "all_antonyms": ["breach", "expand"],
            "all_hypernyms": ["legal document", "size change"],
            "all_hyponyms": ["employment contract", "service contract", "lease"],
            "all_inflections": ["contracts", "contracted", "contracting"],
            "all_derivations": ["contractual", "contractor"],
            "all_collocations": ["sign a contract", "breach a contract"],
        }
    )

    lex.add_entry(
        {
            "word": "agreement",
            "senses": [
                {
                    "part_of_speech": "noun",
                    "sense_index": 0,
                    "definition": "A mutual understanding between parties",
                    "synonyms": ["contract", "accord", "deal"],
                    "antonyms": ["disagreement"],
                    "hypernyms": ["arrangement"],
                    "hyponyms": ["treaty", "settlement"],
                },
            ],
            "all_synonyms": ["contract", "accord", "deal"],
            "all_antonyms": ["disagreement"],
            "all_hypernyms": ["arrangement"],
            "all_hyponyms": ["treaty", "settlement"],
            "all_inflections": ["agreements"],
        }
    )

    lex.add_entry(
        {
            "word": "breach",
            "senses": [
                {
                    "part_of_speech": "noun",
                    "sense_index": 0,
                    "definition": "A failure to perform a contractual obligation",
                    "synonyms": ["violation", "infringement"],
                    "antonyms": ["compliance"],
                    "hypernyms": ["civil wrong"],
                    "hyponyms": ["material breach", "anticipatory breach"],
                },
            ],
            "all_synonyms": ["violation", "infringement", "default"],
            "all_antonyms": ["compliance", "performance"],
            "all_hypernyms": ["civil wrong"],
            "all_hyponyms": ["material breach", "anticipatory breach"],
            "all_inflections": ["breaches", "breached", "breaching"],
        }
    )

    return lex


class TestLexiconBasic:
    def test_contains(self, legal_lexicon):
        assert legal_lexicon.contains("contract")
        assert "contract" in legal_lexicon
        assert "nonexistent" not in legal_lexicon

    def test_len(self, legal_lexicon):
        assert len(legal_lexicon) == 3

    def test_synonyms(self, legal_lexicon):
        syns = legal_lexicon.synonyms("contract")
        assert "agreement" in syns
        assert "pact" in syns

    def test_antonyms(self, legal_lexicon):
        ants = legal_lexicon.antonyms("contract")
        assert "breach" in ants

    def test_hypernyms(self, legal_lexicon):
        hyps = legal_lexicon.hypernyms("contract")
        assert "legal document" in hyps

    def test_hyponyms(self, legal_lexicon):
        hypos = legal_lexicon.hyponyms("contract")
        assert "employment contract" in hypos

    def test_inflections(self, legal_lexicon):
        infl = legal_lexicon.inflections("contract")
        assert "contracts" in infl
        assert "contracted" in infl

    def test_collocations(self, legal_lexicon):
        coll = legal_lexicon.collocations("contract")
        assert "sign a contract" in coll

    def test_missing_word(self, legal_lexicon):
        assert legal_lexicon.synonyms("nonexistent") == []
        assert legal_lexicon.hypernyms("nonexistent") == []


class TestSenseAware:
    def test_noun_synonyms_only(self, legal_lexicon):
        syns = legal_lexicon.related("contract", "synonym", pos="noun", sense_index=0)
        assert "agreement" in syns
        assert "shrink" not in syns

    def test_verb_synonyms_only(self, legal_lexicon):
        syns = legal_lexicon.related("contract", "synonym", pos="verb", sense_index=0)
        assert "shrink" in syns
        assert "agreement" not in syns

    def test_get_senses(self, legal_lexicon):
        senses = legal_lexicon.get_senses("contract")
        assert len(senses) == 2
        assert senses[0].part_of_speech == "noun"
        assert senses[1].part_of_speech == "verb"
        assert "agreement" in senses[0].synonyms
        assert "shrink" in senses[1].synonyms


class TestQueryExpansion:
    def test_synonym_expansion(self, legal_lexicon):
        expanded = set(legal_lexicon.expand_query(["contract"], ["synonym"]))
        assert "contract" in expanded  # original term preserved
        assert "agreement" in expanded
        assert "pact" in expanded

    def test_depth_2_expansion(self, legal_lexicon):
        expanded = set(legal_lexicon.expand_query(["contract"], ["synonym"], max_depth=2))
        # contract -> agreement (depth 1) -> accord (depth 2)
        assert "accord" in expanded
        assert "deal" in expanded

    def test_multi_term_expansion(self, legal_lexicon):
        expanded = set(legal_lexicon.expand_query(["contract", "breach"], ["synonym"]))
        assert "agreement" in expanded  # from contract
        assert "violation" in expanded  # from breach

    def test_inflection_expansion(self, legal_lexicon):
        expanded = set(legal_lexicon.expand_query(["contract"], ["inflection"]))
        assert "contracts" in expanded
        assert "contracted" in expanded
        assert "contracting" in expanded

    def test_multiple_relations(self, legal_lexicon):
        expanded = set(
            legal_lexicon.expand_query(["contract"], ["synonym", "inflection", "hyponym"])
        )
        assert "agreement" in expanded  # synonym
        assert "contracts" in expanded  # inflection
        assert "employment contract" in expanded  # hyponym

    def test_sense_aware_expansion(self, legal_lexicon):
        # Only expand legal noun sense
        expanded = set(
            legal_lexicon.expand_query_sense_aware(
                [("contract", "noun", 0)],
                ["synonym", "hyponym"],
            )
        )
        assert "agreement" in expanded
        assert "employment contract" in expanded
        assert "shrink" not in expanded  # verb sense excluded

    def test_empty_expansion(self, legal_lexicon):
        expanded = legal_lexicon.expand_query(["nonexistent"], ["synonym"])
        assert set(expanded) == {"nonexistent"}


class TestDynamicLoading:
    def test_add_entries_incrementally(self):
        lex = Lexicon()
        assert len(lex) == 0

        lex.add_entry({"word": "test", "all_synonyms": ["exam", "quiz"]})
        assert len(lex) == 1
        assert lex.synonyms("test") == ["exam", "quiz"]

    def test_save_load(self, tmp_path: Path):
        lex = Lexicon()
        lex.add_entry({"word": "contract", "all_synonyms": ["agreement"]})
        path = tmp_path / "lexicon.bin"
        lex.save(str(path))
        restored = Lexicon.load(str(path))
        assert restored.synonyms("contract") == ["agreement"]

        lex.add_entry({"word": "exam", "all_synonyms": ["test", "assessment"]})
        assert len(lex) == 2

    def test_add_entries_batch(self):
        lex = Lexicon()
        lex.add_entries(
            [
                {"word": "alpha", "all_synonyms": ["first"]},
                {"word": "beta", "all_synonyms": ["second"]},
                {"word": "gamma", "all_synonyms": ["third"]},
            ]
        )
        assert len(lex) == 3
        assert lex.synonyms("alpha") == ["first"]

    def test_replace_entry(self):
        lex = Lexicon()
        lex.add_entry({"word": "test", "all_synonyms": ["old"]})
        assert lex.synonyms("test") == ["old"]

        lex.add_entry({"word": "test", "all_synonyms": ["new"]})
        assert lex.synonyms("test") == ["new"]
        assert len(lex) == 1

    def test_minimal_entry(self):
        """An entry with only 'word' should work — all lists default empty."""
        lex = Lexicon()
        lex.add_entry({"word": "minimal"})
        assert "minimal" in lex
        assert lex.synonyms("minimal") == []


class TestPickle:
    def test_pickle_roundtrip(self, legal_lexicon):
        data = pickle.dumps(legal_lexicon)
        lex2 = pickle.loads(data)
        assert len(lex2) == 3
        assert lex2.synonyms("contract") == legal_lexicon.synonyms("contract")
        assert lex2.hypernyms("breach") == legal_lexicon.hypernyms("breach")

    def test_pickle_preserves_senses(self, legal_lexicon):
        data = pickle.dumps(legal_lexicon)
        lex2 = pickle.loads(data)
        senses = lex2.get_senses("contract")
        assert len(senses) == 2
        assert senses[0].synonyms == ["agreement", "pact", "covenant"]


class TestRelatedTyped:
    def test_returns_related_term_records(self, legal_lexicon):
        terms = legal_lexicon.related_typed("contract", "synonym")
        assert all(isinstance(t, RelatedTerm) for t in terms)
        texts = [t.text for t in terms]
        assert "agreement" in texts

    def test_relation_field_set(self, legal_lexicon):
        terms = legal_lexicon.related_typed("contract", "hypernym")
        for t in terms:
            assert t.relation == "hypernym"
            assert t.pos is None
            assert t.sense_index is None

    def test_sense_filter_propagated(self, legal_lexicon):
        terms = legal_lexicon.related_typed("contract", "synonym", pos="noun", sense_index=0)
        assert all(t.pos == "noun" and t.sense_index == 0 for t in terms)
        texts = [t.text for t in terms]
        assert "agreement" in texts
        # The verb-sense synonym shouldn't appear in the noun-sense filter.
        assert "shrink" not in texts

    def test_unknown_word_empty(self, legal_lexicon):
        assert legal_lexicon.related_typed("xyzzyzy", "synonym") == []


class TestDefaultLexicon:
    """The bundled OpenGloss lexicon must be loadable via the default helper."""

    def test_loads_from_in_repo_data(self):
        data_path = Path(__file__).resolve().parent.parent / "data" / OPENGLOSS_FILENAME
        if not data_path.exists():
            pytest.skip("OpenGloss data file not present in repo")
        lex = default_opengloss_lexicon()
        assert len(lex) > 100_000
        assert lex.contains("contract")
        # Cross-domain coverage check.
        assert lex.contains("diagnosis")
        assert lex.contains("hypothesis")

    def test_cached(self):
        data_path = Path(__file__).resolve().parent.parent / "data" / OPENGLOSS_FILENAME
        if not data_path.exists():
            pytest.skip("OpenGloss data file not present in repo")
        lex1 = default_opengloss_lexicon()
        lex2 = default_opengloss_lexicon()
        assert lex1 is lex2

    def test_load_default_static_method(self):
        data_path = Path(__file__).resolve().parent.parent / "data" / OPENGLOSS_FILENAME
        if not data_path.exists():
            pytest.skip("OpenGloss data file not present in repo")
        lex = Lexicon.load_default()
        assert len(lex) > 100_000

    def test_missing_lexicon_helpful_error(self, monkeypatch, tmp_path):
        # Force the loader to look at empty paths only — AND mask the
        # embedded-bytes path on _RustLexicon so the filesystem fallback
        # is the only route. After the wheel-vendored embedding landed,
        # `_RustLexicon.default_embedded()` is the default success path
        # in published wheels; this test exercises the friendly error
        # message that fires when both the embedded copy and every
        # filesystem path miss.
        from kaos_nlp_core import lexicon as lex_module
        from kaos_nlp_core._rust.lexicon import Lexicon as _RustLexicon

        monkeypatch.setattr(lex_module, "_DEFAULT_LEXICON", None)
        monkeypatch.setattr(
            lex_module,
            "_DEFAULT_LEXICON_SEARCH_PATHS",
            (tmp_path / "missing.bin",),
        )
        monkeypatch.delenv("KAOS_NLP_LEXICON_PATH", raising=False)

        def _no_embedded() -> None:
            raise ValueError("embedded path masked for test")

        monkeypatch.setattr(_RustLexicon, "default_embedded", _no_embedded)

        with pytest.raises(FileNotFoundError) as exc_info:
            default_opengloss_lexicon()
        msg = str(exc_info.value)
        assert "OpenGloss" in msg
        assert "build_opengloss_lexicon.py" in msg
        assert "default_english_wordset" in msg

    def test_env_override_takes_precedence(self, monkeypatch, tmp_path):
        from kaos_nlp_core import lexicon as lex_module

        # Build a tiny lexicon and save it
        custom = Lexicon()
        custom.add_entry({"word": "envcustom", "all_synonyms": ["overridden"]})
        path = tmp_path / "custom.bin"
        custom.save(str(path))

        monkeypatch.setattr(lex_module, "_DEFAULT_LEXICON", None)
        monkeypatch.setenv("KAOS_NLP_LEXICON_PATH", str(path))
        try:
            lex = default_opengloss_lexicon()
            assert lex.contains("envcustom")
            assert lex.synonyms("envcustom") == ["overridden"]
        finally:
            # Clear cache so other tests get the real default lexicon back.
            monkeypatch.setattr(lex_module, "_DEFAULT_LEXICON", None)
