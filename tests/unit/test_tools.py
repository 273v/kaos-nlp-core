"""Tests for kaos-nlp-core MCP tools.

Tests tool metadata validity and execution for all 11 NLP tools.
These tests call the tool execute() methods directly without a full
MCP server — they exercise the Python NLP layer, not kaos-core plumbing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kaos_nlp_core.tools import _COMPARE_ALGORITHMS, register_nlp_tools

# ---------------------------------------------------------------------------
# Helpers: instantiate tools without kaos-core runtime
# ---------------------------------------------------------------------------

# We import the tools module and build the tool classes by calling
# register_nlp_tools with a mock runtime that just captures registrations.


class _MockToolsRegistry:
    def __init__(self) -> None:
        self.tools: list = []

    def register_tool(self, tool: object, aliases: list[str] | None = None) -> None:
        del aliases
        self.tools.append(tool)


class _MockRuntime:
    def __init__(self) -> None:
        self.tools = _MockToolsRegistry()


def _build_tools() -> dict:
    """Build and return all tool instances keyed by name."""
    rt = _MockRuntime()
    count = register_nlp_tools(rt)
    assert count == 17
    return {t.metadata.name: t for t in rt.tools.tools}


TOOLS = _build_tools()

# MCP tool name pattern from kaos-core
TOOL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+){2,}$")


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestToolMetadata:
    """Validate metadata for all 17 tools."""

    def test_tool_count(self) -> None:
        assert len(TOOLS) == 17

    @pytest.mark.parametrize("name", list(TOOLS.keys()))
    def test_name_pattern(self, name: str) -> None:
        assert TOOL_NAME_PATTERN.match(name), f"Bad tool name: {name}"

    @pytest.mark.parametrize("name", list(TOOLS.keys()))
    def test_annotations_set(self, name: str) -> None:
        tool = TOOLS[name]
        meta = tool.metadata
        assert meta.annotations is not None, f"{name}: annotations must not be None"

    @pytest.mark.parametrize("name", list(TOOLS.keys()))
    def test_module_and_version(self, name: str) -> None:
        from kaos_nlp_core.tools import _VERSION

        meta = TOOLS[name].metadata
        assert meta.module_name == "kaos-nlp-core"
        # Track tools.py:_VERSION rather than hardcoding so bumping the
        # release version doesn't require a test edit.
        assert meta.version == _VERSION

    @pytest.mark.parametrize("name", list(TOOLS.keys()))
    def test_description_nonempty(self, name: str) -> None:
        meta = TOOLS[name].metadata
        assert len(meta.description) > 10

    @pytest.mark.parametrize("name", list(TOOLS.keys()))
    def test_has_input_schema(self, name: str) -> None:
        meta = TOOLS[name].metadata
        schema = meta.get_input_json_schema()
        assert "properties" in schema

    def test_readonly_annotations(self) -> None:
        """All tools except build-index should be read-only."""
        for name, tool in TOOLS.items():
            ann = tool.metadata.annotations
            if name == "kaos-nlp-build-index":
                assert ann.readOnlyHint is False
            else:
                assert ann.readOnlyHint is True, f"{name} should be read-only"

    def test_expected_names(self) -> None:
        expected = {
            "kaos-nlp-tokenize",
            "kaos-nlp-segment-sentences",
            "kaos-nlp-segment-paragraphs",
            "kaos-nlp-compare",
            "kaos-nlp-find-pattern",
            "kaos-nlp-search-text",
            "kaos-nlp-build-index",
            "kaos-nlp-hash",
            "kaos-nlp-find-duplicates",
            "kaos-nlp-analyze-text",
            "kaos-nlp-score-quality",
            "kaos-nlp-lexicon-related",
            "kaos-nlp-lexicon-expand-query",
            "kaos-nlp-token-frequency",
            "kaos-nlp-extract-concepts",
            "kaos-nlp-label-lines",
            "kaos-nlp-outline",
        }
        assert set(TOOLS.keys()) == expected


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTokenizeTool:
    async def test_simple(self) -> None:
        tool = TOOLS["kaos-nlp-tokenize"]
        result = await tool.execute({"text": "Hello world"})
        assert not result.isError
        data = result.require_structured()
        assert data["count"] == 2
        tokens = data["tokens"]
        assert tokens[0]["text"] == "Hello"
        assert tokens[1]["text"] == "world"

    async def test_multibyte(self) -> None:
        tool = TOOLS["kaos-nlp-tokenize"]
        text = "cafe\u0301 Tokyo\u3000test"
        result = await tool.execute({"text": text})
        assert not result.isError
        data = result.require_structured()
        assert data["count"] >= 2
        # Verify offsets round-trip
        for t in data["tokens"]:
            assert text[t["start"] : t["end"]] == t["text"] or t["text"] in text

    async def test_lowercase(self) -> None:
        tool = TOOLS["kaos-nlp-tokenize"]
        result = await tool.execute({"text": "Hello World", "lowercase": True})
        data = result.require_structured()
        assert data["tokens"][0]["text"] == "hello"

    async def test_empty_text(self) -> None:
        tool = TOOLS["kaos-nlp-tokenize"]
        result = await tool.execute({"text": ""})
        assert result.isError


@pytest.mark.asyncio
class TestSegmentSentencesTool:
    async def test_multi_sentence(self) -> None:
        tool = TOOLS["kaos-nlp-segment-sentences"]
        text = "The court ruled in favor. The defendant appealed. The case was dismissed."
        result = await tool.execute({"text": text})
        assert not result.isError
        data = result.require_structured()
        assert data["count"] >= 2  # At least 2 sentences

    async def test_empty_text(self) -> None:
        tool = TOOLS["kaos-nlp-segment-sentences"]
        result = await tool.execute({"text": ""})
        assert result.isError


@pytest.mark.asyncio
class TestSegmentParagraphsTool:
    async def test_multi_paragraph(self) -> None:
        tool = TOOLS["kaos-nlp-segment-paragraphs"]
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."
        result = await tool.execute({"text": text})
        assert not result.isError
        data = result.require_structured()
        assert data["count"] >= 2

    async def test_single_paragraph(self) -> None:
        tool = TOOLS["kaos-nlp-segment-paragraphs"]
        text = "Just one paragraph with no blank lines."
        result = await tool.execute({"text": text})
        assert not result.isError
        data = result.require_structured()
        assert data["count"] >= 1


@pytest.mark.asyncio
class TestCompareTool:
    async def test_identical(self) -> None:
        tool = TOOLS["kaos-nlp-compare"]
        result = await tool.execute({"text1": "hello", "text2": "hello"})
        assert not result.isError
        data = result.require_structured()
        assert data["similarity"] == pytest.approx(1.0)
        assert data["distance"] == pytest.approx(0.0)

    async def test_different(self) -> None:
        tool = TOOLS["kaos-nlp-compare"]
        result = await tool.execute(
            {"text1": "kitten", "text2": "sitting", "algorithm": "levenshtein"}
        )
        assert not result.isError
        data = result.require_structured()
        assert data["distance"] > 0
        assert data["algorithm"] == "levenshtein"

    async def test_jaro_winkler_default(self) -> None:
        tool = TOOLS["kaos-nlp-compare"]
        result = await tool.execute({"text1": "MARTHA", "text2": "MARHTA"})
        assert not result.isError
        data = result.require_structured()
        assert data["algorithm"] == "jaro_winkler"
        assert data["similarity"] > 0.9  # Very similar strings

    async def test_soundex(self) -> None:
        tool = TOOLS["kaos-nlp-compare"]
        result = await tool.execute({"text1": "Robert", "text2": "Rupert", "algorithm": "soundex"})
        assert not result.isError

    async def test_invalid_algorithm(self) -> None:
        tool = TOOLS["kaos-nlp-compare"]
        result = await tool.execute({"text1": "a", "text2": "b", "algorithm": "nonexistent"})
        assert result.isError

    async def test_missing_text(self) -> None:
        tool = TOOLS["kaos-nlp-compare"]
        result = await tool.execute({"text1": "hello"})
        assert result.isError

    @pytest.mark.parametrize("algo", _COMPARE_ALGORITHMS)
    async def test_all_algorithms(self, algo: str) -> None:
        tool = TOOLS["kaos-nlp-compare"]
        result = await tool.execute({"text1": "hello", "text2": "world", "algorithm": algo})
        assert not result.isError, f"{algo} failed: {result.text}"
        data = result.require_structured()
        assert "similarity" in data
        assert "distance" in data


@pytest.mark.asyncio
class TestFindPatternTool:
    async def test_substring(self) -> None:
        tool = TOOLS["kaos-nlp-find-pattern"]
        result = await tool.execute({"text": "the cat sat on the mat", "pattern": "the"})
        assert not result.isError
        data = result.require_structured()
        assert data["count"] == 2

    async def test_case_insensitive(self) -> None:
        tool = TOOLS["kaos-nlp-find-pattern"]
        result = await tool.execute(
            {
                "text": "Hello HELLO hello",
                "pattern": "hello",
                "case_insensitive": True,
            }
        )
        assert not result.isError
        data = result.require_structured()
        assert data["count"] == 3

    async def test_regex(self) -> None:
        tool = TOOLS["kaos-nlp-find-pattern"]
        result = await tool.execute({"text": "abc 123 def 456", "pattern": r"\d+", "mode": "regex"})
        assert not result.isError
        data = result.require_structured()
        assert data["count"] == 2
        assert data["matches"][0]["text"] == "123"
        assert data["matches"][1]["text"] == "456"

    async def test_no_match(self) -> None:
        tool = TOOLS["kaos-nlp-find-pattern"]
        result = await tool.execute({"text": "hello world", "pattern": "xyz"})
        assert not result.isError
        data = result.require_structured()
        assert data["count"] == 0

    async def test_invalid_regex(self) -> None:
        tool = TOOLS["kaos-nlp-find-pattern"]
        result = await tool.execute({"text": "hello", "pattern": "[invalid", "mode": "regex"})
        assert result.isError

    async def test_missing_params(self) -> None:
        tool = TOOLS["kaos-nlp-find-pattern"]
        result = await tool.execute({"text": "hello"})
        assert result.isError


@pytest.mark.asyncio
class TestSearchTextTool:
    async def test_sentence_search(self) -> None:
        tool = TOOLS["kaos-nlp-search-text"]
        text = (
            "The court ruled that the contract was valid. "
            "The defendant filed an appeal. "
            "The plaintiff sought damages for breach of contract. "
            "The jury reached a verdict after deliberation."
        )
        result = await tool.execute({"text": text, "query": "contract valid"})
        assert not result.isError
        data = result.require_structured()
        assert data["total"] > 0
        assert data["results"][0]["score"] > 0

    async def test_paragraph_search(self) -> None:
        tool = TOOLS["kaos-nlp-search-text"]
        text = (
            "First paragraph about contracts and agreements.\n\n"
            "Second paragraph about intellectual property rights.\n\n"
            "Third paragraph about employment law and regulations."
        )
        result = await tool.execute({"text": text, "query": "contracts", "level": "paragraphs"})
        assert not result.isError
        data = result.require_structured()
        assert data["total"] > 0

    async def test_top_k(self) -> None:
        tool = TOOLS["kaos-nlp-search-text"]
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        result = await tool.execute({"text": text, "query": "sentence", "top_k": 2})
        assert not result.isError
        data = result.require_structured()
        assert data["total"] <= 2


@pytest.mark.asyncio
class TestBuildIndexTool:
    """F3 confines build-index I/O to KAOS_NLP_WORKSPACE_ROOT (default CWD).

    `tmp_path` lives under `/tmp`, outside the test runner's CWD, so each
    test that touches `tmp_path` widens the workspace via monkeypatch.
    """

    async def test_build_from_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_NLP_WORKSPACE_ROOT", str(tmp_path))
        tool = TOOLS["kaos-nlp-build-index"]
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            "The quick brown fox jumps over the lazy dog.\n"
            "A stitch in time saves nine.\n"
            "All that glitters is not gold.\n",
            encoding="utf-8",
        )
        output = tmp_path / "test.kncidx"

        result = await tool.execute({"corpus_path": str(corpus), "output_path": str(output)})
        assert not result.isError, result.text
        data = result.require_structured()
        assert data["doc_count"] == 3
        assert data["term_count"] > 0
        assert output.exists()

    async def test_missing_file(self) -> None:
        tool = TOOLS["kaos-nlp-build-index"]
        result = await tool.execute({"corpus_path": "/nonexistent/file.txt"})
        assert result.isError

    async def test_empty_corpus(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_NLP_WORKSPACE_ROOT", str(tmp_path))
        tool = TOOLS["kaos-nlp-build-index"]
        corpus = tmp_path / "empty.txt"
        corpus.write_text("", encoding="utf-8")
        result = await tool.execute({"corpus_path": str(corpus)})
        assert result.isError  # Empty file should error

    async def test_default_output_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAOS_NLP_WORKSPACE_ROOT", str(tmp_path))
        tool = TOOLS["kaos-nlp-build-index"]
        corpus = tmp_path / "data.txt"
        corpus.write_text("document one\ndocument two\n", encoding="utf-8")
        result = await tool.execute({"corpus_path": str(corpus)})
        assert not result.isError
        data = result.require_structured()
        # Default output should be corpus_path with .kncidx extension
        assert data["index_path"].endswith(".kncidx")

    async def test_corpus_outside_root_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A corpus path outside KAOS_NLP_WORKSPACE_ROOT must be rejected."""
        # Workspace = a subdir; corpus lives one level above it.
        sub = tmp_path / "ws"
        sub.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("hello\n", encoding="utf-8")

        monkeypatch.setenv("KAOS_NLP_WORKSPACE_ROOT", str(sub))
        tool = TOOLS["kaos-nlp-build-index"]
        result = await tool.execute({"corpus_path": str(outside)})
        assert result.isError
        assert "outside the workspace root" in result.text


@pytest.mark.asyncio
class TestHashTool:
    async def test_ctph(self) -> None:
        tool = TOOLS["kaos-nlp-hash"]
        result = await tool.execute({"text": "Hello world " * 100})
        assert not result.isError
        data = result.require_structured()
        assert data["algorithm"] == "ctph"
        assert "hash" in data
        assert isinstance(data["hash"], str)

    async def test_minhash(self) -> None:
        tool = TOOLS["kaos-nlp-hash"]
        result = await tool.execute({"text": "Hello world, this is a test", "algorithm": "minhash"})
        assert not result.isError
        data = result.require_structured()
        assert data["algorithm"] == "minhash"
        assert "signature" in data
        assert data["num_permutations"] == 128

    async def test_empty_text(self) -> None:
        tool = TOOLS["kaos-nlp-hash"]
        result = await tool.execute({"text": ""})
        assert result.isError


@pytest.mark.asyncio
class TestFindDuplicatesTool:
    async def test_known_duplicates(self) -> None:
        tool = TOOLS["kaos-nlp-find-duplicates"]
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox leaps over the lazy dog",
            "Completely different text about something else entirely",
        ]
        result = await tool.execute({"texts": texts, "threshold": 0.3})
        assert not result.isError
        data = result.require_structured()
        assert "groups" in data
        assert "total_groups" in data

    async def test_all_unique(self) -> None:
        tool = TOOLS["kaos-nlp-find-duplicates"]
        texts = [
            "Alpha bravo charlie delta echo foxtrot",
            "Xray yankee zulu one two three four",
            "Completely different text number three here",
        ]
        result = await tool.execute({"texts": texts, "threshold": 0.9})
        assert not result.isError
        data = result.require_structured()
        assert data["total_groups"] == 0

    async def test_too_few_texts(self) -> None:
        tool = TOOLS["kaos-nlp-find-duplicates"]
        result = await tool.execute({"texts": ["only one"]})
        assert result.isError

    async def test_empty_list(self) -> None:
        tool = TOOLS["kaos-nlp-find-duplicates"]
        result = await tool.execute({"texts": []})
        assert result.isError


@pytest.mark.asyncio
class TestAnalyzeTextTool:
    async def test_basic_analysis(self) -> None:
        tool = TOOLS["kaos-nlp-analyze-text"]
        text = (
            "The court ruled that the contract was valid. "
            "The defendant filed an appeal immediately."
        )
        result = await tool.execute({"text": text})
        assert not result.isError
        data = result.require_structured()

        assert data["characters"] == len(text)
        assert data["tokens"] > 0
        assert data["unique_terms"] > 0
        assert data["sentences"] >= 1
        assert data["paragraphs"] >= 1
        assert 0.0 < data["type_token_ratio"] <= 1.0
        assert data["avg_sentence_length"] > 0
        assert len(data["top_terms"]) > 0

    async def test_multiline_text(self) -> None:
        tool = TOOLS["kaos-nlp-analyze-text"]
        text = (
            "First paragraph sentence one. Sentence two.\n\n"
            "Second paragraph. Another sentence here.\n\n"
            "Third paragraph with final content."
        )
        result = await tool.execute({"text": text})
        assert not result.isError
        data = result.require_structured()
        assert data["paragraphs"] >= 2
        assert data["sentences"] >= 3

    async def test_empty_text(self) -> None:
        tool = TOOLS["kaos-nlp-analyze-text"]
        result = await tool.execute({"text": ""})
        assert result.isError

    async def test_top_terms_format(self) -> None:
        tool = TOOLS["kaos-nlp-analyze-text"]
        result = await tool.execute({"text": "the the the cat cat dog"})
        assert not result.isError
        data = result.require_structured()
        for entry in data["top_terms"]:
            assert "term" in entry
            assert "count" in entry
            assert isinstance(entry["count"], int)


# ---------------------------------------------------------------------------
# ScoreQualityTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScoreQualityTool:
    async def test_legal_text(self) -> None:
        tool = TOOLS["kaos-nlp-score-quality"]
        # Reasonably well-formed legal prose should score low (few deviations).
        text = (
            "Section 101. Definitions.\n\n"
            "As used in this title, the following terms have the meanings "
            "indicated. The term 'person' means an individual, partnership, "
            "corporation, association, or other legal entity. The term "
            "'State' includes each of the several States, the District of "
            "Columbia, and the Commonwealth of Puerto Rico.\n\n"
            "Section 102. Application.\n\n"
            "This title applies to all persons engaged in commerce or in "
            "any activity affecting commerce."
        )
        result = await tool.execute({"text": text, "domain": "legal"})
        assert not result.isError
        data = result.require_structured()
        assert "score" in data
        assert isinstance(data["score"], float)
        assert data["score"] >= 0.0
        assert data["domain"] == "legal"
        assert "metrics" in data
        assert "deviations" in data

    async def test_general_domain(self) -> None:
        tool = TOOLS["kaos-nlp-score-quality"]
        text = "Hello world. This is a simple sentence."
        result = await tool.execute({"text": text, "domain": "general"})
        assert not result.isError
        data = result.require_structured()
        assert data["domain"] == "general"

    async def test_default_domain_is_general(self) -> None:
        tool = TOOLS["kaos-nlp-score-quality"]
        text = "The quick brown fox jumps over the lazy dog."
        result = await tool.execute({"text": text})
        assert not result.isError
        data = result.require_structured()
        assert data["domain"] == "general"

    async def test_empty_text_error(self) -> None:
        tool = TOOLS["kaos-nlp-score-quality"]
        result = await tool.execute({"text": ""})
        assert result.isError

    async def test_invalid_domain_error(self) -> None:
        tool = TOOLS["kaos-nlp-score-quality"]
        result = await tool.execute({"text": "hello", "domain": "martian"})
        assert result.isError

    async def test_metrics_keys(self) -> None:
        tool = TOOLS["kaos-nlp-score-quality"]
        text = "Some reasonable text with several words and sentences."
        result = await tool.execute({"text": text})
        assert not result.isError
        metrics = result.require_structured()["metrics"]
        expected_keys = {
            "total_characters",
            "ratio_whitespace",
            "average_line_length",
            "average_paragraph_length",
            "ratio_alphanumeric",
            "ratio_alpha_to_numeric",
            "ratio_non_ascii",
            "ratio_capital",
            "ratio_punctuation",
            "ratio_symbol",
            "average_word_length",
            "type_token_ratio",
            "token_entropy",
            "char_entropy",
            "max_token_frequency_ratio",
            "repetition_rate",
            "ratio_format_tokens",
            "ratio_in_lexicon",
            "num_words",
            "num_lines",
            "num_paragraphs",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    async def test_use_lexicon_false_omits_metric(self) -> None:
        tool = TOOLS["kaos-nlp-score-quality"]
        text = "The quick brown fox jumps over the lazy dog."
        result = await tool.execute({"text": text, "use_lexicon": False})
        assert not result.isError
        metrics = result.require_structured()["metrics"]
        assert "ratio_in_lexicon" not in metrics

    async def test_garbled_text_lexicon_signal(self) -> None:
        tool = TOOLS["kaos-nlp-score-quality"]
        clean = "The quick brown fox jumps over the lazy dog. " * 4
        garbled = "Tlie qiiick browii rox jLnnps oxer tlie iazy dog. " * 4
        clean_res = await tool.execute({"text": clean})
        garbled_res = await tool.execute({"text": garbled})
        clean_lex = clean_res.require_structured()["metrics"]["ratio_in_lexicon"]
        garbled_lex = garbled_res.require_structured()["metrics"]["ratio_in_lexicon"]
        assert clean_lex > garbled_lex
        assert garbled_res.require_structured()["score"] > clean_res.require_structured()["score"]

    async def test_garbage_text_scores_high(self) -> None:
        tool = TOOLS["kaos-nlp-score-quality"]
        # Repetitive nonsense should score higher (worse) than normal text.
        garbage = "AAAA " * 200
        normal = (
            "The Securities and Exchange Commission today announced "
            "charges against three individuals for insider trading. "
            "The complaint alleges that defendants obtained material "
            "nonpublic information about pending mergers."
        )
        r_garbage = await tool.execute({"text": garbage})
        r_normal = await tool.execute({"text": normal})
        assert not r_garbage.isError
        assert not r_normal.isError
        s_garbage = r_garbage.require_structured()["score"]
        s_normal = r_normal.require_structured()["score"]
        assert s_garbage > s_normal


@pytest.mark.asyncio
class TestLexiconRelatedTool:
    async def test_synonyms_default(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-related"]
        result = await tool.execute({"word": "contract"})
        assert not result.isError
        data = result.require_structured()
        assert data["word"] == "contract"
        assert data["relation"] == "synonym"
        assert data["count"] > 0
        texts = [t["text"] for t in data["related"]]
        assert "agreement" in texts

    async def test_relation_enum(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-related"]
        for rel in ("synonym", "hypernym", "hyponym", "inflection"):
            result = await tool.execute({"word": "agreement", "relation": rel})
            assert not result.isError, f"relation={rel} failed: {result.content}"
            data = result.require_structured()
            assert data["relation"] == rel

    async def test_max_results_truncates(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-related"]
        result = await tool.execute({"word": "contract", "relation": "synonym", "max_results": 3})
        data = result.require_structured()
        assert data["count"] <= 3
        if data["total_matches"] > 3:
            assert data["has_more"] is True

    async def test_sense_filter(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-related"]
        result = await tool.execute(
            {
                "word": "contract",
                "relation": "synonym",
                "pos": "noun",
                "sense_index": 0,
            }
        )
        assert not result.isError
        data = result.require_structured()
        for t in data["related"]:
            assert t["pos"] == "noun"
            assert t["sense_index"] == 0
        # Verb-sense synonyms should not leak into the noun-only filter.
        texts = [t["text"] for t in data["related"]]
        assert "shrink" not in texts

    async def test_uppercase_fallback(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-related"]
        result = await tool.execute({"word": "Contract", "relation": "synonym"})
        assert not result.isError
        # Tool reports the lowercased word it actually matched.
        assert result.require_structured()["word"] == "contract"

    async def test_empty_word_error(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-related"]
        result = await tool.execute({"word": ""})
        assert result.isError

    async def test_unknown_word_error(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-related"]
        result = await tool.execute({"word": "xyzzyzzyzy"})
        assert result.isError


@pytest.mark.asyncio
class TestLexiconExpandQueryTool:
    async def test_default_relations(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-expand-query"]
        result = await tool.execute({"terms": ["contract", "termination"]})
        assert not result.isError
        data = result.require_structured()
        assert data["original_count"] == 2
        assert data["expanded_count"] > data["original_count"]
        assert data["expansion_factor"] > 1.0
        assert "agreement" in data["expanded_terms"]

    async def test_custom_relations(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-expand-query"]
        result = await tool.execute(
            {"terms": ["contract"], "relations": ["hypernym"], "max_depth": 1}
        )
        assert not result.isError
        data = result.require_structured()
        assert data["relations"] == ["hypernym"]
        # Hypernyms of "contract" should at minimum include broader categories.
        assert data["expanded_count"] > 1

    async def test_max_depth_increases_expansion(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-expand-query"]
        d1 = await tool.execute({"terms": ["contract"], "max_depth": 1})
        d2 = await tool.execute({"terms": ["contract"], "max_depth": 2})
        n1 = d1.require_structured()["expanded_count"]
        n2 = d2.require_structured()["expanded_count"]
        assert n2 >= n1

    async def test_empty_terms_error(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-expand-query"]
        result = await tool.execute({"terms": []})
        assert result.isError

    async def test_invalid_relation_error(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-expand-query"]
        result = await tool.execute({"terms": ["contract"], "relations": ["bogus_relation"]})
        assert result.isError

    async def test_invalid_depth_error(self) -> None:
        tool = TOOLS["kaos-nlp-lexicon-expand-query"]
        result = await tool.execute({"terms": ["contract"], "max_depth": 99})
        assert result.isError


@pytest.mark.asyncio
class TestTokenFrequencyTool:
    async def test_default_no_lexicon(self) -> None:
        tool = TOOLS["kaos-nlp-token-frequency"]
        result = await tool.execute({"text": "the the cat dog the", "top_k": 5})
        assert not result.isError
        data = result.require_structured()
        assert data["lexicon_mode"] == "none"
        assert data["coverage"] == 1.0
        assert data["unique_terms"] == 3
        assert data["terms"][0]["text"] == "the"
        assert data["terms"][0]["count"] == 3
        assert data["terms"][0]["share"] == pytest.approx(0.6, abs=1e-3)

    async def test_english_lexicon_filter(self) -> None:
        tool = TOOLS["kaos-nlp-token-frequency"]
        # 'xyzzy' is not in any English wordset.
        result = await tool.execute(
            {
                "text": "the cat xyzzy ate the contract",
                "lexicon": "english",
                "top_k": 10,
            }
        )
        assert not result.isError
        data = result.require_structured()
        assert data["lexicon_mode"] == "english"
        terms = {t["text"] for t in data["terms"]}
        assert "xyzzy" not in terms
        assert data["coverage"] < 1.0

    async def test_top_k_truncates(self) -> None:
        tool = TOOLS["kaos-nlp-token-frequency"]
        result = await tool.execute({"text": "a b c d e f g h i j k", "top_k": 3})
        data = result.require_structured()
        assert len(data["terms"]) == 3

    async def test_min_count_drops(self) -> None:
        tool = TOOLS["kaos-nlp-token-frequency"]
        result = await tool.execute({"text": "the the cat", "min_count": 2})
        data = result.require_structured()
        terms = {t["text"] for t in data["terms"]}
        assert "the" in terms
        assert "cat" not in terms

    async def test_empty_text_error(self) -> None:
        tool = TOOLS["kaos-nlp-token-frequency"]
        result = await tool.execute({"text": ""})
        assert result.isError

    async def test_unknown_lexicon_error(self) -> None:
        tool = TOOLS["kaos-nlp-token-frequency"]
        result = await tool.execute({"text": "hello", "lexicon": "klingon"})
        assert result.isError


@pytest.mark.asyncio
class TestExtractConceptsTool:
    async def test_hypernym_default(self) -> None:
        tool = TOOLS["kaos-nlp-extract-concepts"]
        text = (
            "WHEREAS the Plaintiff filed a complaint and summons against the "
            "Defendant alleging breach of contract."
        )
        result = await tool.execute({"text": text, "top_k": 10})
        assert not result.isError
        data = result.require_structured()
        assert data["direction"] == "hypernym"
        # Calibrated stop-list keeps abstract roots out; legal concepts win.
        terms = {c["term"] for c in data["concepts"]}
        legal = {"legal action", "legal document", "legal procedure", "litigant"}
        assert legal & terms, f"expected one of {legal} in {sorted(terms)}"

    async def test_hyponym_direction(self) -> None:
        tool = TOOLS["kaos-nlp-extract-concepts"]
        result = await tool.execute(
            {
                "text": "The court ruled. Both parties agreed.",
                "direction": "hyponym",
                "top_k": 5,
            }
        )
        assert not result.isError
        data = result.require_structured()
        assert data["direction"] == "hyponym"

    async def test_both_direction(self) -> None:
        tool = TOOLS["kaos-nlp-extract-concepts"]
        text = "WHEREAS the Plaintiff filed a complaint and the Defendant received a summons."
        result = await tool.execute({"text": text, "direction": "both", "top_k": 5})
        assert not result.isError
        data = result.require_structured()
        directions = {c["direction"] for c in data["concepts"]}
        assert "hypernym" in directions

    async def test_extra_stop_terms(self) -> None:
        tool = TOOLS["kaos-nlp-extract-concepts"]
        text = "The Plaintiff filed a complaint and summons against the Defendant."
        a = await tool.execute({"text": text, "top_k": 10})
        b = await tool.execute(
            {
                "text": text,
                "top_k": 10,
                "extra_stop_terms": ["legal action", "legal document"],
            }
        )
        b_terms = {c["term"] for c in b.require_structured()["concepts"]}
        # The extras should remove those exact terms.
        assert "legal action" not in b_terms
        assert "legal document" not in b_terms
        # `a` is intentionally executed but not asserted on — sanity that
        # both calls succeed under a single fixture path.
        assert not a.isError

    async def test_invalid_direction_error(self) -> None:
        tool = TOOLS["kaos-nlp-extract-concepts"]
        result = await tool.execute({"text": "hi", "direction": "sideways"})
        assert result.isError

    async def test_invalid_max_depth_error(self) -> None:
        tool = TOOLS["kaos-nlp-extract-concepts"]
        result = await tool.execute({"text": "hi", "max_depth": 99})
        assert result.isError

    async def test_extra_stop_terms_wrong_type(self) -> None:
        tool = TOOLS["kaos-nlp-extract-concepts"]
        result = await tool.execute({"text": "hello", "extra_stop_terms": "not a list"})
        assert result.isError

    async def test_empty_text_error(self) -> None:
        tool = TOOLS["kaos-nlp-extract-concepts"]
        result = await tool.execute({"text": ""})
        assert result.isError


@pytest.mark.asyncio
class TestLabelLinesTool:
    async def test_basic_pipeline(self) -> None:
        tool = TOOLS["kaos-nlp-label-lines"]
        text = "Body text.\n\nDISCUSSION\n\nMore body content here.\n"
        result = await tool.execute({"text": text})
        assert not result.isError
        data = result.require_structured()
        assert "labels" in data
        assert "candidates" in data
        assert "label_counts" in data
        assert len(data["labels"]) == 5  # records: 5 lines
        assert "heading" in data["labels"]
        # Label counts should sum to total line count.
        assert sum(data["label_counts"].values()) == len(data["labels"])

    async def test_lexicon_kwargs_propagate(self) -> None:
        tool = TOOLS["kaos-nlp-label-lines"]
        text = "Article 5 — Définitions\n\nbody.\n"
        result = await tool.execute(
            {
                "text": text,
                "enum_lexicon": "french_legal",
                "hierarchy_lexicon": "french_legal",
            }
        )
        assert not result.isError
        data = result.require_structured()
        # The Article line should be a heading candidate with hier_level >= 1.
        assert any(c["hierarchy_level"] >= 1 for c in data["candidates"]), data["candidates"]

    async def test_unknown_lexicon_returns_error(self) -> None:
        tool = TOOLS["kaos-nlp-label-lines"]
        result = await tool.execute({"text": "foo", "heading_lexicon": "not_a_real_lex"})
        assert result.isError

    async def test_empty_text_returns_error(self) -> None:
        tool = TOOLS["kaos-nlp-label-lines"]
        result = await tool.execute({"text": ""})
        assert result.isError
