"""Tests for kaos-nlp-core CLI.

Tests cover all 11 commands with both human and --json output modes,
error handling, and edge cases.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from kaos_nlp_core.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_cli(argv: list[str]) -> str:
    """Run CLI and capture stdout."""
    stdout = StringIO()
    with patch("sys.stdout", stdout):
        main(argv)
    return stdout.getvalue()


def run_cli_json(argv: list[str]) -> dict:
    """Run CLI with --json and parse output."""
    output = run_cli(argv)
    return json.loads(output)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_text(tmp_path: Path) -> Path:
    """Simple multi-sentence text file."""
    f = tmp_path / "sample.txt"
    f.write_text(
        "Hello world. This is a test.\n"
        "Another line here.\n"
        "\n"
        "A new paragraph begins. It has two sentences.\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    """Multi-line corpus for index building."""
    f = tmp_path / "corpus.txt"
    f.write_text(
        "The cat sat on the mat.\n"
        "The dog barked at the mailman.\n"
        "A bird flew over the fence.\n"
        "The cat chased the bird away.\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def txt_dir(tmp_path: Path) -> Path:
    """Directory with text files for duplicate detection."""
    d = tmp_path / "texts"
    d.mkdir()
    (d / "doc1.txt").write_text("The quick brown fox jumps over the lazy dog.", encoding="utf-8")
    (d / "doc2.txt").write_text("The quick brown fox leaps over the lazy dog.", encoding="utf-8")
    (d / "doc3.txt").write_text(
        "Something completely different about quantum physics.", encoding="utf-8"
    )
    return d


# ---------------------------------------------------------------------------
# 1. tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_human(self, sample_text: Path) -> None:
        output = run_cli(["tokenize", str(sample_text)])
        assert "Hello" in output
        assert "world" in output

    def test_json_envelope(self, sample_text: Path) -> None:
        data = run_cli_json(["tokenize", str(sample_text), "--json"])
        assert data["command"] == "tokenize"
        assert data["file"] == "sample.txt"
        assert data["total"] > 0
        assert isinstance(data["tokens"], list)
        first = data["tokens"][0]
        assert "text" in first
        assert "start" in first
        assert "end" in first

    def test_lowercase(self, sample_text: Path) -> None:
        data = run_cli_json(["tokenize", str(sample_text), "--lowercase", "--json"])
        texts = [t["text"] for t in data["tokens"]]
        assert all(t == t.lower() for t in texts)

    def test_keep_punctuation(self, tmp_path: Path) -> None:
        f = tmp_path / "punct.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        data = run_cli_json(["tokenize", str(f), "--keep-punctuation", "--json"])
        texts = [t["text"] for t in data["tokens"]]
        assert "Hello," in texts
        assert "world!" in texts

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            run_cli(["tokenize", str(tmp_path / "missing.txt")])


# ---------------------------------------------------------------------------
# 2. segment
# ---------------------------------------------------------------------------


class TestSegment:
    def test_sentences_human(self, sample_text: Path) -> None:
        output = run_cli(["segment", str(sample_text)])
        assert "[1]" in output

    def test_sentences_json(self, sample_text: Path) -> None:
        data = run_cli_json(["segment", str(sample_text), "--json"])
        assert data["command"] == "segment"
        assert data["mode"] == "sentences"
        assert data["total"] > 0
        seg = data["segments"][0]
        assert "index" in seg
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg

    def test_lines_mode(self, sample_text: Path) -> None:
        data = run_cli_json(["segment", str(sample_text), "--mode", "lines", "--json"])
        assert data["mode"] == "lines"
        assert data["total"] > 0

    def test_paragraphs_mode(self, sample_text: Path) -> None:
        data = run_cli_json(["segment", str(sample_text), "--mode", "paragraphs", "--json"])
        assert data["mode"] == "paragraphs"
        assert data["total"] > 0

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            run_cli(["segment", str(tmp_path / "missing.txt")])


# ---------------------------------------------------------------------------
# 3. compare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_default_algorithm(self) -> None:
        output = run_cli(["compare", "kitten", "sitting"])
        assert "jaro-winkler" in output
        assert "Similarity" in output

    def test_json_envelope(self) -> None:
        data = run_cli_json(["compare", "kitten", "sitting", "--json"])
        assert data["command"] == "compare"
        assert data["text1"] == "kitten"
        assert data["text2"] == "sitting"
        assert data["algorithm"] == "jaro-winkler"
        assert "distance" in data
        assert "normalized" in data
        assert "similarity" in data

    def test_levenshtein(self) -> None:
        data = run_cli_json(
            ["compare", "kitten", "sitting", "--algorithm", "levenshtein", "--json"]
        )
        assert data["algorithm"] == "levenshtein"
        assert data["distance"] == 3.0

    def test_identical_strings(self) -> None:
        data = run_cli_json(["compare", "hello", "hello", "--json"])
        assert data["similarity"] == 1.0

    def test_hamming(self) -> None:
        data = run_cli_json(["compare", "karolin", "kathrin", "--algorithm", "hamming", "--json"])
        assert data["algorithm"] == "hamming"
        assert data["distance"] > 0

    def test_dice(self) -> None:
        data = run_cli_json(["compare", "night", "nacht", "--algorithm", "dice", "--json"])
        assert data["algorithm"] == "dice"

    def test_soundex(self) -> None:
        data = run_cli_json(["compare", "Robert", "Rupert", "--algorithm", "soundex", "--json"])
        assert data["algorithm"] == "soundex"

    def test_metaphone(self) -> None:
        data = run_cli_json(["compare", "Smith", "Schmidt", "--algorithm", "metaphone", "--json"])
        assert data["algorithm"] == "metaphone"

    def test_damerau(self) -> None:
        data = run_cli_json(["compare", "ab", "ba", "--algorithm", "damerau", "--json"])
        assert data["algorithm"] == "damerau"
        assert data["distance"] == 1.0

    def test_jaro(self) -> None:
        data = run_cli_json(["compare", "hello", "hallo", "--algorithm", "jaro", "--json"])
        assert data["algorithm"] == "jaro"


# ---------------------------------------------------------------------------
# 4. find
# ---------------------------------------------------------------------------


class TestFind:
    def test_basic_human(self, sample_text: Path) -> None:
        output = run_cli(["find", "Hello", str(sample_text)])
        assert "1 match" in output
        assert "Hello" in output

    def test_json_envelope(self, sample_text: Path) -> None:
        data = run_cli_json(["find", "the", str(sample_text), "--json"])
        assert data["command"] == "find"
        assert data["file"] == "sample.txt"
        assert data["pattern"] == "the"
        assert isinstance(data["matches"], list)
        assert "total" in data

    def test_case_insensitive(self, tmp_path: Path) -> None:
        f = tmp_path / "ci.txt"
        f.write_text("Hello HELLO hello", encoding="utf-8")
        data = run_cli_json(["find", "hello", str(f), "--case-insensitive", "--json"])
        assert data["total"] == 3

    def test_no_match(self, sample_text: Path) -> None:
        data = run_cli_json(["find", "zzz_nonexistent", str(sample_text), "--json"])
        assert data["total"] == 0
        assert data["matches"] == []

    def test_no_match_human(self, sample_text: Path) -> None:
        output = run_cli(["find", "zzz_nonexistent", str(sample_text)])
        assert "No matches" in output

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            run_cli(["find", "hello", str(tmp_path / "missing.txt")])


# ---------------------------------------------------------------------------
# 5. search
# ---------------------------------------------------------------------------


class TestSearch:
    def _build_index(self, corpus_file: Path, tmp_path: Path) -> Path:
        """Build an index and return its path."""
        index_path = tmp_path / "test_index.json"
        run_cli(["index", "build", str(corpus_file), "--output", str(index_path)])
        return index_path

    def test_search_human(self, corpus_file: Path, tmp_path: Path) -> None:
        index_path = self._build_index(corpus_file, tmp_path)
        output = run_cli(["search", "--index", str(index_path), "cat"])
        assert "cat" in output.lower()

    def test_search_json(self, corpus_file: Path, tmp_path: Path) -> None:
        index_path = self._build_index(corpus_file, tmp_path)
        data = run_cli_json(["search", "--index", str(index_path), "cat", "--json"])
        assert data["command"] == "search"
        assert data["query"] == "cat"
        assert data["total"] > 0
        assert isinstance(data["results"], list)
        first = data["results"][0]
        assert "doc_id" in first
        assert "score" in first

    def test_search_no_results(self, corpus_file: Path, tmp_path: Path) -> None:
        index_path = self._build_index(corpus_file, tmp_path)
        data = run_cli_json(["search", "--index", str(index_path), "xyzzy_nonexistent", "--json"])
        assert data["total"] == 0

    def test_search_top_k(self, corpus_file: Path, tmp_path: Path) -> None:
        index_path = self._build_index(corpus_file, tmp_path)
        data = run_cli_json(["search", "--index", str(index_path), "the", "--top-k", "2", "--json"])
        assert len(data["results"]) <= 2

    def test_search_native_index(self, corpus_file: Path, tmp_path: Path) -> None:
        index_path = tmp_path / "test_index.kncidx"
        run_cli(["index", "build", str(corpus_file), "--output", str(index_path)])
        data = run_cli_json(["search", "--index", str(index_path), "cat", "--json"])
        assert data["format"] == "native"
        assert data["total"] > 0

    def test_missing_index(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            run_cli(["search", "--index", str(tmp_path / "missing.json"), "hello"])


# ---------------------------------------------------------------------------
# 6. index build
# ---------------------------------------------------------------------------


class TestIndexBuild:
    def test_build_human(self, corpus_file: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "idx.json"
        output = run_cli(["index", "build", str(corpus_file), "--output", str(output_path)])
        assert "Built index" in output
        assert output_path.exists()

    def test_build_json(self, corpus_file: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "idx.json"
        data = run_cli_json(
            ["index", "build", str(corpus_file), "--output", str(output_path), "--json"]
        )
        assert data["command"] == "index"
        assert data["action"] == "build"
        assert data["documents"] == 4
        assert data["terms"] > 0
        assert output_path.exists()

    def test_index_file_structure(self, corpus_file: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "idx.json"
        run_cli(["index", "build", str(corpus_file), "--output", str(output_path)])
        index_data = json.loads(output_path.read_text())
        assert "documents" in index_data
        assert "index_data" in index_data
        assert "total_documents" in index_data
        assert index_data["total_documents"] == 4

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            run_cli(["index", "build", str(tmp_path / "missing.txt")])


# ---------------------------------------------------------------------------
# 7. hash
# ---------------------------------------------------------------------------


class TestHash:
    def test_ctph_human(self, sample_text: Path) -> None:
        output = run_cli(["hash", str(sample_text)])
        assert "CTPH" in output
        assert "Hash:" in output

    def test_ctph_json(self, sample_text: Path) -> None:
        data = run_cli_json(["hash", str(sample_text), "--json"])
        assert data["command"] == "hash"
        assert data["file"] == "sample.txt"
        assert data["algorithm"] == "ctph"
        assert isinstance(data["hash"], str)
        assert ":" in data["hash"]

    def test_minhash_json(self, sample_text: Path) -> None:
        data = run_cli_json(["hash", str(sample_text), "--algorithm", "minhash", "--json"])
        assert data["command"] == "hash"
        assert data["algorithm"] == "minhash"
        assert data["num_permutations"] == 128
        assert isinstance(data["signature_preview"], list)

    def test_minhash_human(self, sample_text: Path) -> None:
        output = run_cli(["hash", str(sample_text), "--algorithm", "minhash"])
        assert "MinHash" in output

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            run_cli(["hash", str(tmp_path / "missing.txt")])


# ---------------------------------------------------------------------------
# 8. duplicates
# ---------------------------------------------------------------------------


class TestDuplicates:
    def test_human_output(self, txt_dir: Path) -> None:
        output = run_cli(["duplicates", str(txt_dir), "--threshold", "0.3"])
        # May or may not find duplicates depending on threshold; just check it runs
        assert isinstance(output, str)

    def test_json_envelope(self, txt_dir: Path) -> None:
        data = run_cli_json(["duplicates", str(txt_dir), "--threshold", "0.3", "--json"])
        assert data["command"] == "duplicates"
        assert data["threshold"] == 0.3
        assert data["files_scanned"] == 3
        assert isinstance(data["groups"], list)

    def test_empty_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(SystemExit):
            run_cli(["duplicates", str(d)])

    def test_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            run_cli(["duplicates", str(tmp_path / "nonexistent")])


# ---------------------------------------------------------------------------
# 9. encode
# ---------------------------------------------------------------------------


class TestEncode:
    def test_soundex_human(self) -> None:
        output = run_cli(["encode", "Robert"])
        assert "soundex" in output
        assert "Encoding:" in output

    def test_soundex_json(self) -> None:
        data = run_cli_json(["encode", "Robert", "--json"])
        assert data["command"] == "encode"
        assert data["text"] == "Robert"
        assert data["algorithm"] == "soundex"
        assert isinstance(data["encoding"], str)
        assert len(data["encoding"]) > 0

    def test_metaphone_json(self) -> None:
        data = run_cli_json(["encode", "Smith", "--algorithm", "metaphone", "--json"])
        assert data["algorithm"] == "metaphone"
        assert isinstance(data["encoding"], str)

    def test_metaphone_human(self) -> None:
        output = run_cli(["encode", "Smith", "--algorithm", "metaphone"])
        assert "metaphone" in output


# ---------------------------------------------------------------------------
# 10. vocab build
# ---------------------------------------------------------------------------


class TestVocab:
    def test_frequency_human(self, sample_text: Path) -> None:
        output = run_cli(["vocab", "build", str(sample_text)])
        assert "frequency" in output
        assert "Unique terms:" in output

    def test_frequency_json(self, sample_text: Path) -> None:
        data = run_cli_json(["vocab", "build", str(sample_text), "--json"])
        assert data["command"] == "vocab"
        assert data["action"] == "build"
        assert data["type"] == "frequency"
        assert data["total_terms"] > 0
        assert data["unique_terms"] > 0
        assert isinstance(data["top_terms"], list)
        if data["top_terms"]:
            assert "term" in data["top_terms"][0]
            assert "count" in data["top_terms"][0]

    def test_indexed_json(self, sample_text: Path) -> None:
        data = run_cli_json(["vocab", "build", str(sample_text), "--type", "indexed", "--json"])
        assert data["type"] == "indexed"
        assert data["unique_terms"] > 0

    def test_bloom_json(self, sample_text: Path) -> None:
        data = run_cli_json(["vocab", "build", str(sample_text), "--type", "bloom", "--json"])
        assert data["type"] == "bloom"
        assert data["approx_unique"] > 0

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            run_cli(["vocab", "build", str(tmp_path / "missing.txt")])


# ---------------------------------------------------------------------------
# 11. analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_human(self, sample_text: Path) -> None:
        output = run_cli(["analyze", str(sample_text)])
        assert "Tokens:" in output
        assert "Sentences:" in output
        assert "Paragraphs:" in output

    def test_json_envelope(self, sample_text: Path) -> None:
        data = run_cli_json(["analyze", str(sample_text), "--json"])
        assert data["command"] == "analyze"
        assert data["file"] == "sample.txt"
        assert data["characters"] > 0
        assert data["tokens"] > 0
        assert data["unique_terms"] > 0
        assert "token_properties" in data
        assert data["sentences"] > 0
        assert data["lines"] > 0
        assert data["paragraphs"] > 0
        assert "avg_sentence_length" in data
        assert "type_token_ratio" in data
        assert isinstance(data["top_terms"], list)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            run_cli(["analyze", str(tmp_path / "missing.txt")])


class TestReadability:
    def test_human(self, sample_text: Path) -> None:
        output = run_cli(["readability", str(sample_text)])
        assert "Flesch-Kincaid:" in output
        assert "Flesch Reading Ease:" in output
        assert "Gunning Fog:" in output
        assert "SMOG:" in output
        assert "Dale-Chall" not in output  # no familiar-word list supplied

    def test_json_golden(self, tmp_path: Path) -> None:
        f = tmp_path / "golden.txt"
        f.write_text("The cat sat on the mat. The dog ate a bone.", encoding="utf-8")
        data = run_cli_json(["readability", str(f), "--json"])
        assert data["command"] == "readability"
        assert data["file"] == "golden.txt"
        assert data["counts"] == {
            "sentences": 2,
            "words": 11,
            "letters": 31,
            "letters_and_digits": 31,
            "syllables": 11,
            "polysyllable_words": 0,
            "fog_complex_words": 0,
            "long_words": 0,
        }
        assert data["scores"] == {
            "flesch_reading_ease": 116.6525,
            "flesch_kincaid_grade": -1.645,
            "automated_readability_index": -5.4064,
            "coleman_liau_index": -4.6109,
            "smog_index": 3.1291,
            "gunning_fog": 2.2,
            "lix": 5.5,
            "rix": 0.0,
            "smog_valid": False,
        }

    def test_naive_fog_flag(self, tmp_path: Path) -> None:
        f = tmp_path / "fog.txt"
        f.write_text("We toured Wisconsin. The state-of-the-art trespasses arrived.")
        strict = run_cli_json(["readability", str(f), "--json"])
        naive = run_cli_json(["readability", str(f), "--json", "--naive-fog"])
        assert naive["scores"]["gunning_fog"] >= strict["scores"]["gunning_fog"]
        assert naive["counts"]["fog_complex_words"] >= strict["counts"]["fog_complex_words"]

    def test_familiar_words_enables_dale_chall(self, tmp_path: Path) -> None:
        from kaos_nlp_core.matching import FstSet

        wordset = tmp_path / "familiar.fst"
        FstSet(["the", "cat", "sat", "on", "mat", "dog", "ate", "a"]).save(str(wordset))
        f = tmp_path / "dc.txt"
        f.write_text("The cat sat on the mat. The dog ate a bone.", encoding="utf-8")
        data = run_cli_json(["readability", str(f), "--json", "--familiar-words", str(wordset)])
        assert data["counts"]["unfamiliar_words"] == 1
        assert data["scores"]["dale_chall"] == pytest.approx(5.3447, abs=1e-3)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            run_cli(["readability", str(tmp_path / "missing.txt")])


# ---------------------------------------------------------------------------
# General / Edge cases
# ---------------------------------------------------------------------------


class TestGeneral:
    def test_no_command(self) -> None:
        with pytest.raises(SystemExit):
            main([])

    def test_help(self) -> None:
        with pytest.raises(SystemExit):
            main(["--help"])

    def test_invalid_command(self) -> None:
        with pytest.raises(SystemExit):
            main(["nonexistent_command"])

    def test_empty_file_tokenize(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        data = run_cli_json(["tokenize", str(f), "--json"])
        assert data["total"] == 0
        assert data["tokens"] == []

    def test_empty_file_segment(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        data = run_cli_json(["segment", str(f), "--json"])
        assert data["total"] == 0

    def test_unicode_text(self, tmp_path: Path) -> None:
        f = tmp_path / "unicode.txt"
        f.write_text("Bonjour le monde. Caf\u00e9 cr\u00e8me.", encoding="utf-8")
        data = run_cli_json(["tokenize", str(f), "--json"])
        assert data["total"] > 0
        # Verify offsets are correct (char-based, not byte-based)
        text = f.read_text(encoding="utf-8")
        for t in data["tokens"]:
            extracted = text[t["start"] : t["end"]]
            # The extracted span should contain the token text
            # (span may include surrounding punctuation that gets stripped)
            assert t["text"] in extracted or extracted.startswith(t["text"])
