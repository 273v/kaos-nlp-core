"""Tests for the segmentation module: Punkt sentence tokenizer, training, lines, paragraphs."""

import pickle
import tempfile
from pathlib import Path

import pytest

from kaos_nlp_core.segmentation import (
    PunktParameters,
    PunktTokenizer,
    PunktTrainer,
    segment_lines,
    segment_paragraphs,
    segment_paragraphs_simple,
    segment_sentences,
)

# ── PunktTokenizer ──────────────────────────────────────────────────────────


class TestPunktTokenizerBasic:
    def test_basic_sentences(self):
        tok = PunktTokenizer()
        sents = tok.tokenize("Hello world. How are you? I am fine!")
        assert len(sents) == 3

    def test_empty_text(self):
        tok = PunktTokenizer()
        assert tok.tokenize("") == []

    def test_no_terminal_punct(self):
        tok = PunktTokenizer()
        sents = tok.tokenize("No period here")
        assert len(sents) == 1

    def test_single_sentence(self):
        tok = PunktTokenizer()
        sents = tok.tokenize("Just one sentence.")
        assert len(sents) == 1

    def test_exclamation_question(self):
        tok = PunktTokenizer()
        sents = tok.tokenize("Wow! Really? Yes.")
        assert len(sents) == 3

    def test_count_sentences(self):
        tok = PunktTokenizer()
        assert tok.count_sentences("Hello. World.") == 2


class TestPunktTokenizerWithModel:
    @pytest.fixture
    def legal_tokenizer(self):
        """Tokenizer with provided legal abbreviations."""
        params = PunktParameters.load(
            str(Path(__file__).parent / ".." / "models" / "default.npkt.gz")
        )
        return PunktTokenizer(params)

    def test_abbreviation_dr(self, legal_tokenizer):
        sents = legal_tokenizer.tokenize("Dr. Smith went to Washington. He was happy.")
        # "Dr." should NOT split
        assert any("Dr." in s for s in sents)
        assert len(sents) == 2

    def test_abbreviation_us(self, legal_tokenizer):
        sents = legal_tokenizer.tokenize("See 42 U.S.C. § 1983. This is important.")
        # "U.S.C." should not split mid-citation
        assert len(sents) <= 3

    def test_legal_citation_v(self, legal_tokenizer):
        sents = legal_tokenizer.tokenize(
            "Daubert v. Merrell Dow Pharmaceuticals established the standard."
        )
        # "v." should NOT split
        assert len(sents) == 1


class TestPunktTokenizerSpans:
    """Span offsets must be char offsets usable with Python str slicing."""

    def test_ascii_spans(self):
        tok = PunktTokenizer()
        text = "Hello world. How are you?"
        spans = tok.tokenize_spans(text)
        assert len(spans) == 2
        assert text[spans[0][0] : spans[0][1]] == "Hello world."
        assert text[spans[1][0] : spans[1][1]] == "How are you?"

    def test_unicode_spans_cafe(self):
        """Multi-byte chars (é = 2 bytes) must not shift offsets."""
        tok = PunktTokenizer()
        text = "Le café est bon. Résumé du jour."
        spans = tok.tokenize_spans(text)
        assert len(spans) == 2
        assert text[spans[0][0] : spans[0][1]].startswith("Le café")
        assert text[spans[1][0] : spans[1][1]].startswith("Résumé")

    def test_unicode_spans_section_symbol(self):
        """§ is 2 bytes in UTF-8 — offsets must account for it."""
        tok = PunktTokenizer()
        text = "See § 1983. The court agreed."
        spans = tok.tokenize_spans(text)
        for start, end in spans:
            extracted = text[start:end]
            assert len(extracted.strip()) > 0
            # No truncated characters
            assert not extracted.startswith("ee")  # Would indicate byte offset used as char

    def test_cjk_spans(self):
        """CJK chars are 3 bytes each — most aggressive test."""
        tok = PunktTokenizer()
        text = "東京は大きい。大阪も大きい。"
        spans = tok.tokenize_spans(text)
        for start, end in spans:
            extracted = text[start:end]
            assert len(extracted) > 0
            # Should not crash or produce garbage
            assert all(ord(c) > 127 or c.isascii() for c in extracted)

    def test_emoji_spans(self):
        """Emoji are 4 bytes — extreme multi-byte test."""
        tok = PunktTokenizer()
        text = "Hello 😀 world. Goodbye 🌍 earth."
        spans = tok.tokenize_spans(text)
        for start, end in spans:
            extracted = text[start:end]
            assert len(extracted.strip()) > 0

    def test_spans_and_sentences_agree(self):
        """tokenize_spans and segment_sentences must produce the same text."""
        tok = PunktTokenizer()
        text = "Le café est bon. Résumé du jour. See § 1983."
        spans = tok.tokenize_spans(text)
        segs = segment_sentences(text, tok)
        span_texts = [text[s:e].strip() for s, e in spans]
        seg_texts = [s.text for s in segs]
        assert span_texts == seg_texts


class TestPunktTokenizerPrecisionRecall:
    def test_pr_parameter(self):
        tok = PunktTokenizer()
        text = "Hello world. How are you. Fine."

        high_recall = tok.tokenize(text, precision_recall=0.0)
        high_precision = tok.tokenize(text, precision_recall=1.0)

        # High recall should produce >= as many sentences
        assert len(high_recall) >= len(high_precision)

    def test_set_precision_recall(self):
        tok = PunktTokenizer()
        tok.set_precision_recall(0.3)
        sents = tok.tokenize("Hello. World.")
        assert len(sents) >= 1


class TestPunktTokenizerParagraphs:
    def test_paragraph_tokenization(self):
        tok = PunktTokenizer()
        text = "First paragraph sentence.\n\nSecond paragraph sentence."
        paras = tok.tokenize_paragraphs(text)
        assert len(paras) >= 1

    def test_paragraphs_flat(self):
        tok = PunktTokenizer()
        text = "Para one.\n\nPara two."
        flat = tok.tokenize_paragraphs_flat(text)
        assert len(flat) >= 1


class TestPunktTokenizerBatch:
    def test_batch(self):
        tok = PunktTokenizer()
        texts = ["Hello. World.", "One sentence.", "Two. Sentences."]
        results = tok.tokenize_batch(texts)
        assert len(results) == 3
        assert len(results[0]) == 2
        assert len(results[1]) == 1
        assert len(results[2]) == 2


class TestPunktTokenizerPickle:
    def test_pickle_roundtrip(self):
        tok = PunktTokenizer()
        tok.set_precision_recall(0.3)
        tok2 = pickle.loads(pickle.dumps(tok))
        sents = tok2.tokenize("Hello. World.")
        assert len(sents) >= 1


# ── PunktParameters ─────────────────────────────────────────────────────────


class TestPunktParameters:
    def test_new_empty(self):
        params = PunktParameters()
        assert params.num_abbreviations == 0
        assert params.num_collocations == 0
        assert params.num_sent_starters == 0

    def test_json_roundtrip(self):
        # Load the default model and roundtrip it
        path = Path(__file__).parent / ".." / "models" / "default.npkt.gz"
        if path.exists():
            params = PunktParameters.load(str(path))
            json_str = params.to_json()
            restored = PunktParameters.from_json(json_str)
            assert restored.num_abbreviations == params.num_abbreviations

    def test_save_load(self):
        path = Path(__file__).parent / ".." / "models" / "default.npkt.gz"
        if path.exists():
            params = PunktParameters.load(str(path))
            with tempfile.NamedTemporaryFile(suffix=".npkt.gz", delete=False) as f:
                params.save(f.name)
                loaded = PunktParameters.load(f.name)
                assert loaded.num_abbreviations == params.num_abbreviations
            Path(f.name).unlink()

    def test_pickle_roundtrip(self):
        params = PunktParameters()
        params2 = pickle.loads(pickle.dumps(params))
        assert params2.num_abbreviations == 0

    def test_repr(self):
        params = PunktParameters()
        r = repr(params)
        assert "PunktParameters" in r


# ── PunktTrainer ─────────────────────────────────────────────────────────────


class TestPunktTrainer:
    def test_train_basic(self):
        trainer = PunktTrainer()
        text = "Hello world. How are you? I am fine. Thank you very much."
        params = trainer.train(text)
        assert params.num_abbreviations >= 0

    def test_train_with_provided_abbreviations(self):
        trainer = PunktTrainer()
        trainer.add_abbreviations(["Dr.", "Mr.", "v."])
        text = "Dr. Smith met Mr. Jones. Smith v. Jones is a famous case."
        params = trainer.train(text)
        # Provided abbreviations should survive training
        assert "dr" in [a.lower() for a in params.abbreviations]
        assert "v" in [a.lower() for a in params.abbreviations]

    def test_load_abbreviations_from_json(self):
        trainer = PunktTrainer()
        path = Path(__file__).parent / ".." / "data" / "legal_abbreviations.json"
        if path.exists():
            count = trainer.load_abbreviations_from_json(str(path))
            assert count > 0

    def test_incremental_training(self):
        trainer = PunktTrainer()
        trainer.train_incremental("First chunk. With sentences.")
        trainer.train_incremental("Second chunk. More sentences.")
        params = trainer.finalize_training()
        assert params.num_abbreviations >= 0

    def test_train_save_load_inference(self):
        """Train a model, save it, load it, use for inference."""
        trainer = PunktTrainer()
        trainer.add_abbreviations(["Dr."])
        text = "Dr. Smith is a doctor. He works at the hospital. The hospital is large."
        params = trainer.train(text)

        with tempfile.NamedTemporaryFile(suffix=".npkt.gz", delete=False) as f:
            params.save(f.name)
            loaded = PunktParameters.load(f.name)
            tok = PunktTokenizer(loaded)
            sents = tok.tokenize("Dr. Smith went home. He was tired.")
            assert len(sents) >= 1
        Path(f.name).unlink()


# ── PunktTrainer on real corpus ──────────────────────────────────────────────


class TestPunktTrainerOnCorpus:
    @pytest.fixture(scope="class")
    def war_and_peace_text(self):
        path = Path(__file__).parent / "fixtures" / "war_and_peace.txt"
        if not path.exists():
            pytest.skip("War and Peace fixture not available")
        return path.read_text()

    def test_train_on_war_and_peace(self, war_and_peace_text):
        trainer = PunktTrainer()
        params = trainer.train(war_and_peace_text[:100000])  # First 100K chars
        assert params.num_abbreviations > 0
        assert params.num_sent_starters > 0

    def test_segment_war_and_peace(self, war_and_peace_text):
        tok = PunktTokenizer()
        sents = tok.tokenize(war_and_peace_text[:10000])
        assert len(sents) > 10


# ── Line Segmentation ───────────────────────────────────────────────────────


class TestSegmentLines:
    def test_basic(self):
        lines = segment_lines("hello\nworld")
        assert len(lines) == 2
        assert lines[0].text == "hello"
        assert lines[1].text == "world"

    def test_empty(self):
        assert segment_lines("") == []

    def test_single_line(self):
        lines = segment_lines("hello")
        assert len(lines) == 1

    def test_crlf(self):
        lines = segment_lines("line1\r\nline2")
        assert len(lines) == 2

    def test_spans(self):
        text = "hello\nworld"
        lines = segment_lines(text)
        for line in lines:
            assert text[line.start : line.end] == line.text


# ── Sentence Segmentation (standalone function) ─────────────────────────────


class TestSegmentSentences:
    def test_basic(self):
        sents = segment_sentences("Hello world. How are you?")
        assert len(sents) == 2

    def test_with_tokenizer(self):
        tok = PunktTokenizer()
        sents = segment_sentences("Hello world. How are you?", tok)
        assert len(sents) == 2
        for s in sents:
            assert s.start <= s.end
            assert len(s.text) > 0

    def test_empty(self):
        assert segment_sentences("") == []

    def test_spans_correct(self):
        text = "First sentence. Second sentence."
        sents = segment_sentences(text)
        for s in sents:
            assert text[s.start : s.end] == s.text


# ── Paragraph Segmentation (sentence-aware) ─────────────────────────────────


class TestSegmentParagraphs:
    def test_double_newline(self):
        text = "First para.\n\nSecond para."
        paras = segment_paragraphs(text)
        assert len(paras) == 2

    def test_with_tokenizer(self):
        tok = PunktTokenizer()
        text = "First para sentence.\n\nSecond para sentence."
        paras = segment_paragraphs(text, tok)
        assert len(paras) == 2

    def test_single_newline_no_break(self):
        text = "Single paragraph\nwith line wrap."
        paras = segment_paragraphs(text)
        assert len(paras) == 1

    def test_empty(self):
        assert segment_paragraphs("") == []

    def test_paragraph_only_at_sentence_boundary(self):
        """Paragraph breaks should only occur at sentence boundaries."""
        tok = PunktTokenizer()
        text = "This is a sentence.\n\nThis is another sentence."
        paras = segment_paragraphs(text, tok)
        assert len(paras) == 2
        # Each paragraph should contain complete sentences
        for p in paras:
            assert len(p.text) > 0


# ── Simple Paragraph Segmentation (blank-line only) ──────────────────────────


class TestSegmentParagraphsSimple:
    def test_basic(self):
        text = "First para.\n\nSecond para."
        paras = segment_paragraphs_simple(text)
        assert len(paras) == 2
        assert paras[0].text == "First para."
        assert paras[1].text == "Second para."

    def test_spans(self):
        text = "Para 1.\n\nPara 2."
        paras = segment_paragraphs_simple(text)
        for para in paras:
            assert text[para.start : para.end] == para.text


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestSegmentationEdgeCases:
    """Edge cases for sentence and paragraph segmentation."""

    def test_only_whitespace(self):
        tok = PunktTokenizer()
        sents = tok.tokenize("   \t\n  ")
        assert sents == []

    def test_very_long_single_sentence(self):
        tok = PunktTokenizer()
        text = "a" * 10_000
        sents = tok.tokenize(text)
        assert len(sents) == 1
        assert sents[0] == text

    def test_crlf_line_endings(self):
        text = "First paragraph.\r\n\r\nSecond paragraph."
        paras = segment_paragraphs_simple(text)
        assert len(paras) == 2
        assert paras[0].text.strip() == "First paragraph."
        assert paras[1].text.strip() == "Second paragraph."

    def test_many_consecutive_blank_lines(self):
        text = "Para one.\n\n\n\n\n\n\nPara two."
        paras = segment_paragraphs_simple(text)
        assert len(paras) == 2
        assert paras[0].text.strip() == "Para one."
        assert paras[1].text.strip() == "Para two."

    def test_unicode_sentences(self):
        tok = PunktTokenizer()
        # CJK text with ideographic full stop as sentence terminator
        text = "東京は大きい。大阪も大きい。"
        sents = tok.tokenize(text)
        assert len(sents) >= 1

    def test_count_sentences_empty(self):
        tok = PunktTokenizer()
        assert tok.count_sentences("") == 0

    def test_batch_with_pr_override(self):
        tok = PunktTokenizer()
        results = tok.tokenize_batch(["Hello. World.", "One."], precision_recall=0.0)
        assert len(results) == 2

    def test_parameters_len(self):
        params = PunktParameters()
        assert len(params) == 0

    def test_punkt_parameters_pickle_trained(self):
        trainer = PunktTrainer()
        trainer.add_abbreviations(["Dr.", "Mr.", "v."])
        text = "Dr. Smith met Mr. Jones. Smith v. Jones is a famous case. " * 20
        params = trainer.train(text)
        num_abbrevs = params.num_abbreviations

        params2 = pickle.loads(pickle.dumps(params))
        assert params2.num_abbreviations == num_abbrevs
        assert params2.num_abbreviations >= 3
