"""Benchmarks for segmentation: sentence tokenization, paragraph splitting, training.

Measures:
  - Sentence segmentation speed on War and Peace (3.2MB)
  - Sentence segmentation with legal model
  - Paragraph segmentation speed
  - Training speed on USC corpus
  - Batch segmentation throughput
"""

from pathlib import Path

import pytest

from kaos_nlp_core.segmentation import (
    PunktParameters,
    PunktTokenizer,
    PunktTrainer,
    segment_lines,
)

MODEL_PATH = Path(__file__).parent / ".." / "models" / "default.npkt.gz"
DATA_PATH = Path(__file__).parent / ".." / "data" / "legal_abbreviations.json"


@pytest.fixture(scope="module")
def legal_tokenizer():
    """Tokenizer with default nupunkt model."""
    if MODEL_PATH.exists():
        params = PunktParameters.load(str(MODEL_PATH))
        return PunktTokenizer(params)
    return PunktTokenizer()


# ── Sentence segmentation benchmarks ─────────────────────────────────────────


@pytest.mark.benchmark(group="segmentation_sentences")
def test_segment_sentences_war_peace(benchmark, war_and_peace_text):
    """Sentence segmentation on War and Peace (3.2MB, no model)."""
    tok = PunktTokenizer()
    result = benchmark(tok.tokenize, war_and_peace_text)
    assert len(result) > 1000


@pytest.mark.benchmark(group="segmentation_sentences")
def test_segment_sentences_war_peace_with_model(benchmark, war_and_peace_text, legal_tokenizer):
    """Sentence segmentation on War and Peace with legal model."""
    result = benchmark(legal_tokenizer.tokenize, war_and_peace_text)
    assert len(result) > 1000


@pytest.mark.benchmark(group="segmentation_sentences")
def test_segment_sentences_short(benchmark, legal_tokenizer):
    """Sentence segmentation on a short legal paragraph."""
    text = (
        "See 42 U.S.C. § 1983. Dr. Smith v. Jones established the standard. "
        "The court held that the defendant was liable. Mr. Johnson testified."
    )
    benchmark(legal_tokenizer.tokenize, text)


@pytest.mark.benchmark(group="segmentation_sentences")
def test_segment_spans_war_peace(benchmark, war_and_peace_text):
    """Sentence span extraction on War and Peace."""
    tok = PunktTokenizer()
    result = benchmark(tok.tokenize_spans, war_and_peace_text)
    assert len(result) > 1000


# ── Paragraph segmentation benchmarks ────────────────────────────────────────


@pytest.mark.benchmark(group="segmentation_paragraphs")
def test_segment_paragraphs_war_peace(benchmark, war_and_peace_text, legal_tokenizer):
    """Paragraph segmentation on War and Peace (sentence-aware)."""
    result = benchmark(legal_tokenizer.tokenize_paragraphs_flat, war_and_peace_text)
    assert len(result) > 100


@pytest.mark.benchmark(group="segmentation_paragraphs")
def test_segment_lines_war_peace(benchmark, war_and_peace_text):
    """Line segmentation on War and Peace."""
    result = benchmark(segment_lines, war_and_peace_text)
    assert len(result) > 1000


# ── Training benchmarks ──────────────────────────────────────────────────────


@pytest.mark.benchmark(group="training")
def test_train_war_peace_100k(benchmark, war_and_peace_text):
    """Train on first 100K chars of War and Peace."""
    text = war_and_peace_text[:100000]

    def train():
        trainer = PunktTrainer()
        return trainer.train(text)

    result = benchmark.pedantic(train, rounds=3, iterations=1)
    assert result.num_abbreviations > 0


@pytest.mark.benchmark(group="training")
def test_train_usc_sample(benchmark, usc_docs):
    """Train on 100 USC documents with legal abbreviations."""
    corpus = "\n\n".join(d["text"] for d in usc_docs[:100])

    def train():
        trainer = PunktTrainer()
        if DATA_PATH.exists():
            trainer.load_abbreviations_from_json(str(DATA_PATH))
        return trainer.train(corpus)

    result = benchmark.pedantic(train, rounds=3, iterations=1)
    assert result.num_abbreviations > 0


# ── Batch benchmarks ─────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="segmentation_batch")
def test_batch_100_usc_docs(benchmark, usc_docs, legal_tokenizer):
    """Batch sentence segmentation on 100 USC documents."""
    texts = [d["text"] for d in usc_docs[:100]]
    result = benchmark(legal_tokenizer.tokenize_batch, texts)
    assert len(result) == 100


@pytest.mark.benchmark(group="segmentation_batch")
def test_count_sentences_100_usc_docs(benchmark, usc_docs, legal_tokenizer):
    """Count sentences in 100 USC documents."""
    texts = [d["text"] for d in usc_docs[:100]]

    def count_all():
        return sum(legal_tokenizer.count_sentences(t) for t in texts)

    total = benchmark(count_all)
    assert total > 100
