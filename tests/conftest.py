"""Shared test fixtures for kaos-nlp-core."""

import json
from pathlib import Path

import pytest

from kaos_nlp_core.segmentation import PunktParameters, PunktTokenizer
from kaos_nlp_core.tokenizer import Tokenizer

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MODEL_DIR = Path(__file__).parent.parent / "models"

HF_SKIP_MSG = (
    "Run: uv run --with huggingface_hub,tokenizers,datasets"
    " python tests/fixtures/download_hf_fixtures.py"
)

# Shared tokenizer instance for consistent tokenization across all tests.
# Matches the kelvin-nlp default: lowercase, strip punctuation, no prefix.
DEFAULT_TOKENIZER = Tokenizer(lowercase=True)

# Shared Punkt sentence tokenizer. Uses default model if available.
_default_model_path = MODEL_DIR / "default.npkt.gz"
if _default_model_path.exists():
    DEFAULT_PUNKT = PunktTokenizer(PunktParameters.load(str(_default_model_path)))
else:
    DEFAULT_PUNKT = PunktTokenizer()


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


@pytest.fixture(scope="session")
def war_and_peace_text() -> str:
    """Full text of War and Peace from Project Gutenberg."""
    path = FIXTURE_DIR / "war_and_peace.txt"
    if not path.exists():
        pytest.skip("Run tests/fixtures/download_fixtures.sh first")
    return path.read_text(encoding="utf-8-sig")


@pytest.fixture(scope="session")
def shakespeare_text() -> str:
    """Complete Works of Shakespeare from Project Gutenberg."""
    path = FIXTURE_DIR / "shakespeare.txt"
    if not path.exists():
        pytest.skip("Run tests/fixtures/download_fixtures.sh first")
    return path.read_text(encoding="utf-8-sig")


@pytest.fixture(scope="session")
def war_and_peace_words(war_and_peace_text: str) -> list[str]:
    """All words from War and Peace, lowercased and punctuation-stripped."""
    return DEFAULT_TOKENIZER.tokenize_words(war_and_peace_text)


@pytest.fixture(scope="session")
def war_and_peace_paragraphs(war_and_peace_text: str) -> list[str]:
    """Paragraphs from War and Peace via Punkt sentence-aware segmentation.

    At least 200 chars, capped at 500 chars.
    """
    paras = DEFAULT_PUNKT.tokenize_paragraphs_flat(war_and_peace_text)
    return [p[:500] for p in paras if len(p) > 200]


# ── HuggingFace dataset fixtures ─────────────────────────────────────────────


@pytest.fixture(scope="session")
def usc_docs() -> list[dict]:
    """Full US Code corpus (~69K sections). Each doc has 'id', 'identifier', 'text'."""
    path = FIXTURE_DIR / "usc.jsonl"
    if not path.exists():
        pytest.skip(HF_SKIP_MSG)
    return _load_jsonl(path)


@pytest.fixture(scope="session")
def edgar_docs() -> list[dict]:
    """200 SEC EDGAR agreements. Each doc has 'id', 'identifier', 'text'."""
    path = FIXTURE_DIR / "edgar_agreements.jsonl"
    if not path.exists():
        pytest.skip(HF_SKIP_MSG)
    return _load_jsonl(path)


@pytest.fixture(scope="session")
def patent_docs() -> list[dict]:
    """200 US patents. Each doc has 'id', 'title', 'abstract', 'claims', 'text'."""
    path = FIXTURE_DIR / "patents.jsonl"
    if not path.exists():
        pytest.skip(HF_SKIP_MSG)
    return _load_jsonl(path)
