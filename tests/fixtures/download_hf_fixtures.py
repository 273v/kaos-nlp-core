#!/usr/bin/env env -S uv run --with huggingface_hub,tokenizers,datasets
"""Download test fixtures from alea-institute HuggingFace datasets.

Produces JSONL fixture files for realistic NLP testing:
  - usc.jsonl           : Full US Code (~69K sections) — BM25/TF-IDF retrieval corpus
  - edgar_agreements.jsonl : 200 SEC EDGAR agreements — legal document similarity
  - patents.jsonl        : 200 US patents (title/abstract/claims) — structured search

Tokenized datasets are decoded using the kl3m-004-128k-cased BPE tokenizer.

Usage:
    uv run --with huggingface_hub,tokenizers,datasets python tests/fixtures/download_hf_fixtures.py
"""

import json
import sys
import time
from pathlib import Path
from typing import TypedDict

FIXTURE_DIR = Path(__file__).parent


# ── Corpus configs ───────────────────────────────────────────────────────────
class CorpusConfig(TypedDict):
    repo: str
    output: str
    n_docs: int | None
    needs_detokenize: bool
    shuffle: bool
    description: str


CORPORA: dict[str, CorpusConfig] = {
    "usc": {
        "repo": "alea-institute/kl3m-data-usc",
        "output": "usc.jsonl",
        "n_docs": None,  # full dataset
        "needs_detokenize": True,
        "shuffle": False,
        "description": "US Code (federal statutes)",
    },
    "edgar_agreements": {
        "repo": "alea-institute/kl3m-data-edgar-agreements",
        "output": "edgar_agreements.jsonl",
        "n_docs": 200,
        "needs_detokenize": True,
        "shuffle": True,
        "description": "SEC EDGAR agreements (contracts)",
    },
    "patents": {
        "repo": "alea-institute/kl3m-sft-patents",
        "output": "patents.jsonl",
        "n_docs": 200,
        "needs_detokenize": False,  # already raw text
        "shuffle": True,
        "description": "US patents (title/abstract/claims)",
    },
}

TOKENIZER_REPO = "alea-institute/kl3m-004-128k-cased"


def load_tokenizer():
    """Load the kl3m BPE tokenizer for detokenizing integer sequences."""
    # `tokenizers` is intentionally NOT in the dev group — this script is
    # invoked via `uv run --with huggingface_hub,tokenizers,datasets ...`
    # per the docstring. ty can't see it, but it resolves at run time.
    from tokenizers import (  # ty: ignore[unresolved-import]
        Tokenizer,
    )

    print(f"Loading tokenizer: {TOKENIZER_REPO}")
    return Tokenizer.from_pretrained(TOKENIZER_REPO)


def download_tokenized_corpus(repo: str, n_docs: int | None, shuffle: bool, tokenizer):
    """Stream a tokenized dataset and decode tokens to text."""
    from datasets import load_dataset  # ty: ignore[unresolved-import]

    ds = load_dataset(repo, split="train", streaming=True)
    if shuffle:
        ds = ds.shuffle(seed=42, buffer_size=2000)

    docs = []
    for i, row in enumerate(ds):
        if n_docs is not None and i >= n_docs:
            break
        text = tokenizer.decode(row["tokens"], skip_special_tokens=True)
        doc = {
            "id": i,
            "identifier": row.get("identifier", ""),
            "text": text,
        }
        if "score" in row and row["score"] is not None:
            doc["score"] = round(float(row["score"]), 4)
        docs.append(doc)
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1} documents decoded")
    return docs


def download_patent_corpus(repo: str, n_docs: int | None, shuffle: bool):
    """Stream the patents dataset (already raw text)."""
    from datasets import load_dataset  # ty: ignore[unresolved-import]

    ds = load_dataset(repo, split="train", streaming=True)
    if shuffle:
        ds = ds.shuffle(seed=42, buffer_size=2000)

    docs = []
    for i, row in enumerate(ds):
        if n_docs is not None and i >= n_docs:
            break
        doc = {
            "id": i,
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "claims": row.get("claims", []),
            "text": row.get("markdown", ""),
        }
        docs.append(doc)
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1} patents loaded")
    return docs


def write_jsonl(docs: list[dict], path: Path):
    """Write documents as newline-delimited JSON."""
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def main():
    # Allow selecting specific corpora via CLI args
    requested = set(sys.argv[1:]) if len(sys.argv) > 1 else set(CORPORA.keys())
    invalid = requested - set(CORPORA.keys())
    if invalid:
        print(f"Unknown corpora: {invalid}. Available: {set(CORPORA.keys())}")
        sys.exit(1)

    tokenizer = None  # lazy load

    for name, cfg in CORPORA.items():
        if name not in requested:
            continue

        output_path = FIXTURE_DIR / cfg["output"]
        if output_path.exists():
            size_mb = output_path.stat().st_size / 1_000_000
            print(f"✓ {cfg['output']} already exists ({size_mb:.1f} MB)")
            continue

        n_label = "all" if cfg["n_docs"] is None else cfg["n_docs"]
        print(f"\nDownloading {cfg['description']} ({n_label} docs)...")
        t0 = time.time()

        if cfg["needs_detokenize"]:
            if tokenizer is None:
                tokenizer = load_tokenizer()
            docs = download_tokenized_corpus(cfg["repo"], cfg["n_docs"], cfg["shuffle"], tokenizer)
        else:
            docs = download_patent_corpus(cfg["repo"], cfg["n_docs"], cfg["shuffle"])

        write_jsonl(docs, output_path)
        elapsed = time.time() - t0
        size_mb = output_path.stat().st_size / 1_000_000
        print(f"✓ {cfg['output']}: {len(docs)} docs, {size_mb:.1f} MB ({elapsed:.1f}s)")

    print("\nAll fixtures ready.")


if __name__ == "__main__":
    main()
