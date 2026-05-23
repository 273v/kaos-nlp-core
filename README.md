# kaos-nlp-core

> **Part of [Kelvin Agentic OS](https://kelvin.legal) (KAOS)** — open agentic
> infrastructure for legal work, built by
> [273 Ventures](https://273ventures.com).
> See the [full KAOS package map](https://github.com/273v) for the rest of the stack.

[![PyPI - Version](https://img.shields.io/pypi/v/kaos-nlp-core)](https://pypi.org/project/kaos-nlp-core/)
[![Python](https://img.shields.io/pypi/pyversions/kaos-nlp-core)](https://pypi.org/project/kaos-nlp-core/)
[![License](https://img.shields.io/pypi/l/kaos-nlp-core)](https://github.com/273v/kaos-nlp-core/blob/main/LICENSE)
[![CI](https://github.com/273v/kaos-nlp-core/actions/workflows/ci.yml/badge.svg)](https://github.com/273v/kaos-nlp-core/actions/workflows/ci.yml)

`kaos-nlp-core` is a high-performance NLP primitives library for KAOS — a
pure-Rust core with Python bindings via PyO3/Maturin. It provides the
text-processing building blocks the rest of the stack relies on:
SIMD-accelerated string operations, multi-pattern matching, finite-state
transducers, sentence segmentation, BM25 retrieval, fuzzy hashing, and
typed Python wrappers throughout.

It is dependency-light: the BASE install pulls `kaos-nlp-core` itself,
`numpy>=2.1` (used by similarity, chunker/aggregation marshalling, and
retrieval helpers — see `pyproject.toml:54`), and the bundled Punkt
sentence-segmenter model (~12 MB). Optional extras layer in the rest
of the KAOS ecosystem.

## Use and authorship disclosure

`kaos-nlp-core` provides **deterministic** primitives — segmentation,
chunking, tokenization, search, aggregation, hashing — that take
text in and produce text spans / scores / hashes back. No network
calls, no LLM calls, no provider-side data transmission happens
from this package; everything runs in-process. Downstream consumers
(notably `kaos-llm-core` Programs) may transmit text derived from
these primitives to LLM providers, so callers handling sensitive
data should check the consuming package's data-handling disclosure.

This codebase is AI-assisted: substantial portions were generated
with Claude (Anthropic) and human-reviewed before commit. Public
behavior is covered by the test suite under `tests/`; large-corpus
scale tests are opt-in (`pytest tests/scale -m slow`) and require
the HF JSONL fixtures (USC, EDGAR, patents). Bug reports welcome
via GitHub Issues; security reports follow [`SECURITY.md`](SECURITY.md).

## Install

```bash
uv add kaos-nlp-core
# or
pip install kaos-nlp-core
```

`kaos-nlp-core` requires Python **3.13** or newer. The published wheels
are `cp313-abi3` — one wheel per OS/architecture covers every CPython
3.13+ minor (3.13, 3.14, 3.15, …). No re-release needed when 3.15 ships.

Platform coverage: Linux x86_64 (manylinux + musllinux), Linux aarch64
(manylinux + musllinux), macOS arm64, Windows x86_64, Windows arm64.

## Quick start

```python
from kaos_nlp_core import tokenizer, algorithms

# Two output shapes for tokenization:
#   tokenize_words → list[str]        — just the surface forms (fastest)
#   tokenize       → list[TokenSpan]  — .text / .start / .end when you
#                                       need character offsets back into
#                                       the source string
words = tokenizer.tokenize_words("kaos-nlp-core ships fast NLP primitives.")
print(words)
# ['kaos-nlp-core', 'ships', 'fast', 'NLP', 'primitives']

for s in tokenizer.tokenize("kaos-nlp-core ships fast NLP primitives.")[:3]:
    print(f"{s.start}-{s.end}: {s.text!r}")
# 0-13: 'kaos-nlp-core'
# 14-19: 'ships'
# 20-24: 'fast'

# Multi-byte safe (CJK + emoji) — offsets are CHARACTER offsets, not bytes
for s in tokenizer.tokenize("東京 emoji 😀 test"):
    print(f"{s.start}-{s.end}: {s.text!r}")
# 0-2: '東京'
# 3-8: 'emoji'
# 9-10: '😀'
# 11-15: 'test'

# Algorithms always return rich typed results
result = algorithms.levenshtein("kitten", "sitting")
print(f"distance={result.distance} similarity={result.similarity:.4f}")
# distance=3.0 similarity=0.5714
```

The `_words` shortcut exists wherever skipping offsets is meaningful work
(tokenization). Everywhere else — segmentation (`segment_sentences`,
`segment_paragraphs`, `segment_lines`), pattern matching, similarity
algorithms — the API only ships the rich typed shape, because the
metadata is the value.

## Concepts

The package is organized around a small set of typed primitives.

| Concept | What it is |
|---|---|
| **Algorithms** | `kaos_nlp_core.algorithms` — Levenshtein, Hamming, Jaro-Winkler, longest common substring, edit-distance variants. SIMD fast paths via stringzilla; ASCII fast paths before Unicode fallbacks. |
| **Tokenizer** | `kaos_nlp_core.tokenizer` — Unicode-aware word/sentence tokenization with byte→char offset translation via `build_byte_to_char_table()`. Multi-byte safe (Latin diacritics, CJK, emoji). |
| **Segmentation** | `kaos_nlp_core.segmentation` — Punkt sentence segmenter (bundled model `models/default.npkt.gz`, ~12 MB Apache-2.0 NLTK port). |
| **Matching** | `kaos_nlp_core.matching` — Aho-Corasick multi-pattern matching, FST-backed fuzzy lookup via Levenshtein automata, regex. |
| **Search** | `kaos_nlp_core.search` — BM25 retrieval, `Searcher`, sentence/paragraph search; pickle-safe with KNC magic header for index files. |
| **Structures** | `kaos_nlp_core.structures` — `Vocabulary`, `InvertedIndex`, `SparseTermMatrix`, `SimilarityMatrix`. Compact, pickle-safe, bincode-2.0 backed. |
| **Hashing** | `kaos_nlp_core.hashing` — CTPH (context-triggered piecewise hashing) via blake3, MinHash, LSH index, near-duplicate grouping. |
| **Lexicon** | `kaos_nlp_core.lexicon` — query expansion, semantic graph traversal, gazetteer lookups. |
| **Documents** | `kaos_nlp_core.documents` — `Document`, `DocumentCollection` with JSONL / HuggingFace loaders. |
| **Quality** | `kaos_nlp_core.quality` — text-quality heuristics (token ratios, Unicode block distribution). |

## CLI

`kaos-nlp-core` ships a `kaos-nlp` administrative CLI plus an optional
`kaos-nlp-serve` MCP server (loopback-only by default; `--http` requires
`KAOS_NLP_HTTP_TOKEN` as an operator acknowledgement that a reverse
proxy is fronting authentication):

```bash
kaos-nlp tokenize doc.txt --lowercase --json          # word tokenization with spans
kaos-nlp segment doc.txt --mode sentences             # sentence segmentation (Punkt)
kaos-nlp compare "Robert" "Rupert" --algorithm jaro-winkler
kaos-nlp find "pattern" doc.txt --case-insensitive    # SIMD substring search
kaos-nlp index build corpus.txt --output idx.kncidx   # native persisted index
kaos-nlp search --index idx.kncidx "query terms"      # ranked search (BM25 default)
kaos-nlp hash doc.txt --algorithm ctph                # fuzzy hash
kaos-nlp duplicates ./corpus/ --threshold 0.5         # near-duplicate detection
kaos-nlp encode "Robert" --algorithm soundex          # phonetic encoding
kaos-nlp vocab build doc.txt --type frequency         # build vocabulary
kaos-nlp analyze doc.txt --json                       # text statistics report

kaos-nlp-serve            # MCP server, stdio transport
kaos-nlp-serve --http     # MCP server, streamable HTTP (operator-token gated)
```

Every command supports `--json` for machine-readable output. CLI search
reads both the native persisted index format (KNC) and legacy `.json`
bundles.

Note: 17 MCP tools are registered by `register_nlp_tools()`. Until
`0.1.0a2`, the `[mcp]` extra is reserved but unpopulated — manually run
`pip install kaos-core kaos-mcp` before using `kaos-nlp-serve`. Once
siblings publish to PyPI, `pip install kaos-nlp-core[mcp]` will cover
the full install. Until then `kaos-nlp-serve` exits with an actionable
install hint if `kaos-core` or `kaos-mcp` are missing.

## Compatibility & status

| Aspect | |
|---|---|
| **Python** | 3.13, 3.14 (informational matrix entries for 3.14t free-threaded and 3.15-dev). One `cp313-abi3` wheel per OS/arch covers all 3.13+ minors. |
| **OS** | Linux (manylinux + musllinux, x86_64 + aarch64), macOS arm64, Windows x86_64, Windows arm64. macOS x86_64 deliberately skipped (Apple ended Intel sales in 2023). |
| **Maturity** | Alpha. The public API is documented in `kaos_nlp_core.__all__`. |
| **Stability policy** | Pre-1.0: minor bumps may change behaviour. Every change is documented in [`CHANGELOG.md`](CHANGELOG.md). |
| **Test coverage** | 298 Rust unit tests + Python pytest suite. Round-trip offset tests cover ASCII, multi-byte Latin, CJK, and emoji. |
| **Type checker** | Validated with [`ty`](https://docs.astral.sh/ty/), Astral's Python type checker. |

## Companion packages

`kaos-nlp-core` is one of the packages in the
[Kelvin Agentic OS](https://kelvin.legal). The broader stack:

| Package | Layer | What it does |
|---|---|---|
| [`kaos-core`](https://github.com/273v/kaos-core) | Core | Foundational runtime, MCP-native types, registries, execution engine, VFS |
| [`kaos-content`](https://github.com/273v/kaos-content) | Core | Typed document AST: Block/Inline, provenance, views |
| [`kaos-mcp`](https://github.com/273v/kaos-mcp) | Bridge | FastMCP server, `kaos` management CLI, MCP resource templates |
| [`kaos-pdf`](https://github.com/273v/kaos-pdf) | Extraction | PDF → AST with provenance |
| [`kaos-web`](https://github.com/273v/kaos-web) | Extraction | Web extraction, browser automation, search, domain intelligence |
| [`kaos-office`](https://github.com/273v/kaos-office) | Extraction | DOCX / PPTX / XLSX readers + writers to AST |
| [`kaos-tabular`](https://github.com/273v/kaos-tabular) | Extraction | DuckDB-powered SQL analytics |
| [`kaos-source`](https://github.com/273v/kaos-source) | Data | Government + financial data connectors (Federal Register, eCFR, EDGAR, GovInfo, PACER, GLEIF) |
| [`kaos-llm-client`](https://github.com/273v/kaos-llm-client) | LLM | Multi-provider LLM transport |
| [`kaos-llm-core`](https://github.com/273v/kaos-llm-core) | LLM | Typed LLM programming (Signatures, Programs, Optimizers) |
| [`kaos-nlp-core`](https://github.com/273v/kaos-nlp-core) | Primitives (Rust) | High-performance NLP primitives |
| [`kaos-nlp-transformers`](https://github.com/273v/kaos-nlp-transformers) | ML | Dense embeddings + retrieval |
| [`kaos-graph`](https://github.com/273v/kaos-graph) | Primitives (Rust) | Graph algorithms + RDF/SPARQL |
| [`kaos-ml-core`](https://github.com/273v/kaos-ml-core) | Primitives (Rust) | Classical ML on the document AST |
| [`kaos-citations`](https://github.com/273v/kaos-citations) | Legal | Legal citation extraction, resolution, verification |
| [`kaos-agents`](https://github.com/273v/kaos-agents) | Agentic | Agent runtime, memory, recipes |
| [`kaos-reference`](https://github.com/273v/kaos-reference) | Sample | Reference module for module authors |

Packages depend on `kaos-core`; everything else is opt-in. Mix and match the
ones you need.

## Development

```bash
git clone https://github.com/273v/kaos-nlp-core
cd kaos-nlp-core
uv sync --group dev
uv run maturin develop --release
```

Install pre-commit hooks (recommended — they run the same checks as CI on
every commit, scoped to staged files):

```bash
uvx pre-commit install
uvx pre-commit run --all-files     # one-time full sweep
```

Manual QA commands (the same set CI runs):

```bash
cargo fmt --check
cargo clippy --no-default-features --all-targets -- -D warnings
cargo test --no-default-features --lib
uv run ruff format --check python/kaos_nlp_core tests
uv run ruff check python/kaos_nlp_core tests
uv run ty check python/kaos_nlp_core tests
uv run pytest tests/
```

## Build from source

```bash
uv build
uv pip install dist/*.whl
```

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for setup, quality gates, pull request expectations, and engineering
standards. By contributing you agree to follow the
[project conduct expectations](CODE_OF_CONDUCT.md) and certify the
[Developer Certificate of Origin v1.1](https://developercertificate.org/) —
sign every commit with `git commit -s`. Please open an issue before starting
on a non-trivial change so we can align on scope.

## Security

For security issues, **please do not file a public issue**. Report privately
via [GitHub Private Vulnerability Reporting](https://github.com/273v/kaos-nlp-core/security/advisories/new)
or email **security@273ventures.com**. See [SECURITY.md](SECURITY.md) for the
full disclosure policy.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Copyright 2026 [273 Ventures LLC](https://273ventures.com).
Built for [kelvin.legal](https://kelvin.legal).
