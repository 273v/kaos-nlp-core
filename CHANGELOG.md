# Changelog

All notable changes to `kaos-nlp-core` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Algorithms — fuzzy ranking helpers**: `most_similar(query, choices, …)`
  and `least_similar(...)` return the top-`k` candidates from a list under
  any of the existing similarity metrics. Parallel scoring via rayon for
  large candidate sets, deterministic tie-breaks, optional similarity
  threshold (floor for descending, ceiling for ascending). Backed by
  `core::algorithms::ranking::rank` and a new shared dispatch helper at
  `core::algorithms::dispatch` that the existing per-pair pyfunctions also
  consume (no more duplicate algorithm-name switches).
- **Documents — segment-level diff**: `documents.diff_documents(a, b, …)`
  segments two documents at sentence/paragraph/line granularity, scores
  every pair with a configurable similarity metric, and emits
  `SegmentChange { kind, left, right, score, … }` rows where `kind` is
  one of `unchanged`, `modified`, `moved`, `added`, `removed`. Greedy
  highest-score-first assignment with `match_threshold` /
  `modify_threshold` knobs; optional move detection via
  `move_distance_ratio`. Backed by `core::diff` and exposed through a
  new `kaos_nlp_core._rust.diff` PyO3 submodule.
- **Embedded Punkt model promoted to core**: the trained legal model
  (`DEFAULT_PUNKT_BYTES`) and a `default_tokenizer()` factory now live in
  `core::segmentation`, so non-Python callers can also use the bundled
  model via `include_bytes!` with no filesystem state. The PyO3 binding
  re-uses the same constant rather than embedding its own copy.

## [0.1.0a1] — 2026-05-05

First public alpha. High-performance NLP primitives for the Kelvin
Agentic Operating System: a pure-Rust core with PyO3 bindings, shipping
SIMD-accelerated string operations, multi-pattern matching, finite-state
transducers, sentence segmentation, BM25 retrieval, fuzzy hashing, and
typed Python wrappers throughout.

### Added

- **Rust core** (`rust/core/`) — pure-Rust implementations with no PyO3
  dependency. Tests run via `cargo test --no-default-features` (298
  passing at v0.1.0a1).
- **Algorithms** — Levenshtein, Hamming, Jaro-Winkler, longest common
  substring, edit-distance variants. SIMD fast paths via stringzilla
  on 4.6+.
- **Tokenizer** — fast Unicode-aware word/sentence tokenizer with byte-
  to-char offset translation. Multi-byte safe.
- **Segmentation** — Punkt sentence segmenter (bundled model
  `models/default.npkt.gz`, ~12 MB Apache-2.0 NLTK port).
- **Matching** — Aho-Corasick multi-pattern, FST-backed compact
  dictionaries with fuzzy lookup (Levenshtein automata via `fst[levenshtein]`).
- **Hashing** — CTPH (context-triggered piecewise hashing) via blake3
  for similarity-preserving fingerprints.
- **Search** — inverted-index BM25 retrieval with typed `ScoredDoc`
  results. Pickle-safe.
- **Retrieval** — typed wrappers for the BM25 + lexicon + FST search
  surface.
- **Lexicon** — compact, pickle-safe gazetteers and lookup tables.
- **Quality** — text quality heuristics (token ratios, Unicode
  block distribution).
- **Documents** — typed document-level wrappers.
- **Types** — `Token`, `Sentence`, `Span`, `ScoredDoc`, `DistanceResult`,
  and other typed dataclass results emitted at the PyO3 boundary.
- **MCP tools** — registration via `register_nlp_tools()` (planned
  for v0.1.0a2 once `kaos-mcp` publishes; the function is exported but
  errors with a clear message until then).
- **CLI** — `kaos-nlp` (administrative); `kaos-nlp-serve` (HTTP server,
  optional, requires `[mcp]` extra).
- Python 3.13 + 3.14 support; `requires-python = ">=3.13"`.

### Architecture

Three-layer design:
1. **Pure Rust core** — no PyO3 dep; testable standalone via
   `cargo test --no-default-features`.
2. **PyO3 bindings** — thin wrappers exposing Rust to Python; declare
   `module = "kaos_nlp_core._rust.<sub>"` for pickle support.
3. **Python package** — typed re-exports, dataclass conversions at the
   FFI boundary.

### Security

Pre-release security hardening (2026-05-05):

- **Dependency audit** — replaced `bincode` with `postcard 1.1`
  (RUSTSEC-2025-0141; the bincode team ceased development on all 1.x/2.x/3.x
  with no patched versions; postcard is the maintained successor recommended
  by the advisory). Bumped `fastbloom 0.11` → `0.17` with
  `default-features = false` (clears RUSTSEC-2026-0097 via transitive
  `rand 0.9.2`), `lru 0.14` → `0.18` (clears RUSTSEC-2026-0002 unsoundness),
  `icu_properties 2.0` → `2.2`, `rayon 1.10` → `1.12`. Removed unused
  crates: `compact_str`, `simsimd`, `bitvec`. `cargo audit` and
  `pip-audit --strict` both clean.
- **`.load()` hardening (F4)** — disk artifacts now carry a `KNC1` magic
  header + u16 format version; `load_bincode_from_path()` enforces a
  configurable size cap via `KAOS_NLP_MAX_LOAD_BYTES` (default 256 MiB)
  and rejects mismatched magic / version. Existing pre-0.1.0a1 binary
  artifacts are not readable by 0.1.0a1+ — this is a one-time format
  break tied to the move off bincode.
- **MCP HTTP server lockdown (F3)** — `kaos-nlp-serve --http` refuses to
  start without `KAOS_NLP_HTTP_TOKEN`; the `kaos-nlp-build-index` tool
  resolves `corpus_path` / `output_path` against `KAOS_NLP_WORKSPACE_ROOT`
  (defaults to CWD) and rejects path traversal + caps input file size.
- **Panic-to-exception (F5)** — `MinHashSignature::jaccard`,
  `MinHashIndex::insert` / `query_candidates`, `SimilarityMatrix::get_distance`,
  and `SparseTermMatrix::iter_document_terms` now return `Result` and
  surface `ValueError` at the Python boundary instead of panicking on
  invalid input.
- **Byte-vs-char offset audit** — every binding that returns text
  positions to Python uses `build_byte_to_char_table()` for O(n)
  conversion. ASCII fast path: byte == char offsets. Round-trip tests
  cover ASCII, multi-byte Latin (café), CJK (東京), emoji (😀).
- **Pickle safety** — every `#[pyclass]` declares
  `module = "kaos_nlp_core._rust.<sub>"`. Stateful classes use
  `__getstate__`/`__setstate__` + postcard.

### License

This release is the first to ship under the Apache License 2.0. Earlier
internal versions were proprietary. The bundled Punkt model (`models/
default.npkt.gz`) is Apache-2.0 from the NLTK distribution.

[Unreleased]: https://github.com/273v/kaos-nlp-core/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/273v/kaos-nlp-core/releases/tag/v0.1.0a1
