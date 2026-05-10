# Changelog

All notable changes to `kaos-nlp-core` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **uv.lock is now tracked in git.** Previously gitignored at v0.1.0a1
  because the ``[mcp]`` optional extra (and the ``kaos-mcp`` dev
  dependency) referenced a sibling not yet on PyPI; ``uv lock``
  couldn't resolve them. ``kaos-mcp`` shipped (0.1.0a2), so the
  original gating reason no longer applies. Tracking the lockfile
  gives reproducible local dev environments, lets Dependabot surface
  sibling-version bumps as PRs, and makes the supply-chain pin set
  publicly auditable. Mirrors the org-wide convention being adopted
  across all 16 kaos-* repos.
### Security

- **bandit + vulture now run in both pre-commit and CI.** The
  ``.pre-commit-config.yaml`` gains two new hooks (bandit static
  security scan + vulture dead-code scan), mirrored by jobs in
  ``security.yml`` so the scan is publicly visible on every PR.
  Bandit skip list is justified inline per audit
  (``B101,B404,B603,B607``); vulture runs at ``--min-confidence
  100`` with a shared ``--ignore-names`` list for framework
  callbacks / signal handlers / OAuth field names that vulture
  can't infer from the import graph alone. Both hooks currently
  pass clean. Mirrors the rollout pattern from kaos-core.

### Removed

- **musllinux wheels (Alpine Linux / musl libc)** dropped from the
  release.yml matrix. ``kaos_nlp_core-*-cp313-abi3-musllinux_1_2_x86_64.whl``
  and ``-aarch64.whl`` will not ship on the next release. Rationale:
  family-consistency. ``kaos-nlp-transformers`` can't ship musllinux
  (ort's ``download-binaries`` feature pulls Microsoft's official
  libonnxruntime which is glibc-only); shipping musllinux for
  ``kaos-nlp-core`` while the downstream ML sibling can't install
  there creates a fragmented Alpine user experience. The 0.1.0a2
  release retains its musllinux wheels on PyPI; Alpine users requiring
  this package standalone should pin ``kaos-nlp-core==0.1.0a2`` until
  the ML runtime constraint is lifted.

## [0.1.0a2] — 2026-05-07

Audit-driven hardening release covering eight findings (KNC-001 …
KNC-008) from the independent `kaos-modules/docs/audit-01/kaos-nlp-core.md`
pass, plus three substantive feature additions queued during the
intervening cycle.

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

### Fixed

- **README Quick Start now actually runs (KNC-A0).** The 0.1.0a1 example
  referenced a nonexistent `tokenizer.tokenize_spans()` and treated
  `tokenize_words()` output (`list[str]`) as objects with a `.text`
  attribute. Replaced with a verified-runnable block that surfaces the
  raw-string vs. rich-`TokenSpan` shape distinction explicitly, with
  literal-output comments captured from a real run.
- **Default `pytest tests/` no longer hits the live Federal Register API
  (KNC-004).** Tests reorganized into `tests/unit/` (offline; default
  testpath) and `tests/integration/` (network/live, opt-in). Added
  `addopts = [..., "-m", "not network and not live"]` and updated
  `[tool.pytest.ini_options].testpaths` and `[tool.ty.src].exclude` to
  match. 1696 unit tests pass; live test only collected with explicit
  `pytest tests/integration -m "network or integration"`.
- **`models/__init__.py` declares `__all__` (KNC-005).** Matches the
  KAOS rule that every `__init__.py` declare its public surface (here:
  empty, since the package only bundles model bytes via
  `importlib.resources`).
- **Tool count metadata corrected to 17 (KNC-007).** `tools.py` module
  + `register_nlp_tools` docstrings and the upstream
  `kaos-modules/docs/architecture.md` tool list previously reported
  10–11 tools; the registry has always contained 17.

### Changed

- **PyO3 typed-pyclass hot-path standard codified (KNC-001 + KNC-002).**
  `rust/bindings/quality.rs` (`PyCharClassCounts`, `PyWordStats`,
  `PyQualityRaw`) and `rust/bindings/spans.rs` (`PyTokenSpan`,
  `PyMatchSpan`, `PyPatternMatchSpan`, `PyRegexMatchSpan`,
  `PyFstSearchResult`, `PyScoredDoc`, `PyPostingEntry`, `PySegmentHit`)
  now declare `skip_from_py_object` to silence PyO3 0.28's deprecation
  diagnostics for cloned `#[pyclass]` types. `cargo clippy --all-targets
  -- -D warnings` (default features) passes clean again. The dict-at-the-
  boundary-plus-Python-dataclass pattern remains the default; the
  hot-path native-pyclass exception is now formally documented in
  `kaos-modules/docs/oss/30-rust-packaging/pyo3-typed-api.md`.
- **Rust crate root hardened (KNC-008).** Added the documented warn-lint
  set (`missing_docs`, `rust_2018_idioms`, `rust_2021_compatibility`,
  `unreachable_pub`, `unused_qualifications`) at `rust/lib.rs`; the six
  `clippy::pedantic` lints are wired with `#![allow(...)]` + a
  `TODO(KNC-008-followup)` marker for a future cleanup pass. Tightened
  ~14 `pub fn register_module` declarations across `rust/bindings/*.rs`
  to `pub(crate)` (correct visibility — they're consumed only by
  `lib.rs`). Audited every `#[pyclass]` for `Sync` compliance and added
  `#[pymodule(gil_used = false)]` to opt into PyO3's free-threaded
  contract for Python 3.13t.

### Documentation

- **`[mcp]` extra deferral note (KNC-003).** `kaos-nlp-serve` already
  prints an actionable install hint when `kaos-core` / `kaos-mcp` are
  missing; serve.py's import-order is correct (the kaos-core /
  kaos-mcp guard precedes `from kaos_nlp_core.settings import …`). The
  README and this changelog now spell out that the `[mcp]` extra
  remains unpopulated until `kaos-mcp` ships to PyPI (F009 lesson #4 in
  `kaos-modules/docs/oss/00-overview/decisions.md` — `uv lock` refuses
  to resolve declared-but-unresolvable extras).
- **Cargo URL convention clarified (KNC-006).** Per-module-repo
  manifests legitimately point at per-module URLs
  (`https://github.com/273v/kaos-nlp-core`, `https://kelvin.legal`).
  The upstream `cargo-conventions.md` has been updated to mark this as
  the correct shape under D015 ("two-shape pyproject"); no changes to
  this repo's `Cargo.toml` were required.

### Known limitations

- **`kaos-nlp-serve` still requires a manual sibling install.** The
  `[mcp]` optional-dependency extra is reserved but unpopulated because
  `kaos-mcp` has not yet published to PyPI. Run
  `pip install kaos-core kaos-mcp` manually before invoking
  `kaos-nlp-serve`. The extra will be populated in the release that
  follows kaos-mcp's first PyPI cut.

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

[Unreleased]: https://github.com/273v/kaos-nlp-core/compare/v0.1.0a2...HEAD
[0.1.0a2]: https://github.com/273v/kaos-nlp-core/releases/tag/v0.1.0a2
[0.1.0a1]: https://github.com/273v/kaos-nlp-core/releases/tag/v0.1.0a1
