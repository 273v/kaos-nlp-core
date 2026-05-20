# Changelog

All notable changes to `kaos-nlp-core` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]


## [0.1.0rc1] — 2026-05-20

### Changed — WU-J of 0.1.0 GA plan

- Release candidate cut per WU-J of the 0.1.0 GA plan. Freezes the
  public Python + Rust API surface ahead of GA. No source changes
  relative to 0.1.0a9; this release exists to raise the kaos-core
  dev-group pin floor to the rc track and signal API freeze to
  downstream consumers.
- Pin floor raised to `kaos-core>=0.1.0rc1,<0.2` across `kaos-*` deps
  in the dev group. The `<0.2` ceiling protects against legacy
  `0.2.0a*` lines (e.g. kaos-nlp-transformers) leaking into resolution.
- Cargo crate version bumped to `0.1.0-rc.1`; maturin emits the
  PEP 440-normalized wheel metadata `0.1.0rc1`.

### Verified
- Rust QA: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test --no-default-features` (677 tests passed).
- Python QA: `ruff format --check`, `ruff check`, `ty check`,
  `pytest -m "not live and not network and not slow and not integration"`
  (1696 passed, 67 skipped — fixture-gated).


## [0.1.0a9] - 2026-05-20

### Changed — kaos-core 0.1.0a12 catch-up (WU-D.1)

- Layer 1 catch-up release per the 0.1.0 GA plan (WU-D.1). No source
  changes; Rust API is already aligned with the kaos-core 0.1.0a10 URI
  redesign + 0.1.0a12 capability type contract (this package's Rust
  core has no kaos-core dependency at the boundary). Dev-group
  `kaos-core` pin refreshed to 0.1.0a12 via `uv lock` so contributors
  test against the same floor downstream Layer 5 consumers depend on.
- Linux x86_64 `maturin develop --release` build is green; CI matrix
  builds macOS arm64 + Windows wheels on tag push.

### Verified
- Rust QA: `cargo fmt`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test --no-default-features` (677 tests passed).
- Python QA: `ruff format --check`, `ruff check`, `ty check`,
  `pytest -m "not live and not network and not slow and not integration"`
  (2054 passed, 68 skipped — fixture-gated, 20 deselected).


## [0.1.0a8] - 2026-05-16

### Fixed — `HierarchicalChunker` depth-0 / depth-2

- **`HierarchicalChunker`** now records `metadata["over_budget"]`
  on every depth-0 chunk (boolean — `True` when the section's
  token count exceeds the chunker's `max_tokens`). Callers that
  want a coarse table-of-contents view can filter for
  `depth == 0 and not over_budget` without recomputing the budget
  comparison. The chunk's `metadata["max_tokens"]` is also written
  so downstream consumers see the budget the section was scored
  against.
- **`HierarchicalChunker`** no longer emits depth-2 chunks that
  duplicate their depth-1 parent. The depth-2 fallback runs the
  injected `SentenceChunker` on each oversize paragraph sub-chunk;
  when the resulting split returns exactly one chunk (the common
  case where `ParagraphChunker` already subdivided via the same
  `SentenceChunker`), depth-2 is suppressed. Sentence-level
  subdivision still fires whenever the caller supplies a tighter
  `sentence_chunker=SentenceChunker(max_tokens=...)`, which is
  where the additional granularity actually exists.
- The change is observable in `chunker-scale-*-hierarchical.json`
  benchmarks as a small drop in total chunk count (USC: 21713 →
  21697; EDGAR: 39353 → 39245) reflecting the no-op depth-2
  duplicates that were previously emitted. Public API and offset
  round-trip invariants are unchanged.

Regression coverage in
`tests/test_chunking_chunkers.py::TestHierarchicalChunker` —
`test_depth_zero_over_budget_flag`,
`test_depth_zero_within_budget_flag`,
`test_depth_two_suppressed_when_sentence_split_is_noop`, and the
revised `test_sentence_subdivision_when_paragraph_oversize` (which
now injects a tighter `SentenceChunker` to exercise the legitimate
depth-2 path).


## [0.1.0a7] - 2026-05-15

### Added — `kaos_nlp_core.content_type` (PRD PR 4)

- **`kaos_nlp_core.content_type.detect(bytes) -> ContentTypeResult`** —
  magic-byte content classifier feeding the kaos-agents per-turn
  planner's `corpus_kinds` Signature input. Returns a frozen
  `ContentTypeResult(mime_type, extension, group)` where `group` is
  one of a fixed enumeration the planner's few-shot examples are
  written against: `pdf`, `office-docx`, `office-xlsx`,
  `office-pptx`, `office-doc`, `office-xls`, `office-ppt`, `image`,
  `audio`, `video`, `archive`, `email`, `html`, `text`, `font`,
  `binary`, `unknown`.
- Backed by the `infer` Rust crate (MIT, ~80KB, zero runtime deps,
  350+ file types). Pure magic-byte sniffing — no ML model, no
  ONNX runtime, no impact on wheel size or CI build times. Covers
  the kelvin-legal upload set (PDF / DOCX / XLSX / PPTX / JPEG /
  PNG / ZIP / EML / ...) at effectively 100% accuracy on samples
  ≥256 bytes. Google Magika 1.0 (ML, ONNX) was considered but
  deferred — see PRD PR 4 §7 ("References") for the comparison.
- Rust core: `rust/core/content_type/mod.rs` (5 unit tests).
  PyO3 binding: `rust/bindings/content_type.rs`. Python facade:
  `python/kaos_nlp_core/content_type/__init__.py` (typed frozen
  dataclass + `is_known` property). Type stubs:
  `python/kaos_nlp_core/_rust/content_type.pyi`. Cross-boundary
  tests: `tests/test_content_type.py` (14 tests).

Motivated by `kaos-modules/docs/internal/dynamic-tool-planning-prd.md`
§4 (round-2 decision #7) — a session that uploads "10MB PDF + a CSV
+ an HTML snapshot" should surface `corpus_kinds = ["pdf",
"spreadsheet", "html"]` to the planner so it can rationalize its
ceiling around the actual document mix.

The classifier is purely additive: no existing public surface
changes. kaos-content uploaders + the single-user-chat backend
opt in when they want corpus tagging.


## [0.1.0a6] - 2026-05-15

### Added

- **Pre-normalised cosine fast paths** in `kaos_nlp_core.similarity`:
  `cosine_normalized`, `cosine_one_to_many_normalized`, and
  `cosine_adjacent_normalized`. Skip the per-vector `‖a‖²` / `‖b‖²`
  work and the rsqrt finalisation; pure dot + clamp. Callers that
  feed unit-norm embeddings (kaos-nlp-transformers' SemanticChunker
  + ExtractiveRanker) should use these for the production hot path.
- New kernel-layer entry points exposed for downstream Rust
  consumers: `cosine_components_f32`, `dot_f32`, `norm_sq_f32`,
  `cosine_f32_normalized`, `finalize_cosine_f32`, and the
  `cosine_{one_to_many,adjacent}_{,normalized}_into` write-into-
  slice variants. See `docs/design-similarity-simd.md`.

### Changed

- **Hand-rolled f32 SIMD kernels with runtime ISA dispatch** replace
  the previous auto-vectorised `f64`-accumulator design. New layout
  at `rust/core/similarity/kernels.rs`:
  - **AVX-512F + FMA** path (16-wide `f32`) — Intel Skylake-X /
    Sapphire Rapids, AMD Genoa.
  - **AVX2 + FMA** path (8-wide `f32`) — Intel Haswell+, AMD Zen+,
    the modal consumer + cloud x86 hardware.
  - **NEON** path (4-wide `f32`) — Apple Silicon, ARM Linux,
    Windows ARM64.
  - **Scalar** fallback for every other target.
  Dispatch is feature-detected at first call and cached in a
  `OnceLock<u8>`; per-call cost is a `Relaxed` atomic load.
- **Fused single-pass cosine kernel** — every public cosine entry
  point now computes `(dot, ‖a‖², ‖b‖²)` in one SIMD loop over the
  data (was three separate passes). Finalisation uses an `rsqrt`
  estimate with one Newton-Raphson refinement (AVX2/AVX-512) or two
  (NEON), borrowed from NumKong's `nk_angular_normalize_*` design.
- **`cosine_one_to_many` runs the full row loop inside the ISA
  kernel** — one dispatch decision per call, one query-norm
  computation per call, two FMAs per element per row. Previously
  it dispatched per row and recomputed the query norm each iteration.
- **MSRV bumped to 1.89** to enable stable AVX-512 intrinsics on
  x86_64. All shipping wheels build with Rust ≥ 1.93 on current CI
  toolchains; the bump only affects callers who consume the source
  crate directly with an older toolchain.

### Perf envelope (Intel i7-12700K, AVX2+FMA, no AVX-512)

Measured vs `numpy.dot` (BLAS sgemv) at the production callsites
(SemanticChunker + ExtractiveRanker, unit-norm rows):

| Shape                        | Generic path | Pre-normalised |
|------------------------------|-------------:|---------------:|
| `n=50,   dim=384`            |          n/a |   **4.0× win** |
| `n=200,  dim=384`            |          n/a |   **2.5× win** |
| `n=1000, dim=384`            |        0.54× |        1.04×   |
| `n=1000, dim=768`            |  **48× win** | **242× win**¹  |
| `n=10000, dim=384`           |   **15× win**|   **6.8× win** |

¹ `n=1000 d=768` cell has high variance — numpy's BLAS shape
heuristic occasionally falls back to the un-tiled `sgemv` here.
Median over 30 trials.

The `cosine_adjacent_normalized` path used by SemanticChunker is
**1.5–7.4× faster than numpy at every tested shape** (`n ∈ [10, 1000]`,
`dim ∈ {256, 384, 768}`), no losses. See
`docs/benchmarks/similarity-cosine-adjacent-normalized.json` and the
companion `*-one-to-many-normalized.json` for the full grid; raw
numbers are committed alongside the source so release-over-release
drift is visible.

### Documentation

- `docs/design-similarity-simd.md` documents the kernel layout,
  the per-function NumKong inspirations (file + symbol map), the
  dispatch flag, the numerical-stability rationale, and what we
  intentionally did **not** port.
- `NOTICE` updated with NumKong attribution
  (https://github.com/ashvardanian/NumKong; Apache-2.0). The Rust
  port is clean-room; no NumKong source is bundled or linked.


## [0.1.0a5] - 2026-05-15

### Changed

- **Dropped the `numkong = "7.6"` Cargo dependency** and ported its
  *design* (f32 inputs, f64 accumulators, pairwise-style reduction
  via 8 parallel lanes) into a portable in-house kernel at
  `rust/core/similarity/kernels.rs`. The new kernel auto-vectorises
  to AVX2 (x86_64) and NEON (aarch64) at `-C opt-level=3` and runs
  on every platform we ship a wheel for, including
  `aarch64-pc-windows-msvc` and `x86_64-pc-windows-msvc` where
  numkong's vendored C broke (`immintrin.h` rejected on MSVC ARM64
  and a `DllMain` symbol that collided with `stringzilla`'s).
- Numerical contract unchanged: cosine clipped to `[-1, 1]`, zero-
  norm vectors yield 0.0, f32 inputs accumulated in f64. The 8-lane
  reduction gives an `O(log N)` error bound comparable to Neumaier
  compensation for the dimensionalities we serve (256-1536).
- `cosine_one_to_many` now precomputes the query norm once per call,
  saving the per-row redundant norm computation that numkong's
  serial path was doing internally. Modest perf win on long batches.

### Removed

- `numkong` Cargo dep + the doc-comment claims about NumKong's
  AVX-512 / SVE / SME runtime dispatch. The replacement covers AVX2
  + NEON via auto-vec, which is the highest-supported ISA on
  every wheel target except niche AVX-512 servers (where we recover
  the throughput when LLVM ever auto-vec's 256-bit loops on those
  cores; benchmarks remain in `docs/benchmarks/similarity-*.json`
  for the new perf envelope).


## [0.1.0a4] - 2026-05-15

### Added

- **Rust chunking kernels** (`rust/core/chunking/`) — the greedy
  unit packer that drives every concrete chunker
  (Fixed / Sentence / Paragraph / Section / Hierarchical) is now in
  Rust as `pack_units(starts, ends, token_counts, max_tokens,
  overlap_units)`, returning five parallel ``uint32`` arrays
  describing the resulting groups. A second kernel
  ``semantic_pack(..., adj_sim, drop_threshold)`` covers
  ``kaos_nlp_transformers.SemanticChunker``'s budget+topic-shift
  scan. Both run with the GIL released via ``py.detach``. The
  Python wrappers in ``kaos_nlp_core.chunking._pack`` (and
  ``SemanticChunker._pack``) marshal `_Unit` lists / numpy
  embeddings into the CSR-style input format, then materialise
  Chunks from the Rust groups. Behaviour is bit-identical to the
  prior pure-Python loop (244 chunker tests + 26 SemanticChunker
  tests + scale tests verify).
- **Rust aggregation kernels** (`rust/core/aggregation/`) — all six
  primitives (`vote`, `majority`, `union`, `intersection`,
  `weighted_single` / `weighted_multi`, `max_score_single` /
  `max_score_multi`) now run in Rust. The Python wrappers in
  ``kaos_nlp_core.aggregation`` intern string label names to
  ``u32`` ids preserving first-appearance order (so the
  "lowest first_seen wins" tiebreak maps to "lowest id wins" in
  Rust), then dispatch through a CSR-style ragged-array
  representation. Kernels are pure functions of their inputs and
  free of hash-randomization side-channels; the existing
  determinism test suite (`test_aggregation_determinism.py`)
  passes unchanged. Benchmark report at
  ``docs/benchmarks/aggregation-rust-vs-python.json``.
- **`numpy` declared as a runtime dependency** — the dense similarity,
  chunker, and aggregation Rust bindings all marshal through
  ``numpy.ndarray``; the top-level ``kaos_nlp_core`` package
  unconditionally imports them. Declares ``numpy>=1.26`` in
  ``[project.dependencies]`` to make the existing transitive
  requirement explicit. Affects downstream wheels (kaos-llm-core,
  kaos-nlp-transformers, etc.) that consume kaos-nlp-core's
  public surface — numpy will be pulled in automatically.
- **`kaos_nlp_core.similarity`** — new module exposing
  hardware-accelerated dense-vector primitives. Backed by the
  vendored NumKong C kernels (successor to SimSIMD; Apache-2.0)
  routed through a thin Rust+PyO3 layer at
  ``rust/core/similarity/`` and ``rust/bindings/similarity.rs``.
  Public surface:
  - ``cosine(a, b) -> float`` — single-pair cosine similarity.
  - ``cosine_one_to_many(query, matrix) -> ndarray`` — one query vs
    every row.
  - ``cosine_adjacent(matrix) -> ndarray`` — cos of every adjacent
    row pair (semantic chunker).
  - ``top_k_cosine(query, matrix, k) -> TopKResult`` — heap-based
    selection with ascending-index tiebreak.
  - ``mmr_select(matrix, relevance, k, lambda_) -> MMRResult`` —
    Maximal Marginal Relevance.
  - ``l2_normalize_in_place(vector) -> bool`` — unit-norm in place.
  - `TopKResult` and `MMRResult` typed `@dataclass(frozen=True,
    slots=True)` result containers.
  - Runtime dispatch (AVX-512 / AVX2 / NEON / SVE / scalar) chosen
    by NumKong's CPU-feature probe. f32 accumulates in f64 with
    Neumaier-Kahan-Babuška compensation; cosine results clipped to
    ``[-1, 1]``; zero-norm vectors return ``0.0``; ties broken by
    ascending row index.
  - Benchmarks under ``docs/benchmarks/similarity-*.json`` track
    Rust-vs-numpy across the dim x corpus-size grid. Real wins on
    long workloads (17-67x on 1000-row x 768-d cosine + MMR); numpy
    competitive or faster on small dim=256 / n=100 cases where PyO3
    boundary overhead dominates.

### Documentation

- **Use + AI-authorship disclosure** added to the README. Notes
  that `kaos-nlp-core` is fully deterministic and local (no LLM
  calls, no network) but that downstream consumers may transmit
  derived text to LLM providers. AI-assisted authorship disclosure
  (Claude, Anthropic; human-reviewed) added.

### Fixed

- **`SentenceChunker` allocates less metadata per chunk.** Previously
  each `_Unit` carried a per-sentence
  `metadata={"sentence_confidence": ...}` dict, which the packer
  then copied into a merged dict, which ``Chunk.__post_init__``
  then re-wrapped in ``MappingProxyType(dict(...))`` — three
  allocations per chunk. ``sentence_confidence`` is never consumed
  downstream (verified by grep across all three repos), and the
  underlying ``Segment.confidence`` value remains available on the
  raw Punkt output. Now: the unit metadata is omitted (no
  per-sentence dict), and the shared
  ``_pack._EMPTY_METADATA_DICT`` sentinel replaces the
  ``metadata or {}`` allocation when no metadata is supplied. Net:
  measurably fewer dict allocations on the SentenceChunker hot
  path; previously the 1000-USC-doc scale run reported a 253 MB
  RSS delta.

- **`vote`, `majority`, and `weighted` aggregation primitives are now
  deterministic across processes.** Previously they iterated each
  chunk's labels via ``for name in set(chunk_labels)``, which uses
  Python's hash-randomized set iteration order — so the ``first_seen``
  map (and therefore the documented "ties broken by order of first
  appearance" tiebreak) silently depended on ``PYTHONHASHSEED`` and
  could disagree across processes. Replaced ``set(chunk_labels)``
  with ``dict.fromkeys(chunk_labels)`` in all three functions
  (`/python/kaos_nlp_core/aggregation/__init__.py`); the dedup is
  identical but insertion order is preserved. New regression test
  suite in `tests/test_aggregation_determinism.py` runs each
  function under six different ``PYTHONHASHSEED`` values in fresh
  subprocesses and asserts identical winners.

### Added

- **`kaos_nlp_core.chunking`** — new module for document chunking
  primitives. Phase 0 + Phase 1 of the summarization/classification
  cross-module plan.
  - **Foundation types** (Phase 0):
    - `Chunk` — frozen, slotted dataclass holding ``text``, ``start``,
      ``end``, ``parent_id``, deterministic ``chunk_id``, optional
      ``token_count``/``depth``, and read-only ``metadata``.
    - `Chunker` — runtime-checkable protocol for callables that split
      a source string into ``list[Chunk]``.
    - `compute_chunk_id` — deterministic SHA-256-based identifier
      function (32-char hex digest of
      ``parent_id|start|end|text``).
    - `validate_chunk_offsets` — round-trip helper that asserts
      ``source[chunk.start:chunk.end] == chunk.text``.
  - **Concrete chunkers** (Phase 1):
    - `FixedTokenChunker` — sliding character-window chunker sized
      by an approximate token budget with optional overlap.
    - `SentenceChunker` — Punkt-backed sentence packer that respects
      sentence boundaries up to the token budget; oversize sentences
      emit alone.
    - `ParagraphChunker` — paragraph-aware packer that falls back to
      :class:`SentenceChunker` for oversize paragraphs.
    - `SectionChunker` — detects enumerator-bearing sections
      (``1.``, ``§``, ``Article``, ``# Heading``, etc.) via
      :func:`segment_lines` + :func:`parse_enumerator`, then chunks
      within each section.
    - `HierarchicalChunker` — multi-depth chunker emitting
      section-level (``depth=0``), paragraph-level (``depth=1``),
      and sentence-level (``depth=2``) chunks in a single flat
      list, each tagged with its ``level`` and ``section_*``
      metadata for tree reconstruction.
    - `default_token_counter` — public ``ceil(chars / 4)``
      approximation; pluggable per chunker via
      ``token_counter=...``.
  - `chunking` is re-exported from ``kaos_nlp_core`` and listed in
    ``__all__``.
  - Determinism, round-trip offsets (``source[c.start:c.end] ==
    c.text``), Unicode/CJK/emoji round-trip, ``parent_id``
    propagation, and ordering are property-checked across all
    chunkers in ``tests/test_chunking_chunkers.py``.
  - Benchmark harness in ``tests/bench_chunking.py``.

- **`kaos_nlp_core.aggregation`** — new module of pure, deterministic
  label-aggregation primitives consumed by the
  :class:`~kaos_llm_core.composition.aggregate.Aggregator` strategy
  classes in ``kaos-llm-core``. Phase 2 of the
  summarization/classification cross-module plan.
  - `vote(per_chunk)` — plurality, ties broken by first appearance.
  - `majority(per_chunk, threshold=0.5)` — threshold-gated majority.
  - `union(per_chunk)` — multi-label any-chunk union.
  - `intersection(per_chunk)` — multi-label every-chunk intersection.
  - `weighted(per_chunk, weights, threshold, multi)` — weighted
    aggregation supporting both single-label and multi-label modes.
  - `max_score(per_chunk_scores, multi, threshold)` — aggregate by
    the highest single-chunk score per label.
  - ``aggregation`` is re-exported from ``kaos_nlp_core`` and listed
    in ``__all__``.

## [0.1.0a3] — 2026-05-11

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

[Unreleased]: https://github.com/273v/kaos-nlp-core/compare/v0.1.0a3...HEAD
[0.1.0a3]: https://github.com/273v/kaos-nlp-core/compare/v0.1.0a2...v0.1.0a3
[0.1.0a2]: https://github.com/273v/kaos-nlp-core/releases/tag/v0.1.0a2
[0.1.0a1]: https://github.com/273v/kaos-nlp-core/releases/tag/v0.1.0a1
