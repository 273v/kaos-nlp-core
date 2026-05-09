# Agent Guidance

## Scope

This file is the canonical repository-local instruction file for coding
agents working in `kaos-nlp-core`. Follow it for repository work, and use
the linked contributor and standards documents for the detailed rules.

Keep changes scoped. Preserve existing user changes, avoid unrelated
refactors, and treat generated artifacts, release metadata, lockfiles,
source code, tests, `README.md`, `CONTRIBUTING.md`, and
`docs/standards/` as out of scope unless the task explicitly asks for
them.

## Project Identity

- Distribution: `kaos-nlp-core`.
- Import package: `kaos_nlp_core`.
- Python CLI entry points: `kaos-nlp` and `kaos-nlp-serve`.
- Runtime shape: Rust core exposed to Python with PyO3 and maturin.
- Python support starts at 3.13. Wheels use `cp313-abi3` behavior so one
  wheel per supported OS/architecture works across CPython 3.13+ minors.

## Setup

Read [CONTRIBUTING.md](CONTRIBUTING.md) before non-trivial work. The
normal development setup is:

```bash
uv sync --group dev
uv run maturin develop --release
uvx pre-commit install
```

Use `uv` for Python environments, commands, builds, and tooling. Use
`maturin` for the Python extension build.

## Local Checks

Run the cheapest checks that cover the changed surface. For Python-only
changes, use:

```bash
uv run ruff format --check python/kaos_nlp_core tests
uv run ruff check python/kaos_nlp_core tests
uv run ty check python/kaos_nlp_core tests
uv run pytest tests/ --no-cov
```

For Rust/PyO3 changes, also use:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
uv run maturin build --release
```

Run `cargo audit` and `cargo deny check` when dependency, release,
packaging, or security-sensitive changes are involved. Use `ty`, not
mypy; `# type: ignore[...]` is not a substitute for a `ty` ignore.

## Architecture Rules

Follow the standards in:

- [Python design and architecture](docs/standards/python-design-and-architecture.md)
- [Rust/PyO3 design and architecture](docs/standards/rust-pyo3-design-and-architecture.md)
- [Code quality standards](docs/standards/code-quality-standards.md)
- [Tests, fixtures, and CI](docs/standards/tests-fixtures-ci.md)
- [Engineering process](docs/standards/engineering-process.md)

Keep tokenizer, matcher, similarity, hashing, segmentation, search, and
other CPU-bound NLP algorithms in the Rust core unless there is a clear
reason to keep logic in Python. Keep PyO3 bindings narrow: convert
inputs and outputs, map errors, and expose stable extension types. Keep
Python wrappers typed, documented, and thin; public users should import
from `kaos_nlp_core` wrapper modules rather than raw private extension
internals.

Preserve PyO3, `abi3-py313`, and maturin wheel behavior. Keep Cargo and
Python package metadata aligned. Convert Rust errors into appropriate
Python exceptions with safe context. Avoid `unsafe` unless the reason is
narrow, documented, and covered by tests. Release the GIL for heavy Rust
work when safe, and test both Rust internals and Python boundary
behavior.

## NLP Rules

Outputs for segmentation, tokenization, matching, hashing, search, and
similarity must be deterministic for the same inputs and configuration.
Be explicit about byte offsets versus character offsets, and test
Unicode text boundaries with non-ASCII inputs, including multi-byte
Latin, CJK, emoji, punctuation, and mixed-script text.

Bundled models, lexicons, and fixtures must have clear provenance,
compatible licensing, and deterministic loading behavior. Do not add
large or unknown-license data. Preserve stable public API, CLI, JSON,
pickle, index, and serialized artifact behavior unless the task includes
the required documentation, tests, changelog, and versioning work.

## Testing

New behavior needs tests through the real public entry point. Bug fixes
need regression tests. Rust algorithm changes should have Rust tests and
Python boundary tests. PyO3 conversion, exception mapping, serialization,
pickle, CLI output, and Unicode offset behavior need direct coverage
when changed.

Default tests must not require network access, credentials, local
services, or large downloads. Mark network, live, slow, and integration
tests according to [tests and fixtures standards](docs/standards/tests-fixtures-ci.md).

## Security

Never commit secrets, credentials, local environment files, private
tokens, build caches, virtual environments, or unreviewed generated
artifacts. Validate untrusted paths, URLs, archives, binary formats,
serialized inputs, model/data files, and CLI inputs early with bounded
limits. Keep error messages useful without exposing secrets, raw large
payloads, or sensitive filesystem details.

Security reports belong in the private reporting path described in
[SECURITY.md](SECURITY.md), not public issues.

## Commits, PRs, And Releases

Use conventional commits and DCO sign-off (`git commit -s`). Keep each
commit focused and stage only the intended files. Before committing,
fetch `origin`, rebase on `origin/main` when needed, run
`git diff --check`, and run relevant local checks.

PRs should explain what changed, why, how it was tested, and whether the
change affects public API, CLI behavior, package metadata, fixtures,
security posture, release artifacts, or changelog requirements. Release
work must follow [engineering process](docs/standards/engineering-process.md)
and the packaging gates in the standards.
