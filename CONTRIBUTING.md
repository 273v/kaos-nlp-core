# Contributing to kaos-nlp-core

Thank you for considering a contribution. This guide covers the local
development workflow, the QA gates a change must pass, and how we accept
contributions.

## Code of conduct

Participation in this project is governed by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please read
it before opening an issue or pull request.

## How to report issues

- **Bugs and feature requests** — open a GitHub issue using the templates
  in [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/).
- **Security vulnerabilities** — see [`SECURITY.md`](SECURITY.md). Do not
  file a public issue for a security report.

## Local development

`kaos-nlp-core` is a mixed Rust + Python project. You need a Rust
toolchain (1.83+) and a recent Python (3.13 or 3.14).

```bash
# Clone, then from the repo root:
uv venv && source .venv/bin/activate
uv sync --group dev
uv run maturin develop --release    # builds the PyO3 extension
uv run pytest tests/ -v             # 900+ Python tests
cargo test --no-default-features    # 290+ Rust core tests
```

The `--release` flag on `maturin develop` matters: debug builds of the
extension are 10–100× slower than release builds and many tests will
appear hung. There is no separate "build" step beyond `maturin develop`.

### Test fixtures

A few suites depend on downloaded fixtures (Gutenberg texts, HuggingFace
agreements, USC corpora):

```bash
./tests/fixtures/download_fixtures.sh                                   # public-domain texts
uv run --with huggingface_hub,tokenizers,datasets python tests/fixtures/download_hf_fixtures.py
```

Fixture downloads land in `tests/fixtures/` and are gitignored.

## QA gates (must pass before merge)

The CI workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
runs all of these. Run them locally before pushing:

```bash
# Rust
cargo fmt --check
cargo clippy --no-default-features --all-targets -- -D warnings
cargo test --no-default-features --lib

# Python (after maturin develop)
uv run ruff format --check python/kaos_nlp_core tests
uv run ruff check python/kaos_nlp_core tests
uv run ty check python/kaos_nlp_core tests
uv run pytest -m "not live and not network and not slow" tests/
```

`ty` is the required type checker — **not** mypy. `# type: ignore[…]`
comments are mypy syntax; ty needs `# ty: ignore[…]`.

The pre-commit hook configured in `.pre-commit-config.yaml` covers most
of the Python side automatically. Install it once per clone:

```bash
uvx pre-commit install
```

## Pull request checklist

- [ ] Branch from `main`, rebase before opening the PR.
- [ ] One logical change per PR. Bug fix + refactor → two PRs.
- [ ] Tests added or updated. New public APIs require a test that
      exercises them through Python (mocked-only tests are not
      sufficient evidence — see "Testing standards" below).
- [ ] CHANGELOG.md updated under `[Unreleased]` with a one-line entry.
- [ ] Commit messages follow the existing style (Conventional Commit
      prefix `feat(scope): …` / `fix(scope): …` / `docs: …` / etc).
- [ ] DCO sign-off on every commit: `git commit -s`.

## Testing standards

- **Live integration tests are the quality bar.** Mocked unit tests
  document an API; they do not prove correctness. New features that
  touch a real provider, real corpus, or real algorithm should have a
  test that exercises the production path end-to-end.
- **Assertions verify behavior, not envelope shape.** `assert
  len(results) > 0` is rarely enough. Verify that the right doc ranks
  first, that a known query has known matches, that round-trip
  serialization preserves content.
- **Test fixtures must be real.** PIL/numpy-generated images, hand-
  crafted PDFs, public-domain Gutenberg texts. Never use `b"hello"` as
  a stand-in for real content.
- **Multi-byte text round-trips** — every Rust binding that returns
  text offsets is required to have a round-trip test covering ASCII,
  multi-byte Latin (`café`), CJK (`東京`), and emoji (`😀`). See the
  byte-offset section of `CLAUDE.md`.

## Architectural conventions

Detailed in `CLAUDE.md`. Highlights:

- **Three-layer design** — pure Rust core (`rust/core/`, no PyO3),
  PyO3 bindings (`rust/bindings/`), Python re-exports
  (`python/kaos_nlp_core/`).
- **Rust core uses byte offsets internally; bindings convert** to char
  offsets before returning to Python. Use
  `build_byte_to_char_table()` for the O(n) lookup.
- **Use `ahash` instead of `std::collections::HashMap`** for
  performance-sensitive maps.
- **All `#[pyclass]` types declare `module = "kaos_nlp_core._rust.<sub>"`**
  — required for pickle support.
- **PyO3 wrappers prefix `Py`**; Python re-exports drop it
  (`PyFstSet` → `FstSet`).

## License

By contributing, you agree that your contributions are licensed under
the [Apache License 2.0](LICENSE) and that you have the right to license
them under that license. The DCO sign-off (`-s`) on each commit is your
attestation.
