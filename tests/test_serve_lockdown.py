"""Regression tests for F3: kaos-nlp-serve --http lockdown and
build-index workspace-root + size-cap enforcement.

These tests cover both layers:

1. The CLI gate in `kaos_nlp_core.serve`: `--http` refuses to start
   without `KAOS_NLP_HTTP_TOKEN`. Stdio transport is unaffected.
2. The path/size guards in `kaos_nlp_core.tools` used by
   `kaos-nlp-build-index`: paths must resolve inside
   `KAOS_NLP_WORKSPACE_ROOT` (default CWD) and the corpus file is
   bounded by `KAOS_NLP_MAX_CORPUS_BYTES`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from kaos_nlp_core import serve, tools

# ─── 1. HTTP gate ──────────────────────────────────────────────────────────


def test_missing_kaos_core_or_mcp_emits_install_hint(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Base-install regression: missing kaos-core/kaos-mcp must surface the
    `pip install kaos-core kaos-mcp` hint *before* settings.py loads, not a
    raw ModuleNotFoundError from inside KaosNlpSettings.

    Why this matters: KaosNlpSettings inherits from kaos_core.config
    .ModuleSettings, so importing it would trigger the chained import
    error and bury the actionable message. The fix in serve.main() hoists
    the friendly try/except above the settings import.
    """
    # Simulate a base install (no kaos-core / kaos-mcp on path) by
    # masking the modules. Using a finder is overkill — a sentinel in
    # sys.modules that raises on access is enough.
    monkeypatch.setitem(sys.modules, "kaos_core", None)
    monkeypatch.setitem(sys.modules, "kaos_mcp", None)
    with pytest.raises(SystemExit) as exc_info:
        serve.main([])
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "pip install kaos-core kaos-mcp" in err
    assert "MCP server" in err


def test_http_without_token_exits(capsys: pytest.CaptureFixture[str]) -> None:
    """`--http` without KAOS_NLP_HTTP_TOKEN must refuse to start.

    Skips when kaos-mcp isn't installed: the friendly install-hint gate
    (regression-tested separately in
    test_missing_kaos_core_or_mcp_emits_install_hint) fires *first* and
    shadows the HTTP-token check. At v0.1.0a1 of the per-module repo,
    kaos-mcp is not on PyPI yet (Wave 3); the HTTP-gate tests re-run in
    0.1.0a2 once the sibling ships and `pip install kaos-nlp-core[mcp]`
    starts pulling kaos-mcp in.
    """
    pytest.importorskip("kaos_mcp")
    os.environ.pop("KAOS_NLP_HTTP_TOKEN", None)
    with pytest.raises(SystemExit) as exc_info:
        serve.main(["--http"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "KAOS_NLP_HTTP_TOKEN" in err
    assert "refuses to start" in err


def test_http_with_blank_token_exits(capsys: pytest.CaptureFixture[str]) -> None:
    """An empty-string token still trips the gate. Skips without kaos-mcp."""
    pytest.importorskip("kaos_mcp")
    os.environ["KAOS_NLP_HTTP_TOKEN"] = ""
    try:
        with pytest.raises(SystemExit) as exc_info:
            serve.main(["--http"])
        assert exc_info.value.code == 2
    finally:
        os.environ.pop("KAOS_NLP_HTTP_TOKEN", None)


# ─── 2. workspace-root + size cap on build-index ──────────────────────────


def test_workspace_root_default_is_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    from kaos_nlp_core.settings import KaosNlpSettings

    monkeypatch.delenv("KAOS_NLP_WORKSPACE_ROOT", raising=False)
    assert tools._workspace_root_from_settings(KaosNlpSettings()) == Path.cwd().resolve()


def test_workspace_root_honors_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from kaos_nlp_core.settings import KaosNlpSettings

    monkeypatch.setenv("KAOS_NLP_WORKSPACE_ROOT", str(tmp_path))
    assert tools._workspace_root_from_settings(KaosNlpSettings()) == tmp_path.resolve()


def test_resolve_within_root_accepts_inside(tmp_path: Path) -> None:
    inside = tmp_path / "corpus.txt"
    inside.write_text("hello world\n")
    assert tools._resolve_within_root(str(inside), tmp_path.resolve()) == inside.resolve()


def test_resolve_within_root_rejects_traversal(tmp_path: Path) -> None:
    """`/etc/passwd` must not resolve inside a tmp-path workspace."""
    with pytest.raises(ValueError, match="outside the workspace root"):
        tools._resolve_within_root("/etc/passwd", tmp_path.resolve())


def test_resolve_within_root_rejects_dotdot(tmp_path: Path) -> None:
    """`../` traversal escaping the root must be rejected."""
    sub = tmp_path / "sub"
    sub.mkdir()
    target = sub / ".." / ".." / "outside.txt"
    with pytest.raises(ValueError, match="outside the workspace root"):
        tools._resolve_within_root(str(target), sub.resolve())


def test_max_corpus_bytes_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from kaos_nlp_core.settings import KaosNlpSettings

    monkeypatch.delenv("KAOS_NLP_MAX_CORPUS_BYTES", raising=False)
    assert KaosNlpSettings().max_corpus_bytes == 256 * 1024 * 1024


def test_max_corpus_bytes_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from kaos_nlp_core.settings import KaosNlpSettings

    monkeypatch.setenv("KAOS_NLP_MAX_CORPUS_BYTES", "1024")
    assert KaosNlpSettings().max_corpus_bytes == 1024


def test_max_corpus_bytes_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid values now raise ValidationError at the settings boundary
    rather than silently falling back to the default. This is a
    deliberate behaviour change from the prior ``_max_corpus_bytes()``
    helper — pydantic's typed validation surfaces the bad input loudly."""
    from pydantic import ValidationError

    from kaos_nlp_core.settings import KaosNlpSettings

    monkeypatch.setenv("KAOS_NLP_MAX_CORPUS_BYTES", "not-a-number")
    with pytest.raises(ValidationError):
        KaosNlpSettings()
    monkeypatch.setenv("KAOS_NLP_MAX_CORPUS_BYTES", "-5")
    with pytest.raises(ValidationError):
        KaosNlpSettings()
