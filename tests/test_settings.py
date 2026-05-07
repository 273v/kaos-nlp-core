"""Tests for ``KaosNlpSettings`` and the per-request override pattern.

Mirrors the pattern that closed kaos-graph audit follow-up #3: every
``KAOS_NLP_*`` env var resolves through a typed ``ModuleSettings`` class,
and MCP request contexts can override on top of env defaults via
``_meta.kaos_config``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_defaults_when_no_env_or_context() -> None:
    from kaos_nlp_core.settings import KaosNlpSettings

    settings = KaosNlpSettings()
    assert settings.workspace_root is None
    assert settings.max_corpus_bytes == 256 * 1024 * 1024
    assert settings.lexicon_path is None
    assert settings.http_token is None


def test_env_var_lookup_uses_kaos_nlp_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """All four env vars resolve through ``env_prefix='KAOS_NLP_'``."""
    from kaos_nlp_core.settings import KaosNlpSettings

    monkeypatch.setenv("KAOS_NLP_WORKSPACE_ROOT", "/tmp/sandbox")
    monkeypatch.setenv("KAOS_NLP_MAX_CORPUS_BYTES", "12345")
    monkeypatch.setenv("KAOS_NLP_LEXICON_PATH", "/tmp/lexicon.bin")
    monkeypatch.setenv("KAOS_NLP_HTTP_TOKEN", "ops-ack-secret")

    settings = KaosNlpSettings()
    assert settings.workspace_root == "/tmp/sandbox"
    assert settings.max_corpus_bytes == 12345
    assert settings.lexicon_path == "/tmp/lexicon.bin"
    assert settings.http_token is not None
    assert settings.http_token.get_secret_value() == "ops-ack-secret"


def test_negative_max_corpus_bytes_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Field(ge=0)`` blocks negative caps from being honored."""
    from pydantic import ValidationError

    from kaos_nlp_core.settings import KaosNlpSettings

    monkeypatch.setenv("KAOS_NLP_MAX_CORPUS_BYTES", "-100")
    with pytest.raises(ValidationError):
        KaosNlpSettings()


def test_explicit_overrides_take_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    from kaos_nlp_core.settings import KaosNlpSettings

    monkeypatch.setenv("KAOS_NLP_MAX_CORPUS_BYTES", "999")
    settings = KaosNlpSettings(max_corpus_bytes=12345)
    assert settings.max_corpus_bytes == 12345  # explicit wins


def test_from_context_threads_kaos_config_overrides() -> None:
    """``KaosNlpSettings.from_context`` honors ``_meta.kaos_config`` overrides.

    Mirrors the kaos-graph A2-followup-#3 pattern — per-request settings
    overrides land via the context's ``_config`` dict.
    """
    try:
        from kaos_core.base.context import KaosContext  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("kaos-core not installed; from_context unavailable")

    from kaos_nlp_core.settings import KaosNlpSettings

    # Build a minimal context with a kaos_config override for the corpus cap.
    context = KaosContext(session_id="test-session")
    object.__setattr__(context, "_config", {"max_corpus_bytes": 1024})
    settings = KaosNlpSettings.from_context(context)
    assert settings.max_corpus_bytes == 1024


def test_workspace_root_helper_falls_back_to_cwd(tmp_path: Path) -> None:
    """``_workspace_root_from_settings(None)`` returns ``Path.cwd().resolve()``."""
    from kaos_nlp_core.tools import _workspace_root_from_settings

    resolved = _workspace_root_from_settings(None)
    assert resolved == Path.cwd().resolve()


def test_workspace_root_helper_resolves_settings_value(tmp_path: Path) -> None:
    from kaos_nlp_core.settings import KaosNlpSettings
    from kaos_nlp_core.tools import _workspace_root_from_settings

    settings = KaosNlpSettings(workspace_root=str(tmp_path))
    resolved = _workspace_root_from_settings(settings)
    assert resolved == tmp_path.resolve()


def test_resolve_within_root_blocks_traversal(tmp_path: Path) -> None:
    from kaos_nlp_core.tools import _resolve_within_root

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")

    with pytest.raises(ValueError, match="outside the workspace root"):
        _resolve_within_root(str(outside), sandbox.resolve())


def test_resolve_within_root_accepts_inside(tmp_path: Path) -> None:
    from kaos_nlp_core.tools import _resolve_within_root

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    inside = sandbox / "ok.txt"
    inside.write_text("ok")

    result = _resolve_within_root(str(inside), sandbox.resolve())
    assert result == inside.resolve()


def test_lexicon_path_setting_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """``KAOS_NLP_LEXICON_PATH`` flows through ``KaosNlpSettings`` into
    ``_candidate_paths`` (which is what ``Lexicon.load_default()`` calls).
    """
    from kaos_nlp_core.lexicon import _candidate_paths

    # Clear any prior cached state by making sure we read via the env path.
    monkeypatch.setenv("KAOS_NLP_LEXICON_PATH", "/tmp/custom-lexicon.bin")
    paths = _candidate_paths()
    assert paths[0] == Path("/tmp/custom-lexicon.bin")
    # Defaults still appended after the override.
    assert len(paths) > 1


def test_http_token_redaction_in_repr() -> None:
    """``SecretStr`` redacts the token in ``repr()`` and ``str()``."""
    from pydantic import SecretStr

    from kaos_nlp_core.settings import KaosNlpSettings

    settings = KaosNlpSettings(http_token=SecretStr("super-secret-value"))
    assert "super-secret-value" not in repr(settings)
    # The redacted form is conventionally `**********` from pydantic.
    assert settings.http_token is not None
    assert "***" in str(settings.http_token)
    # Explicit unwrap still works.
    assert settings.http_token.get_secret_value() == "super-secret-value"
