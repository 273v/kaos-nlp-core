"""kaos-nlp-core module settings.

Uses ``ModuleSettings`` from kaos-core for typed, environment-aware
configuration. Resolution order:

    explicit overrides
        > KaosContext._config (per-request _meta.kaos_config)
        > KAOS_NLP_* env vars
        > .env file
        > field defaults

This module centralizes the four environment variables previously read
ad-hoc via ``os.environ.get()``:

    KAOS_NLP_WORKSPACE_ROOT     filesystem sandbox for tool I/O
    KAOS_NLP_MAX_CORPUS_BYTES   corpus file size cap (default 256 MiB)
    KAOS_NLP_LEXICON_PATH       opengloss lexicon override path
    KAOS_NLP_HTTP_TOKEN         operator ack for kaos-nlp-serve --http

MCP tools thread the request context through ``KaosNlpSettings.from_context(context)``
so per-request ``_meta.kaos_config`` overrides apply on top of the env
defaults — same pattern that closed audit follow-up #3 in kaos-graph.

Importing this module requires kaos-core (for ``ModuleSettings``);
standalone ``import kaos_nlp_core`` does not pull it in.
"""

from __future__ import annotations

from kaos_core.config import ModuleSettings
from pydantic import Field, SecretStr
from pydantic_settings import SettingsConfigDict

__all__ = ["KaosNlpSettings"]


# Defaults preserved from the prior os.environ.get() implementation.
_DEFAULT_MAX_CORPUS_BYTES = 256 * 1024 * 1024  # 256 MiB


class KaosNlpSettings(ModuleSettings):
    """Settings for the kaos-nlp-core module.

    Field names map 1:1 to the existing ``KAOS_NLP_*`` env-var surface so
    no legacy-name fallback shim is needed (pydantic-settings honors the
    ``env_prefix`` automatically).
    """

    # Filesystem sandbox for the tools that read/write files
    # (``kaos-nlp-build-index`` is the canonical example). When ``None``
    # the tool layer falls back to ``Path.cwd().resolve()`` so the
    # historical "default to CWD" behaviour is preserved.
    workspace_root: str | None = None

    # Per-corpus file-size cap (bytes). 0 disables the cap entirely;
    # ge=0 keeps pydantic from accepting negatives.
    max_corpus_bytes: int = Field(default=_DEFAULT_MAX_CORPUS_BYTES, ge=0)

    # Override path for the bundled opengloss lexicon binary. When set,
    # ``kaos_nlp_core.lexicon._candidate_paths()`` prepends this path to
    # the search list.
    lexicon_path: str | None = None

    # Operator-ack token for ``kaos-nlp-serve --http``. The value itself
    # isn't checked against incoming requests at v0.1.0a1 — kaos-mcp's
    # current transport doesn't enforce bearer-token auth — but the
    # presence of *any* non-empty value is required to start the HTTP
    # server (per the F3 sandbox model). ``SecretStr`` redacts in logs
    # and config dumps.
    http_token: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_prefix="KAOS_NLP_",
        env_file=".env",
        extra="ignore",
    )
