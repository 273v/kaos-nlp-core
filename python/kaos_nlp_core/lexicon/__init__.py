"""Semantic knowledge graph for query expansion.

Generic — works with OpenGloss or any dataset providing words, senses,
and typed semantic edges (synonym, antonym, hypernym, hyponym, etc.).

The bundled OpenGloss v1.3 lexicon (~206 k entries) is auto-discovered
by :func:`default_opengloss_lexicon` from either the in-repo data
directory or the platform user cache. Loading the full lexicon takes
~1 s and ~250 MB RAM; the loader caches the result process-wide so
subsequent calls are free.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from kaos_nlp_core._rust.lexicon import Lexicon as _RustLexicon
from kaos_nlp_core.matching import FstSet
from kaos_nlp_core.types import RelatedTerm, SenseEntry

logger = logging.getLogger(__name__)

OPENGLOSS_FILENAME = "opengloss-v1.3.lexicon.bin"

# Search order for the bundled OpenGloss binary: in-repo data dir first
# (covers editable installs and source checkouts), then the user cache
# directory (matches the kaos-agents convention).
_DEFAULT_LEXICON_SEARCH_PATHS: tuple[Path, ...] = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / OPENGLOSS_FILENAME,
    Path.home() / ".cache" / "kaos" / OPENGLOSS_FILENAME,
)

_DEFAULT_LEXICON: Lexicon | None = None


class Lexicon:
    """Semantic knowledge graph with typed sense + related-term results.

    Thin wrapper over the Rust ``Lexicon`` so ``get_senses()`` and
    ``related_typed()`` return frozen dataclasses instead of dicts.
    """

    def __init__(self) -> None:
        self._inner = _RustLexicon()

    @staticmethod
    def load(path: str) -> Lexicon:
        """Load a lexicon binary from disk."""
        lex = Lexicon.__new__(Lexicon)
        lex._inner = _RustLexicon.load(path)
        return lex

    @staticmethod
    def load_default() -> Lexicon:
        """Locate and load the bundled OpenGloss lexicon.

        Searches the in-repo ``data/`` directory first, then
        ``~/.cache/kaos/``. Override the search via the
        ``KAOS_NLP_LEXICON_PATH`` env var. Returns a process-cached
        instance so subsequent calls are free.

        Raises:
            FileNotFoundError: if the lexicon cannot be found in any
                known location, with instructions on how to install it.
        """
        return default_opengloss_lexicon()

    def save(self, path: str) -> None:
        self._inner.save(path)

    def add_entry(self, entry: dict[str, Any]) -> None:
        self._inner.add_entry(entry)

    def add_entries(self, entries: list[dict[str, Any]]) -> None:
        self._inner.add_entries(entries)

    def __len__(self) -> int:
        return len(self._inner)

    def __contains__(self, word: str) -> bool:
        return self._inner.contains(word)

    def contains(self, word: str) -> bool:
        return self._inner.contains(word)

    def synonyms(self, word: str) -> list[str]:
        return self._inner.synonyms(word)

    def antonyms(self, word: str) -> list[str]:
        return self._inner.antonyms(word)

    def hypernyms(self, word: str) -> list[str]:
        return self._inner.hypernyms(word)

    def hyponyms(self, word: str) -> list[str]:
        return self._inner.hyponyms(word)

    def inflections(self, word: str) -> list[str]:
        return self._inner.inflections(word)

    def collocations(self, word: str) -> list[str]:
        return self._inner.collocations(word)

    def related(
        self,
        word: str,
        relation: str,
        pos: str | None = None,
        sense_index: int | None = None,
    ) -> list[str]:
        """Return related terms as plain strings (back-compat)."""
        return self._inner.related(word, relation, pos, sense_index)

    def related_typed(
        self,
        word: str,
        relation: str,
        pos: str | None = None,
        sense_index: int | None = None,
    ) -> list[RelatedTerm]:
        """Return related terms as :class:`RelatedTerm` records.

        Each result carries the supplied relation / pos / sense_index
        so callers can flatten multiple lookups into a single typed
        list without losing track of how each term was reached.
        """
        raw = self._inner.related(word, relation, pos, sense_index)
        return [
            RelatedTerm(text=t, relation=relation, pos=pos, sense_index=sense_index) for t in raw
        ]

    def expand_query(
        self,
        terms: list[str],
        relations: list[str],
        max_depth: int = 1,
    ) -> list[str]:
        return self._inner.expand_query(terms, relations, max_depth)

    def expand_query_sense_aware(
        self,
        terms: list[tuple[str, str | None, int | None]],
        relations: list[str],
    ) -> list[str]:
        return self._inner.expand_query_sense_aware(terms, relations)

    def get_senses(self, word: str) -> list[SenseEntry]:
        raw = self._inner.get_senses(word)
        return [
            SenseEntry(
                part_of_speech=s["part_of_speech"],
                sense_index=s["sense_index"],
                definition=s["definition"],
                synonyms=s.get("synonyms"),
                antonyms=s.get("antonyms"),
                hypernyms=s.get("hypernyms"),
                hyponyms=s.get("hyponyms"),
            )
            for s in raw
        ]

    def to_fst_set(self, include_inflections: bool = True) -> FstSet:
        """Build a compact FST-backed word set from this lexicon.

        Includes every headword (and, by default, every aggregated
        inflection), lower-cased and de-duplicated. The result is
        suitable as a ``lexicon=`` argument to
        :func:`kaos_nlp_core.quality.compute_metrics`.
        """
        return FstSet._from_inner(self._inner.to_fst_set(include_inflections))


# ─── Default loader ─────────────────────────────────────────────────────────


def default_opengloss_lexicon() -> Lexicon:
    """Locate and load the bundled OpenGloss lexicon.

    First call pays the load cost (~1 s, ~250 MB RAM); the result is
    cached at module level. Override the search path via the
    ``KAOS_NLP_LEXICON_PATH`` environment variable.

    Search order:
      1. ``$KAOS_NLP_LEXICON_PATH`` if set.
      2. In-repo ``data/opengloss-v1.3.lexicon.bin``.
      3. ``~/.cache/kaos/opengloss-v1.3.lexicon.bin``.

    Raises:
        FileNotFoundError: if the lexicon cannot be located, with
            installation instructions.
    """
    global _DEFAULT_LEXICON
    if _DEFAULT_LEXICON is not None:
        return _DEFAULT_LEXICON

    path = _resolve_lexicon_path()
    if path is None:
        searched = [str(p) for p in _candidate_paths()]
        raise FileNotFoundError(
            f"OpenGloss lexicon ({OPENGLOSS_FILENAME}) not found in any known "
            f"location. Searched: {', '.join(searched)}. To install: run "
            f"`uv run --with datasets,huggingface_hub python "
            f"scripts/build_opengloss_lexicon.py` from the kaos-nlp-core repo, "
            f"or set KAOS_NLP_LEXICON_PATH to an existing lexicon binary. "
            f"For metric-only OCR detection without the full graph, use "
            f"kaos_nlp_core.quality.default_english_wordset() instead — it "
            f"ships in the wheel."
        )
    logger.debug("kaos_nlp_core.lexicon: loading OpenGloss from %s", path)
    _DEFAULT_LEXICON = Lexicon.load(str(path))
    logger.debug("kaos_nlp_core.lexicon: loaded %d entries from %s", len(_DEFAULT_LEXICON), path)
    return _DEFAULT_LEXICON


def _candidate_paths() -> tuple[Path, ...]:
    """The full search path for the bundled lexicon (env override + defaults).

    Reads ``KAOS_NLP_LEXICON_PATH`` via the centralized
    :class:`KaosNlpSettings` (env_prefix=``KAOS_NLP_``). Falls back to a
    direct ``os.environ`` read when ``kaos-core`` isn't installed, so
    standalone consumers without the ``[mcp]`` extra still get the env
    override.
    """
    env_path: str | None = None
    try:
        from kaos_nlp_core.settings import KaosNlpSettings

        env_path = KaosNlpSettings().lexicon_path
    except ImportError:
        env_path = os.environ.get("KAOS_NLP_LEXICON_PATH")
    if env_path:
        return (Path(env_path), *_DEFAULT_LEXICON_SEARCH_PATHS)
    return _DEFAULT_LEXICON_SEARCH_PATHS


def _resolve_lexicon_path() -> Path | None:
    for candidate in _candidate_paths():
        if candidate.is_file():
            return candidate
    return None


__all__ = [
    "OPENGLOSS_FILENAME",
    "FstSet",
    "Lexicon",
    "RelatedTerm",
    "SenseEntry",
    "default_opengloss_lexicon",
]
