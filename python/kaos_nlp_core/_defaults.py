"""Shared package defaults and model loaders."""

from __future__ import annotations

import importlib.resources
import logging
from contextlib import suppress
from functools import lru_cache
from pathlib import Path

from kaos_nlp_core._rust.segmentation import PunktParameters, PunktTokenizer

logger = logging.getLogger(__name__)

_MODEL_FILENAME = "default.npkt.gz"


def _resolve_default_model_path() -> Path | None:
    """Locate the bundled Punkt model file (filesystem fallback).

    The default loader prefers the embedded bytes baked into ``_rust.abi3.so``
    via ``include_bytes!``; this filesystem search is only used when the
    embedded path is unavailable (e.g. dev builds via ``maturin develop``
    against an older ``_rust.so`` that predates the embedding).

    Resolution order:
    1. ``importlib.resources`` — works for editable installs where the
       ``kaos_nlp_core.models`` subpackage still has the .npkt.gz on disk.
    2. Source-tree fallback — ``parents[2] / "models"`` relative to this
       file, resolving to the monorepo top-level ``models/`` for contributors.

    Returns ``None`` when the model file cannot be found at any location.
    """
    # --- Strategy 1: importlib.resources (Python 3.9+) ---
    with suppress(Exception):
        ref = importlib.resources.files("kaos_nlp_core.models").joinpath(_MODEL_FILENAME)
        # In editable installs ref is typically a Path; in wheel installs it
        # may be a zipfile traversable.  ``as_posix()`` / ``__fspath__`` only
        # exist on real paths.
        candidate = Path(str(ref))
        if candidate.exists():
            return candidate

    # --- Strategy 2: source-tree relative path (editable installs) ---
    source_tree = Path(__file__).resolve().parents[2] / "models" / _MODEL_FILENAME
    if source_tree.exists():
        return source_tree

    return None


DEFAULT_PUNKT_MODEL_PATH: Path | None = _resolve_default_model_path()


def load_default_punkt_parameters() -> PunktParameters | None:
    """Load the default Punkt parameters.

    Tries the embedded bytes first (always available in a published wheel),
    then falls back to filesystem search for editable / dev installs.
    Returns ``None`` only if the embedded path errors AND no filesystem copy
    is reachable.
    """
    # --- Strategy 1: embedded bytes baked into _rust.abi3.so. Default path
    # in published wheels — no filesystem state required.
    try:
        return PunktParameters.default_embedded()
    except (AttributeError, OSError) as exc:
        logger.debug(
            "kaos_nlp_core._defaults: embedded Punkt unavailable (%s); "
            "falling back to filesystem search",
            exc,
        )

    # --- Strategy 2: filesystem copy (editable / dev builds).
    if DEFAULT_PUNKT_MODEL_PATH is None or not DEFAULT_PUNKT_MODEL_PATH.exists():
        return None
    return PunktParameters.load(str(DEFAULT_PUNKT_MODEL_PATH))


@lru_cache(maxsize=1)
def get_default_punkt_tokenizer() -> PunktTokenizer:
    """Return a cached tokenizer using the bundled legal model when available."""
    params = load_default_punkt_parameters()
    if params is None:
        return PunktTokenizer()
    return PunktTokenizer(params)
