"""Shared package defaults and model loaders."""

from __future__ import annotations

import importlib.resources
from contextlib import suppress
from functools import lru_cache
from pathlib import Path

from kaos_nlp_core._rust.segmentation import PunktParameters, PunktTokenizer

_MODEL_FILENAME = "default.npkt.gz"


def _resolve_default_model_path() -> Path | None:
    """Locate the bundled Punkt model file.

    Resolution order:
    1. ``importlib.resources`` — works for both wheel installs (model embedded in
       the ``kaos_nlp_core.models`` subpackage) and editable installs (symlink
       in the source tree).
    2. Source-tree fallback — ``parents[2] / "models"`` relative to this file,
       which resolves to the project root during editable installs even if
       ``importlib.resources`` returns a non-filesystem traversable.

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
    """Load bundled Punkt parameters when available."""
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
