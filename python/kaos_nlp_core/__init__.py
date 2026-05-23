"""kaos-nlp-core: High-performance NLP primitives for the Kelvin Agentic OS."""

from kaos_nlp_core import (
    aggregation,
    algorithms,
    chunking,
    concepts,
    content_type,
    documents,
    extract,
    hashing,
    lexicon,
    locale_data,
    matching,
    quality,
    retrieval,
    search,
    segmentation,
    similarity,
    structure,
    structures,
    token_properties,
    tokenizer,
    types,
    vocabulary,
)
from kaos_nlp_core.tools import register_nlp_tools

# `__version__` reads from installed package metadata so it matches what
# `pip show kaos-nlp-core` reports (PEP 440 form, e.g. "0.1.0a1"). Falling
# back to the Rust extension's Cargo SemVer string ("0.1.0-alpha.1") only
# happens for editable/in-place builds where dist-info is missing — which
# would otherwise cause a version drift between `kaos_nlp_core.__version__`
# and PyPI's metadata. See per-package-release.md A7 / F009 lesson #3.
try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        __version__ = _version("kaos-nlp-core")
    except _PackageNotFoundError:  # pragma: no cover - source/editable build only
        from kaos_nlp_core._rust import __version__
    del _version, _PackageNotFoundError
except Exception:  # pragma: no cover - defensive: importlib.metadata always present on 3.13+
    from kaos_nlp_core._rust import __version__

__all__ = [
    "__version__",
    "aggregation",
    "algorithms",
    "chunking",
    "concepts",
    "content_type",
    "documents",
    "extract",
    "hashing",
    "lexicon",
    "locale_data",
    "matching",
    "quality",
    "register_nlp_tools",
    "retrieval",
    "search",
    "segmentation",
    "similarity",
    "structure",
    "structures",
    "token_properties",
    "tokenizer",
    "types",
    "vocabulary",
]
