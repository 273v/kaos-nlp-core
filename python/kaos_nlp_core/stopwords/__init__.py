"""Curated English stopword resource.

A versioned, derived stopword list — never a hand-typed guess. It is built
(see ``scripts/build_stopwords.py``) by a **hybrid** method recorded in the
asset's provenance:

1. **Cross-domain document frequency** over KL3M sources (copyright-clean),
   tokenized with the kaos-nlp-core Rust tokenizer: a term qualifies only
   when near-universal across *multiple* domains, so genuine function words
   survive while domain content words (``section``/``shall``/``agreement``)
   are excluded.
2. **OpenGloss closed-class POS** (determiner / adposition / conjunction /
   pronoun / auxiliary / particle), single-word and all-senses-closed,
   de-langed against the multilingual dictionary.
3. A manual-review completion of the standard English closed class that the
   automatic halves structurally miss (open-class homonyms like
   ``do``/``can``/``will``; pronouns the legal corpus underuses).

Loading the bundled asset is a trivial cached read — the *derivation* (the
CPU work: tokenization, document-frequency counting) happens offline in the
build script via the Rust tokenizer.

Example::

    from kaos_nlp_core.stopwords import stopwords
    from kaos_nlp_core.ctfidf import class_tfidf

    labels = class_tfidf(texts, cluster_ids, stopwords=stopwords())
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

_ASSETS: dict[str, str] = {"en": "stopwords-en-v1.json"}


def _load(language: str) -> dict[str, Any]:
    if language not in _ASSETS:
        available = ", ".join(sorted(_ASSETS))
        msg = f"no stopword resource for language {language!r}; available: [{available}]"
        raise ValueError(msg)
    # ``data`` is a resource directory, not a package, so navigate from the
    # package root (matches how the bundled .fst / lexicon assets resolve).
    ref = resources.files("kaos_nlp_core").joinpath("data", _ASSETS[language])
    return json.loads(ref.read_text("utf-8"))


@lru_cache(maxsize=4)
def stopwords(language: str = "en") -> frozenset[str]:
    """Return the curated stopword set for ``language`` (default English).

    Args:
        language: language code. Only ``"en"`` ships today.

    Returns:
        A ``frozenset[str]`` of lowercase stopwords. Cached per language.

    Raises:
        ValueError: no resource for the requested language.
    """
    return frozenset(_load(language)["terms"])


@lru_cache(maxsize=4)
def stopwords_provenance(language: str = "en") -> dict[str, Any]:
    """Return the derivation provenance for ``language``'s stopword set.

    The recorded method, corpus/source documents, thresholds, and per-half
    selection counts — so the list is auditable, never opaque.
    """
    return dict(_load(language)["provenance"])


__all__ = ["stopwords", "stopwords_provenance"]
