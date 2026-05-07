"""Type stubs for kaos_nlp_core._rust.diff."""

from typing import Any

from kaos_nlp_core._rust.segmentation import PunktTokenizer

def py_diff_documents(
    a: str,
    b: str,
    *,
    granularity: str = "sentence",
    algorithm: str = "token-jaccard",
    n: int = 2,
    lowercase: bool = True,
    prefix_weight: float = 0.1,
    match_threshold: float = 0.85,
    modify_threshold: float = 0.4,
    detect_moves: bool = False,
    move_distance_ratio: float = 0.1,
    tokenizer: PunktTokenizer | None = None,
) -> list[dict[str, Any]]: ...
