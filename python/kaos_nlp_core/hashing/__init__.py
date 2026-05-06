"""Fuzzy hashing: MinHash/LSH near-duplicate detection, CTPH similarity."""

from kaos_nlp_core._rust.hashing import (
    CTPH,
    CTPHDigest,
    DuplicateGroup,
    MinHasher,
    MinHashIndex,
    MinHashSignature,
    TokenCTPH,
    ctph_hash_bytes,
    ctph_hash_str,
    ctph_piece_similarity,
    ctph_similarity,
    find_duplicates,
    token_ctph_hash,
)

__all__ = [
    "CTPH",
    "CTPHDigest",
    "DuplicateGroup",
    "MinHashIndex",
    "MinHashSignature",
    "MinHasher",
    "TokenCTPH",
    "ctph_hash_bytes",
    "ctph_hash_str",
    "ctph_piece_similarity",
    "ctph_similarity",
    "find_duplicates",
    "token_ctph_hash",
]
