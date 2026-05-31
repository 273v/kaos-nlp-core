"""Regression tests for aggregation determinism across PYTHONHASHSEED values.

Background
==========

``vote``, ``majority``, and ``weighted`` document a contract that ties
are broken by "order of first appearance." Prior to the determinism
fix, those functions iterated each chunk's labels via
``for name in set(chunk_labels)``, which uses Python's
hash-randomized set iteration order. That order is **non-deterministic
across processes** under default Python settings — `PYTHONHASHSEED` is
randomized on every interpreter start unless pinned. So the
``first_seen`` map (and therefore the tiebreak winner) silently
depended on hash order, violating both the documented contract and the
``kaos-nlp-core`` AGENTS.md determinism rule.

The fix replaces ``set(chunk_labels)`` with
``dict.fromkeys(chunk_labels)``, which preserves input order while
still deduping. These tests re-run each function across many fresh
Python subprocesses (each with a different ``PYTHONHASHSEED``) and
assert all subprocesses returned the same winner.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# A fixture engineered to maximize the chance of tiebreak drift before
# the fix: many labels per chunk, sized so two candidates tie.
_TIE_INPUTS = [
    ["alpha", "beta", "gamma"],
    ["beta", "alpha", "delta"],
    ["gamma", "delta", "alpha"],
    ["alpha", "epsilon", "zeta"],
    ["eta", "alpha", "theta"],
    ["iota", "alpha", "kappa"],
]


def _run_under_hashseed(seed: int, fn_source: str) -> str:
    """Run ``fn_source`` (a Python snippet) under ``PYTHONHASHSEED=seed``.

    Returns stdout.strip() (the JSON-encoded result).
    """
    code = (
        "import json\n"
        "from kaos_nlp_core.aggregation import "
        "vote, majority, union, intersection, weighted, max_score\n"
        f"_INPUTS = {json.dumps(_TIE_INPUTS)}\n"
        f"{fn_source}\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        # Inherit the full environment (USERPROFILE/HOME/SYSTEMROOT on Windows,
        # etc.) and only override PYTHONHASHSEED — a hardcoded minimal env
        # broke the kaos_nlp_core import on Windows (home-dir resolution).
        env={**os.environ, "PYTHONHASHSEED": str(seed)},
    )
    return proc.stdout.decode("utf-8").strip()


def _check_invariant_across_seeds(fn_source: str, seeds: list[int]) -> None:
    outputs = {seed: _run_under_hashseed(seed, fn_source) for seed in seeds}
    distinct = set(outputs.values())
    assert len(distinct) == 1, "Non-determinism across PYTHONHASHSEED values:\n" + "\n".join(
        f"  seed={s}: {outputs[s]}" for s in seeds
    )


def test_vote_is_deterministic_across_hash_seeds() -> None:
    """``vote`` returns the same winner under any PYTHONHASHSEED."""
    _check_invariant_across_seeds(
        "print(json.dumps(vote(_INPUTS)))",
        seeds=[0, 1, 7, 42, 31337, 99999],
    )


def test_majority_is_deterministic_across_hash_seeds() -> None:
    _check_invariant_across_seeds(
        "print(json.dumps(majority(_INPUTS, threshold=0.4)))",
        seeds=[0, 1, 7, 42, 31337, 99999],
    )


def test_weighted_single_is_deterministic_across_hash_seeds() -> None:
    _check_invariant_across_seeds(
        "print(json.dumps(weighted(_INPUTS, threshold=0.4)))",
        seeds=[0, 1, 7, 42, 31337, 99999],
    )


def test_weighted_multi_is_deterministic_across_hash_seeds() -> None:
    """frozenset values serialize to sorted lists for cross-process compare."""
    _check_invariant_across_seeds(
        "result = weighted(_INPUTS, threshold=0.4, multi=True)\nprint(json.dumps(sorted(result)))",
        seeds=[0, 1, 7, 42, 31337, 99999],
    )


def test_in_process_smoke_tie_break_by_input_order() -> None:
    """A focused in-process test: ``vote`` on a known tie should pick
    the first-input-appearing label deterministically.
    """
    from kaos_nlp_core.aggregation import vote

    # ``b`` appears first in chunk 0, then ``a`` in chunk 1.
    # Both end up tied at count 2. First-appearance is ``b``.
    result = vote([["b", "a"], ["a", "b"], ["a"], ["b"]])
    assert result == "b"


def test_majority_tie_break_uses_input_order() -> None:
    from kaos_nlp_core.aggregation import majority

    result = majority(
        [["b", "a"], ["b", "a"], ["a", "b"]],
        threshold=0.5,
    )
    # Both 'a' and 'b' have count 3 (each appears once in every chunk).
    # First-appearance (in chunk 0): 'b'.
    assert result == "b"
