# Dense similarity-graph primitives — design note

Status: implemented in `rust/core/similarity/graph.rs` at v0.1.5
(this release). Companion to `design-similarity-simd.md` (the cosine
kernels these build on) and `rust/core/similarity/topk.rs` (the single-
query selection these generalise).

## Why

Callers doing retrieval dedup or topic discovery over an embedding
matrix kept hand-rolling the same two things on top of `top_k_cosine`:

1. An `N × top_k_cosine` Python loop to get every row's nearest
   neighbours (an all-pairs kNN graph).
2. A union-find to collapse "row i is near row j" pairs into duplicate
   groups.

Both are slow and error-prone in Python: the loop pays per-call
overhead `N` times and never parallelises across rows, and the
union-find is easy to get subtly wrong (non-deterministic labels,
quadratic merges). The feedback that prompted this work named exactly
this: *"hand-rolled an N×top_k_cosine loop + union-find."*

## What this module adds

Two **dense-vector** primitives — they compute cosines, so they belong
in `kaos-nlp-core` next to `top_k_cosine`/`mmr_select`:

- `knn_graph(matrix, k, include_self, normalized)` — for every row, its
  `k` nearest other rows. One fused Rust call. Each row's sweep reuses
  the existing SIMD cosine kernel + the `select_top_k` heap (factored
  out of `top_k_cosine` so the tie-break logic is shared, not copied),
  and the rows fan out across Rayon worker threads with the GIL
  released. Output is a rectangular `(n_rows, effective_k)` neighbour
  table.
- `near_duplicates(matrix, threshold, normalized, max_pairs)` — every
  upper-triangle pair `(i, j)`, `i < j`, with cosine `>= threshold`.
  The sparse edge set a dedup pre-pass needs; each row sweeps its tail
  in one SIMD pass, rows fan out across Rayon, and the `max_pairs` cap
  is never silent (it sets `truncated` and the Python wrapper warns).

## What this module deliberately does NOT add

**Component labelling / clustering is not a dense-vector concern.** It
operates on the *edge set*, not on vectors, and `kaos-graph` already
owns it — `weakly_connected_components` etc. are backed by petgraph's
`UnionFind`. Reinventing a disjoint-set here would duplicate the graph
package and split a single concern across two repos.

So the layer cake is:

```
kaos-nlp-core (this module)   vectors ──► similarity edges
        │  KnnGraph.edges() / NearDuplicates.pairs  →  (m, 2) uint32
        ▼
kaos-graph                    edges ──► components / communities
        │  connected_components_from_edges(n_nodes, edges) → labels
        ▼
kaos-content                  components ──► semantic-dedup orchestration
                              (policy, canonical-record selection, levels)
```

Each layer is independent (no cross-dependency between the two
primitive packages); the application composes them. `KnnGraph.edges()`
and `NearDuplicates.pairs` emit the `(m, 2)` uint32 edge array that
`kaos-graph` ingests directly.

Approximate nearest neighbour (HNSW / IVF) also stays out: these
primitives are exact / brute-force. The exact sweep is well within
budget up to ~100k rows on commodity hardware thanks to the SIMD
kernels; beyond that an ANN index belongs in a sibling module so this
one stays small and deterministic.

## Determinism

Every function is pure — same inputs produce the same output bytes
regardless of Rayon thread count:

- `knn_graph` inherits `top_k_cosine`'s ascending-index tie-break. Rows
  are assembled in row order after the parallel map, so the table is
  index-stable.
- `near_duplicates` emits pairs in `(i, j)` lexicographic order
  (per-row edge lists concatenated in row order).
- A row that cannot fill all `k` neighbour slots — only possible when
  the matrix contains NaN/inf rows, whose cosines are unrankable and so
  are dropped — is padded with the `NO_NEIGHBOR` sentinel
  (`u32::MAX`) and a `NaN` score, keeping the table rectangular. With
  finite inputs (the embedding contract) padding never occurs.

## Contiguity

The primitives require C-contiguous float32, like every function in
this module; the numpy buffer layer raises `TypeError` otherwise.
`EmbeddingModel.embed()` output already satisfies the contract.
`as_contiguous_f32` is the one-call escape hatch for the cases that
don't (column slices, `np.stack` of heterogeneous sources, foreign
dtypes) — the coercion is explicit, never hidden inside the hot path.
