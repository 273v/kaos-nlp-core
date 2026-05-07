"""Comprehensive quality and performance evaluation for fuzzy hashing primitives.

Tests against real data fixtures:
- War & Peace / Shakespeare (literature)
- USC (68K US Code sections)
- EDGAR agreements (200 SEC filings)
- Patents (200 patent texts)

Measures:
- Similarity quality (known-identical, known-different, near-duplicate detection)
- Throughput (docs/sec, MB/sec)
- Memory footprint
- Deduplication accuracy on real corpora
"""

import json
import sys
import time
import tracemalloc
from pathlib import Path

from kaos_nlp_core.hashing import (
    CTPH,
    MinHasher,
    MinHashIndex,
    TokenCTPH,
    find_duplicates,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def load_text(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


def load_jsonl(name: str, max_docs: int = 0, text_field: str = "text") -> list[dict]:
    docs = []
    with (FIXTURES_DIR / name).open() as f:
        for line in f:
            obj = json.loads(line)
            docs.append(obj)
            if max_docs and len(docs) >= max_docs:
                break
    return docs


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / (1024 * 1024):.1f} MB"


def fmt_rate(count: int, elapsed: float) -> str:
    return f"{count / elapsed:,.0f} docs/sec" if elapsed > 0 else "inf"


def fmt_throughput(total_bytes: int, elapsed: float) -> str:
    mb = total_bytes / (1024 * 1024)
    return f"{mb / elapsed:.1f} MB/sec" if elapsed > 0 else "inf"


def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def subsection(title: str):
    print(f"\n--- {title} ---")


# ─── 1. MinHash Quality ──────────────────────────────────────────────────────


def eval_minhash_quality():
    section("1. MinHash / LSH Quality")

    hasher = MinHasher(256, 42)

    # 1a. Self-similarity
    subsection("1a. Self-similarity (identical texts)")
    texts = [
        ("War & Peace chapter 1", load_text("war_and_peace.txt")[:5000]),
        ("Shakespeare sonnet 1", load_text("shakespeare.txt")[:2000]),
    ]
    for name, text in texts:
        words = text.split()
        sig = hasher.hash_token_shingles(words, 3)
        sim = sig.jaccard(sig)
        print(f"  {name}: self-similarity = {sim:.6f} (expected: 1.0)")

    # 1b. Near-duplicate detection
    subsection("1b. Near-duplicate detection (minor edits)")
    war_peace = load_text("war_and_peace.txt")

    # Take a chapter and make small edits
    chapter = war_peace[:5000]
    chapter_words = chapter.split()

    # Variant 1: replace 5% of words
    variant_5pct = chapter_words.copy()
    for i in range(0, len(variant_5pct), 20):
        variant_5pct[i] = "REPLACED"
    sig_orig = hasher.hash_token_shingles(chapter_words, 3)
    sig_5pct = hasher.hash_token_shingles(variant_5pct, 3)
    print(f"  Original vs 5% replaced: Jaccard = {sig_orig.jaccard(sig_5pct):.4f}")

    # Variant 2: replace 10% of words
    variant_10pct = chapter_words.copy()
    for i in range(0, len(variant_10pct), 10):
        variant_10pct[i] = "REPLACED"
    sig_10pct = hasher.hash_token_shingles(variant_10pct, 3)
    print(f"  Original vs 10% replaced: Jaccard = {sig_orig.jaccard(sig_10pct):.4f}")

    # Variant 3: replace 20% of words
    variant_20pct = chapter_words.copy()
    for i in range(0, len(variant_20pct), 5):
        variant_20pct[i] = "REPLACED"
    sig_20pct = hasher.hash_token_shingles(variant_20pct, 3)
    print(f"  Original vs 20% replaced: Jaccard = {sig_orig.jaccard(sig_20pct):.4f}")

    # Variant 4: append 10% extra text
    extended = chapter_words + chapter_words[: len(chapter_words) // 10]
    sig_ext = hasher.hash_token_shingles(extended, 3)
    print(f"  Original vs 10% appended: Jaccard = {sig_orig.jaccard(sig_ext):.4f}")

    # 1c. Cross-document similarity
    subsection("1c. Cross-document similarity (different works)")
    war_peace_words = war_peace[:10000].split()
    shakespeare_words = load_text("shakespeare.txt")[:10000].split()
    sig_wp = hasher.hash_token_shingles(war_peace_words, 3)
    sig_sh = hasher.hash_token_shingles(shakespeare_words, 3)
    print(f"  War & Peace vs Shakespeare: Jaccard = {sig_wp.jaccard(sig_sh):.4f}")

    # 1d. USC section similarity (legal text)
    subsection("1d. Legal text similarity (USC sections)")
    usc_docs = load_jsonl("usc.jsonl", max_docs=100)
    sigs = []
    for doc in usc_docs:
        words = doc["text"].split()
        sig = hasher.hash_token_shingles(words, 3)
        sigs.append((doc["identifier"], sig))

    # Show top-5 most similar pairs
    pairs = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            sim = sigs[i][1].jaccard(sigs[j][1])
            pairs.append((sim, sigs[i][0], sigs[j][0]))
    pairs.sort(reverse=True)
    print("  Top-5 most similar USC section pairs:")
    for sim, id1, id2 in pairs[:5]:
        print(f"    {sim:.4f}  {id1[:50]:50s} vs {id2[:50]}")

    # Distribution stats
    sims = [p[0] for p in pairs]
    print(f"  Similarity distribution (100 docs, {len(pairs)} pairs):")
    print(
        f"    min={min(sims):.4f}  median={sorted(sims)[len(sims) // 2]:.4f}  "
        f"max={max(sims):.4f}  mean={sum(sims) / len(sims):.4f}"
    )


# ─── 2. MinHash / LSH Performance ────────────────────────────────────────────


def eval_minhash_performance():
    section("2. MinHash / LSH Performance")

    # 2a. Signature computation throughput
    subsection("2a. Signature computation throughput")
    hasher = MinHasher(128, 42)

    for name, n_docs in [("USC (1K)", 1000), ("USC (10K)", 10000)]:
        usc_docs = load_jsonl("usc.jsonl", max_docs=n_docs)
        total_bytes = sum(len(d["text"].encode()) for d in usc_docs)

        tracemalloc.start()
        t0 = time.perf_counter()
        sigs = []
        for doc in usc_docs:
            words = doc["text"].split()
            sig = hasher.hash_token_shingles(words, 3)
            sigs.append(sig)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(
            f"  {name}: {len(usc_docs)} docs, "
            f"{fmt_bytes(total_bytes)} text, "
            f"{elapsed:.3f}s, "
            f"{fmt_rate(len(usc_docs), elapsed)}, "
            f"{fmt_throughput(total_bytes, elapsed)}, "
            f"peak_mem={fmt_bytes(peak)}"
        )

    # 2b. LSH index build + query
    subsection("2b. LSH index build + query")
    n = 5000
    usc_docs = load_jsonl("usc.jsonl", max_docs=n)
    sigs = []
    for doc in usc_docs:
        words = doc["text"].split()
        sig = hasher.hash_token_shingles(words, 3)
        sigs.append(sig)

    for threshold in [0.3, 0.5, 0.8]:
        index = MinHashIndex.with_threshold(128, threshold)

        tracemalloc.start()
        t0 = time.perf_counter()
        for i, sig in enumerate(sigs):
            index.insert(i, sig)
        build_time = time.perf_counter() - t0
        _, build_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Query performance (100 random queries)
        t0 = time.perf_counter()
        total_candidates = 0
        for sig in sigs[:100]:
            candidates = index.query_candidates(sig)
            total_candidates += len(candidates)
        query_time = time.perf_counter() - t0
        avg_candidates = total_candidates / 100

        # query_above_threshold
        t0 = time.perf_counter()
        total_above = 0
        for sig in sigs[:100]:
            results = index.query_above_threshold(sig, threshold)
            total_above += len(results)
        query_thresh_time = time.perf_counter() - t0
        avg_above = total_above / 100

        print(
            f"  threshold={threshold}: build={build_time:.3f}s, "
            f"query_100={query_time * 1000:.1f}ms (avg_candidates={avg_candidates:.0f}), "
            f"query_thresh_100={query_thresh_time * 1000:.1f}ms (avg_above={avg_above:.1f}), "
            f"index_mem={fmt_bytes(build_peak)}"
        )

    # 2c. find_duplicates end-to-end
    subsection("2c. find_duplicates end-to-end")
    for name, n_docs in [("USC 1K", 1000), ("USC 5K", 5000)]:
        usc_docs = load_jsonl("usc.jsonl", max_docs=n_docs)
        doc_tuples = [(i, usc_docs[i]["text"].split()) for i in range(len(usc_docs))]

        tracemalloc.start()
        t0 = time.perf_counter()
        groups = find_duplicates(hasher, doc_tuples, shingle_size=3, threshold=0.5)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        n_dupes = sum(len(g.duplicates) for g in groups)
        print(
            f"  {name}: {elapsed:.3f}s, {len(groups)} groups, "
            f"{n_dupes} duplicates, peak_mem={fmt_bytes(peak)}"
        )

        if groups:
            # Show first 3 groups
            for g in groups[:3]:
                canon_id = g.canonical_id
                canon_ident = usc_docs[canon_id]["identifier"][:60]
                dupe_info = [(d_id, f"{sim:.3f}") for d_id, sim in g.duplicates[:3]]
                print(f"    canonical={canon_ident}")
                for d_id, sim in dupe_info:
                    d_ident = usc_docs[d_id]["identifier"][:60]
                    print(f"      dup: {d_ident} (sim={sim})")


# ─── 3. CTPH Quality ─────────────────────────────────────────────────────────


def eval_ctph_quality():
    section("3. CTPH Quality (Byte-level)")

    ctph = CTPH(64, 8, 4)

    # 3a. Self-similarity
    subsection("3a. Self-similarity")
    war_peace = load_text("war_and_peace.txt")
    d = ctph.hash_str(war_peace)
    print(f"  War & Peace self-similarity: {d.similarity(d):.6f} (expected: 1.0)")
    print(f"  War & Peace digest: {len(d.blocks)} blocks")

    # 3b. Incremental edits
    subsection("3b. Incremental edits (War & Peace first 10KB)")
    base = war_peace[:10000]
    d_base = ctph.hash_str(base)

    # Append small suffix
    d_append_1pct = ctph.hash_str(base + "x" * 100)
    print(f"  Original vs +1% appended: {d_base.similarity(d_append_1pct):.4f}")

    d_append_10pct = ctph.hash_str(base + "x" * 1000)
    print(f"  Original vs +10% appended: {d_base.similarity(d_append_10pct):.4f}")

    # Replace middle
    mid = len(base) // 2
    replaced_5pct = base[: mid - 250] + "X" * 500 + base[mid + 250 :]
    d_replaced = ctph.hash_str(replaced_5pct)
    print(f"  Original vs 5% middle replaced: {d_base.similarity(d_replaced):.4f}")

    # 3c. Different works
    subsection("3c. Cross-work similarity")
    shakespeare = load_text("shakespeare.txt")[:10000]
    d_sh = ctph.hash_str(shakespeare)
    print(f"  War & Peace vs Shakespeare (10KB each): {d_base.similarity(d_sh):.4f}")

    # 3d. EDGAR agreements
    subsection("3d. EDGAR agreement similarity matrix (first 20)")
    edgar = load_jsonl("edgar_agreements.jsonl", max_docs=20)
    digests = [ctph.hash_str(d["text"]) for d in edgar]

    pairs = []
    for i in range(len(digests)):
        for j in range(i + 1, len(digests)):
            sim = digests[i].similarity(digests[j])
            pairs.append((sim, i, j))
    pairs.sort(reverse=True)

    print("  Top-5 most similar agreement pairs:")
    for sim, i, j in pairs[:5]:
        print(f"    {sim:.4f}  doc_{i} vs doc_{j}")

    sims = [p[0] for p in pairs]
    print(f"  Distribution ({len(pairs)} pairs):")
    print(
        f"    min={min(sims):.4f}  median={sorted(sims)[len(sims) // 2]:.4f}  "
        f"max={max(sims):.4f}  mean={sum(sims) / len(sims):.4f}"
    )

    # 3e. Precision sweep
    subsection("3e. Precision impact on quality")
    base_text = war_peace[:10000]
    modified = base_text[:9000] + "MODIFIED SECTION" * 60 + base_text[9000:]
    for precision in [1, 2, 4, 8]:
        c = CTPH(64, 8, precision)
        d1 = c.hash_str(base_text)
        d2 = c.hash_str(modified)
        print(
            f"  precision={precision} ({precision * 8}-bit): "
            f"blocks={len(d1.blocks)}/{len(d2.blocks)}, "
            f"similarity={d1.similarity(d2):.4f}"
        )


# ─── 4. CTPH Performance ─────────────────────────────────────────────────────


def eval_ctph_performance():
    section("4. CTPH Performance")

    subsection("4a. Throughput by document size")
    ctph = CTPH(64, 8, 4)
    war_peace = load_text("war_and_peace.txt")

    for label, text in [
        ("1 KB", war_peace[:1024]),
        ("10 KB", war_peace[:10240]),
        ("100 KB", war_peace[:102400]),
        ("1 MB", war_peace[:1048576]),
        ("3.3 MB (full War & Peace)", war_peace),
        ("5.4 MB (full Shakespeare)", load_text("shakespeare.txt")),
    ]:
        n_iters = max(1, min(1000, 10_000_000 // max(len(text), 1)))
        t0 = time.perf_counter()
        for _ in range(n_iters):
            ctph.hash_str(text)
        elapsed = time.perf_counter() - t0
        per_call = elapsed / n_iters
        throughput = len(text.encode()) / per_call / (1024 * 1024)
        print(f"  {label:30s}: {per_call * 1000:.3f} ms/call, {throughput:.1f} MB/sec")

    # 4b. Corpus-level throughput
    subsection("4b. Corpus-level throughput (EDGAR 200 agreements)")
    edgar = load_jsonl("edgar_agreements.jsonl", max_docs=200)
    total_bytes = sum(len(d["text"].encode()) for d in edgar)

    tracemalloc.start()
    t0 = time.perf_counter()
    digests = [ctph.hash_str(d["text"]) for d in edgar]
    hash_time = time.perf_counter() - t0
    _, hash_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(
        f"  Hashing: {hash_time:.3f}s, {fmt_rate(len(edgar), hash_time)}, "
        f"{fmt_throughput(total_bytes, hash_time)}, peak_mem={fmt_bytes(hash_peak)}"
    )

    # Pairwise similarity
    t0 = time.perf_counter()
    pair_count = 0
    for i in range(len(digests)):
        for j in range(i + 1, len(digests)):
            digests[i].similarity(digests[j])
            pair_count += 1
    sim_time = time.perf_counter() - t0
    print(
        f"  Pairwise similarity ({pair_count} pairs): {sim_time * 1000:.1f} ms, "
        f"{pair_count / sim_time:,.0f} comparisons/sec"
    )

    # 4c. Window/digest parameter impact
    subsection("4c. Parameter impact on performance")
    text = war_peace[:100000]
    for ws, ds in [(16, 4), (32, 8), (64, 8), (128, 16), (256, 32)]:
        c = CTPH(ws, ds, 4)
        t0 = time.perf_counter()
        for _ in range(10):
            d = c.hash_str(text)
        elapsed = (time.perf_counter() - t0) / 10
        print(f"  window={ws:3d}, digest={ds:2d}: {elapsed * 1000:.2f} ms, {len(d.blocks)} blocks")


# ─── 5. TokenCTPH Quality ────────────────────────────────────────────────────


def eval_token_ctph_quality():
    section("5. Token CTPH Quality")

    ctph = TokenCTPH(4, 8)

    # Simulate token sequences from USC sections
    subsection("5a. Simulated token sequences (USC word hashes)")
    usc_docs = load_jsonl("usc.jsonl", max_docs=50)

    # Use word hash as token ID proxy
    def text_to_token_ids(text: str) -> list[int]:
        return [hash(w) % 100000 for w in text.split()]

    token_seqs = [(doc["identifier"], text_to_token_ids(doc["text"])) for doc in usc_docs]

    # Self-similarity
    d = ctph.compute(token_seqs[0][1])
    print(f"  Self-similarity: {d.similarity(d):.6f} (expected: 1.0)")

    # Near-duplicate: same section with 5% token replacement
    orig = token_seqs[0][1]
    modified = orig.copy()
    for i in range(0, len(modified), 20):
        modified[i] = 99999
    d_orig = ctph.compute(orig)
    d_mod = ctph.compute(modified)
    print(f"  Original vs 5% replaced tokens: {d_orig.similarity(d_mod):.4f}")

    # Cross-section
    d1 = ctph.compute(token_seqs[0][1])
    d2 = ctph.compute(token_seqs[1][1])
    print(f"  Section 0 vs Section 1: {d1.similarity(d2):.4f}")

    # Top-5 most similar pairs
    subsection("5b. Token CTPH pairwise similarity (50 USC sections)")
    digests = [(name, ctph.compute(tokens)) for name, tokens in token_seqs]
    pairs = []
    for i in range(len(digests)):
        for j in range(i + 1, len(digests)):
            sim = digests[i][1].similarity(digests[j][1])
            pairs.append((sim, digests[i][0], digests[j][0]))
    pairs.sort(reverse=True)
    print("  Top-5 most similar token sequence pairs:")
    for sim, id1, id2 in pairs[:5]:
        print(f"    {sim:.4f}  {id1[:50]:50s} vs {id2[:50]}")


# ─── 6. TokenCTPH Performance ────────────────────────────────────────────────


def eval_token_ctph_performance():
    section("6. Token CTPH Performance")

    ctph = TokenCTPH(4, 8)

    subsection("6a. Throughput by sequence length")
    for n_tokens in [100, 1000, 10000, 50000]:
        tokens = list(range(n_tokens))
        n_iters = max(1, min(10000, 1_000_000 // max(n_tokens, 1)))
        t0 = time.perf_counter()
        for _ in range(n_iters):
            ctph.compute(tokens)
        elapsed = time.perf_counter() - t0
        per_call = elapsed / n_iters
        tokens_per_sec = n_tokens / per_call
        print(
            f"  {n_tokens:6d} tokens: {per_call * 1000:.3f} ms/call, "
            f"{tokens_per_sec:,.0f} tokens/sec"
        )

    subsection("6b. Corpus-level (USC 1K sections as token hashes)")
    usc_docs = load_jsonl("usc.jsonl", max_docs=1000)
    token_seqs = [[hash(w) % 100000 for w in d["text"].split()] for d in usc_docs]
    total_tokens = sum(len(s) for s in token_seqs)

    tracemalloc.start()
    t0 = time.perf_counter()
    _digests = [ctph.compute(seq) for seq in token_seqs]
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(
        f"  1K docs, {total_tokens:,} total tokens: {elapsed:.3f}s, "
        f"{len(usc_docs) / elapsed:,.0f} docs/sec, "
        f"{total_tokens / elapsed:,.0f} tokens/sec, "
        f"peak_mem={fmt_bytes(peak)}"
    )


# ─── 7. Memory footprint ─────────────────────────────────────────────────────


def eval_memory():
    section("7. Memory Footprint")

    hasher = MinHasher(128, 42)

    for label, n_docs in [("1K", 1000), ("5K", 5000), ("10K", 10000)]:
        usc_docs = load_jsonl("usc.jsonl", max_docs=n_docs)

        # MinHash signatures
        tracemalloc.start()
        sigs = []
        for doc in usc_docs:
            words = doc["text"].split()
            sig = hasher.hash_token_shingles(words, 3)
            sigs.append(sig)
        _, sig_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # LSH index
        tracemalloc.start()
        index = MinHashIndex.with_threshold(128, 0.5)
        for i, sig in enumerate(sigs):
            index.insert(i, sig)
        _, idx_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"  {label} docs: signatures={fmt_bytes(sig_peak)}, LSH_index={fmt_bytes(idx_peak)}")


# ─── 8. EDGAR & Patent dedup quality ─────────────────────────────────────────


def eval_dedup_real_corpora():
    section("8. Deduplication on Real Corpora")

    hasher = MinHasher(128, 42)

    for corpus_name, file_name in [
        ("EDGAR Agreements (200)", "edgar_agreements.jsonl"),
        ("Patents (200)", "patents.jsonl"),
    ]:
        docs = load_jsonl(file_name, max_docs=200)
        doc_tuples = [(i, docs[i]["text"].split()) for i in range(len(docs))]

        subsection(f"{corpus_name}")

        for threshold in [0.3, 0.5, 0.8]:
            t0 = time.perf_counter()
            groups = find_duplicates(hasher, doc_tuples, shingle_size=3, threshold=threshold)
            elapsed = time.perf_counter() - t0

            n_dupes = sum(len(g.duplicates) for g in groups)
            print(
                f"  threshold={threshold}: {len(groups)} groups, "
                f"{n_dupes} duplicates, {elapsed:.3f}s"
            )

            if groups and threshold == 0.5:
                for g in groups[:2]:
                    canonical = g.canonical_id
                    ident = docs[canonical].get(
                        "identifier", docs[canonical].get("title", f"doc_{canonical}")
                    )
                    print(f"    canonical: {str(ident)[:70]}")
                    for d_id, sim in g.duplicates[:3]:
                        d_ident = docs[d_id].get(
                            "identifier", docs[d_id].get("title", f"doc_{d_id}")
                        )
                        print(f"      dup: {str(d_ident)[:60]} (sim={sim:.3f})")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print("KAOS-NLP-CORE: Fuzzy Hashing Quality & Performance Evaluation")
    print(f"Python: {sys.version}")
    print(f"Fixtures: {FIXTURES_DIR}")

    eval_minhash_quality()
    eval_minhash_performance()
    eval_ctph_quality()
    eval_ctph_performance()
    eval_token_ctph_quality()
    eval_token_ctph_performance()
    eval_memory()
    eval_dedup_real_corpora()

    section("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
