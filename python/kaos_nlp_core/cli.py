"""Command-line interface for kaos-nlp-core.

Every command supports --json for structured output (pipe-friendly).
Without --json, output is human-readable.

Usage:
    kaos-nlp tokenize FILE [--lowercase] [--keep-punctuation] [--json]
    kaos-nlp segment FILE [--mode sentences|lines|paragraphs] [--json]
    kaos-nlp compare TEXT1 TEXT2 [--algorithm jaro-winkler|...] [--json]
    kaos-nlp find PATTERN FILE [--case-insensitive] [--json]
    kaos-nlp search --index INDEX_PATH QUERY [--top-k 10] [--scoring bm25|tfidf] [--json]
    kaos-nlp index build FILE [--output index.kncidx] [--format auto|native|json] [--json]
    kaos-nlp hash FILE [--algorithm ctph|minhash] [--json]
    kaos-nlp duplicates DIR [--threshold 0.5] [--json]
    kaos-nlp encode TEXT [--algorithm soundex|metaphone] [--json]
    kaos-nlp vocab build FILE [--type frequency|indexed|bloom] [--json]
    kaos-nlp analyze FILE [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> None:
    """Entry point for kaos-nlp CLI."""
    parser = argparse.ArgumentParser(
        prog="kaos-nlp",
        description="High-performance NLP primitives for text analysis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- tokenize ---
    p_tokenize = sub.add_parser("tokenize", help="Tokenize text from a file")
    p_tokenize.add_argument("file", type=Path, help="Text file path")
    p_tokenize.add_argument("--lowercase", action="store_true", help="Lowercase tokens")
    p_tokenize.add_argument(
        "--keep-punctuation", action="store_true", help="Keep punctuation on tokens"
    )
    p_tokenize.add_argument("--json", action="store_true", help="Structured JSON output")

    # --- segment ---
    p_segment = sub.add_parser("segment", help="Segment text into sentences, lines, or paragraphs")
    p_segment.add_argument("file", type=Path, help="Text file path")
    p_segment.add_argument(
        "--mode",
        choices=["sentences", "lines", "paragraphs"],
        default="sentences",
        help="Segmentation mode (default: sentences)",
    )
    p_segment.add_argument("--json", action="store_true", help="Structured JSON output")

    # --- compare ---
    p_compare = sub.add_parser("compare", help="Compare two strings using similarity algorithms")
    p_compare.add_argument("text1", help="First string")
    p_compare.add_argument("text2", help="Second string")
    p_compare.add_argument(
        "--algorithm",
        choices=[
            "jaro-winkler",
            "levenshtein",
            "damerau",
            "hamming",
            "jaro",
            "dice",
            "soundex",
            "metaphone",
        ],
        default="jaro-winkler",
        help="Algorithm (default: jaro-winkler)",
    )
    p_compare.add_argument("--json", action="store_true", help="Structured JSON output")

    # --- find ---
    p_find = sub.add_parser("find", help="Find pattern occurrences in a file")
    p_find.add_argument("pattern", help="Pattern to search for")
    p_find.add_argument("file", type=Path, help="Text file path")
    p_find.add_argument(
        "--case-insensitive", "-i", action="store_true", help="Case-insensitive search"
    )
    p_find.add_argument("--json", action="store_true", help="Structured JSON output")

    # --- search ---
    p_search = sub.add_parser("search", help="Search a pre-built index")
    p_search.add_argument("--index", required=True, type=Path, help="Path to index file")
    p_search.add_argument("query", help="Search query string")
    p_search.add_argument("--top-k", "-k", type=int, default=10, help="Max results (default: 10)")
    p_search.add_argument(
        "--scoring",
        choices=["bm25", "tfidf"],
        default="bm25",
        help="Scoring method (default: bm25)",
    )
    p_search.add_argument("--json", action="store_true", help="Structured JSON output")

    # --- index build ---
    p_index = sub.add_parser("index", help="Build a BM25 inverted index")
    index_sub = p_index.add_subparsers(dest="action", required=True)
    p_index_build = index_sub.add_parser("build", help="Build index from a corpus file")
    p_index_build.add_argument("file", type=Path, help="Corpus file (one document per line)")
    p_index_build.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("index.kncidx"),
        help="Output path (default: index.kncidx)",
    )
    p_index_build.add_argument(
        "--format",
        choices=["auto", "native", "json"],
        default="auto",
        help="Persistence format (default: infer from extension)",
    )
    p_index_build.add_argument("--json", action="store_true", help="Structured JSON output")

    # --- hash ---
    p_hash = sub.add_parser("hash", help="Compute fuzzy hash of a file")
    p_hash.add_argument("file", type=Path, help="Text file path")
    p_hash.add_argument(
        "--algorithm",
        choices=["ctph", "minhash"],
        default="ctph",
        help="Hash algorithm (default: ctph)",
    )
    p_hash.add_argument("--json", action="store_true", help="Structured JSON output")

    # --- duplicates ---
    p_dups = sub.add_parser("duplicates", help="Find near-duplicate text files in a directory")
    p_dups.add_argument("directory", type=Path, help="Directory containing .txt files")
    p_dups.add_argument(
        "--threshold", "-t", type=float, default=0.5, help="Similarity threshold (default: 0.5)"
    )
    p_dups.add_argument("--json", action="store_true", help="Structured JSON output")

    # --- encode ---
    p_encode = sub.add_parser("encode", help="Phonetic encoding of text")
    p_encode.add_argument("text", help="Text to encode")
    p_encode.add_argument(
        "--algorithm",
        choices=["soundex", "metaphone"],
        default="soundex",
        help="Encoding algorithm (default: soundex)",
    )
    p_encode.add_argument("--json", action="store_true", help="Structured JSON output")

    # --- vocab build ---
    p_vocab = sub.add_parser("vocab", help="Build vocabulary from a file")
    vocab_sub = p_vocab.add_subparsers(dest="action", required=True)
    p_vocab_build = vocab_sub.add_parser("build", help="Build vocabulary from file tokens")
    p_vocab_build.add_argument("file", type=Path, help="Text file path")
    p_vocab_build.add_argument(
        "--type",
        choices=["frequency", "indexed", "bloom"],
        default="frequency",
        dest="vocab_type",
        help="Vocabulary type (default: frequency)",
    )
    p_vocab_build.add_argument(
        "--top-n", type=int, default=20, help="Show top N terms (default: 20)"
    )
    p_vocab_build.add_argument("--json", action="store_true", help="Structured JSON output")

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Comprehensive text analysis")
    p_analyze.add_argument("file", type=Path, help="Text file path")
    p_analyze.add_argument("--json", action="store_true", help="Structured JSON output")

    args = parser.parse_args(argv)

    handlers: dict[str, Any] = {
        "tokenize": _cmd_tokenize,
        "segment": _cmd_segment,
        "compare": _cmd_compare,
        "find": _cmd_find,
        "search": _cmd_search,
        "index": _cmd_index,
        "hash": _cmd_hash,
        "duplicates": _cmd_duplicates,
        "encode": _cmd_encode,
        "vocab": _cmd_vocab,
        "analyze": _cmd_analyze,
    }

    try:
        handlers[args.command](args)
    except FileNotFoundError as exc:
        _error(str(exc))
    except Exception as exc:
        _error(str(exc))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _json_out(data: dict[str, Any]) -> None:
    """Print structured JSON to stdout."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _error(msg: str) -> None:
    """Print error to stderr and exit."""
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _validate_file(path: Path) -> Path:
    """Validate that file exists and return resolved path."""
    path = path.resolve()
    if not path.is_file():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    return path


def _validate_dir(path: Path) -> Path:
    """Validate that directory exists and return resolved path."""
    path = path.resolve()
    if not path.is_dir():
        msg = f"Directory not found: {path}"
        raise FileNotFoundError(msg)
    return path


def _read_text(path: Path) -> str:
    """Read text from a file."""
    return path.read_text(encoding="utf-8")


def _detect_index_format(path: Path, requested: str) -> str:
    """Resolve the index persistence format from CLI settings."""
    if requested != "auto":
        return requested
    if path.suffix == ".json":
        return "json"
    return "native"


def _native_docs_path(index_path: Path) -> Path:
    """Return the sidecar path used for stored document text."""
    return index_path.with_name(f"{index_path.name}.docs.json")


# ---------------------------------------------------------------------------
# Commands — Phase 1: Core
# ---------------------------------------------------------------------------


def _cmd_tokenize(args: argparse.Namespace) -> None:
    """Tokenize text from a file."""
    from kaos_nlp_core.tokenizer import Tokenizer, tokenize

    path = _validate_file(args.file)
    text = _read_text(path)

    if args.keep_punctuation:
        # Convenience function doesn't support keep_punctuation, use Tokenizer class
        tok = Tokenizer(lowercase=args.lowercase, keep_punctuation=True)
        tokens = tok.tokenize(text)
    else:
        tokens = tokenize(text, lowercase=args.lowercase)

    if args.json:
        _json_out(
            {
                "command": "tokenize",
                "file": path.name,
                "total": len(tokens),
                "tokens": [{"text": t.text, "start": t.start, "end": t.end} for t in tokens],
            }
        )
        return

    for t in tokens:
        print(f"{t.text}\t{t.start}:{t.end}")


def _cmd_segment(args: argparse.Namespace) -> None:
    """Segment text from a file."""
    from kaos_nlp_core.segmentation import segment_lines, segment_paragraphs, segment_sentences

    path = _validate_file(args.file)
    text = _read_text(path)

    segmenters = {
        "sentences": segment_sentences,
        "lines": segment_lines,
        "paragraphs": segment_paragraphs,
    }

    raw_segments = segmenters[args.mode](text)

    if args.json:
        _json_out(
            {
                "command": "segment",
                "file": path.name,
                "mode": args.mode,
                "total": len(raw_segments),
                "segments": [
                    {"index": i + 1, "start": s.start, "end": s.end, "text": s.text}
                    for i, s in enumerate(raw_segments)
                ],
            }
        )
        return

    for i, seg in enumerate(raw_segments):
        preview = seg.text[:200]
        if len(seg.text) > 200:
            preview += "..."
        print(f"[{i + 1}] ({seg.start}:{seg.end}) {preview}")


def _cmd_compare(args: argparse.Namespace) -> None:
    """Compare two strings using similarity algorithms."""
    from kaos_nlp_core.algorithms import (
        damerau_levenshtein,
        hamming,
        jaro,
        jaro_winkler,
        levenshtein,
        metaphone_distance,
        sorensen_dice,
        soundex_distance,
    )

    algo_map = {
        "jaro-winkler": jaro_winkler,
        "levenshtein": levenshtein,
        "damerau": damerau_levenshtein,
        "hamming": hamming,
        "jaro": jaro,
        "dice": sorensen_dice,
        "soundex": soundex_distance,
        "metaphone": metaphone_distance,
    }

    func = algo_map[args.algorithm]
    result = func(args.text1, args.text2)

    if args.json:
        _json_out(
            {
                "command": "compare",
                "text1": args.text1,
                "text2": args.text2,
                "algorithm": args.algorithm,
                "distance": result.distance,
                "normalized": result.normalized,
                "similarity": result.similarity,
            }
        )
        return

    print(f"Algorithm:  {args.algorithm}")
    print(f"Text 1:     {args.text1!r}")
    print(f"Text 2:     {args.text2!r}")
    print(f"Distance:   {result.distance:.4f}")
    print(f"Normalized: {result.normalized:.4f}")
    print(f"Similarity: {result.similarity:.4f}")


def _cmd_find(args: argparse.Namespace) -> None:
    """Find pattern occurrences in a file."""
    from kaos_nlp_core.matching import substring_find_all, substring_find_all_case_insensitive

    path = _validate_file(args.file)
    text = _read_text(path)

    if args.case_insensitive:
        matches = substring_find_all_case_insensitive(text, args.pattern)
    else:
        matches = substring_find_all(text, args.pattern)

    if args.json:
        _json_out(
            {
                "command": "find",
                "file": path.name,
                "pattern": args.pattern,
                "total": len(matches),
                "matches": [{"start": m.start, "end": m.end, "text": m.text} for m in matches],
            }
        )
        return

    if not matches:
        print(f'No matches for "{args.pattern}"')
        return

    print(f'Found {len(matches)} match(es) for "{args.pattern}":\n')
    lines = text.splitlines(keepends=True)
    for i, m in enumerate(matches, 1):
        # Find line number for this match
        char_count = 0
        line_num = 1
        for ln, line in enumerate(lines, 1):
            if char_count + len(line) > m.start:
                line_num = ln
                break
            char_count += len(line)

        # Show context: the line containing the match
        line_text = lines[line_num - 1].rstrip("\n\r") if line_num <= len(lines) else ""
        print(f"  [{i}] Line {line_num}, offset {m.start}:{m.end}")
        print(f"       {line_text}")
        print(f"       Match: {m.text!r}")
        print()


def _cmd_search(args: argparse.Namespace) -> None:
    """Search a pre-built index."""
    from kaos_nlp_core.documents import DocumentCollection
    from kaos_nlp_core.search import Searcher
    from kaos_nlp_core.structures import InvertedIndex

    index_path = _validate_file(args.index)
    index_format = _detect_index_format(index_path, "auto")

    idx: InvertedIndex
    collection: DocumentCollection | None = None

    if index_format == "json":
        index_data = json.loads(_read_text(index_path))
        idx = InvertedIndex()
        for doc_id_str, doc_terms in index_data.get("index_data", {}).items():
            idx.add_document(int(doc_id_str), doc_terms)
        if index_data.get("documents"):
            records = [
                {"id": int(doc_id), "text": text}
                for doc_id, text in index_data["documents"].items()
            ]
            collection = DocumentCollection.from_records(records)
    else:
        idx = InvertedIndex.load(str(index_path))
        docs_path = _native_docs_path(index_path)
        if docs_path.exists():
            collection = DocumentCollection.from_records(json.loads(_read_text(docs_path)))

    searcher = Searcher(index=idx, collection=collection, scoring=args.scoring)
    results = searcher.search(args.query, top_k=args.top_k)

    if args.json:
        result_list = []
        for r in results:
            entry: dict[str, Any] = {"doc_id": r.doc_id, "score": r.score}
            if r.text:
                entry["text"] = r.text[:500]
            if r.external_id is not None:
                entry["external_id"] = r.external_id
            result_list.append(entry)
        _json_out(
            {
                "command": "search",
                "index": index_path.name,
                "format": index_format,
                "scoring": args.scoring,
                "query": args.query,
                "total": len(results),
                "results": result_list,
            }
        )
        return

    if not results:
        print(f'No results for "{args.query}"')
        return

    print(f'Results for "{args.query}" ({len(results)} matches):\n')
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Doc {r.doc_id} (score: {r.score:.4f})")
        preview = r.text
        if len(preview) > 200:
            preview = preview[:200] + "..."
        if preview:
            print(f"       {preview}")
        print()


def _cmd_index(args: argparse.Namespace) -> None:
    """Build a BM25 inverted index from a corpus file."""
    from kaos_nlp_core.documents import DocumentCollection
    from kaos_nlp_core.tokenizer import Tokenizer

    path = _validate_file(args.file)
    text = _read_text(path)

    # Split corpus: one document per line (skip blank lines)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    output_path = args.output.resolve()
    output_format = _detect_index_format(output_path, args.format)

    tokenizer = Tokenizer(lowercase=True)
    records = [{"id": doc_id, "text": line} for doc_id, line in enumerate(lines)]
    collection = DocumentCollection.from_records(records)
    idx = collection.build_index(tokenizer=tokenizer)

    documents_path: str | None = None
    if output_format == "json":
        doc_terms_map: dict[str, list[str]] = {}
        documents: dict[str, str] = {}
        for record in records:
            doc_id = int(record["id"])
            doc_terms_map[str(doc_id)] = tokenizer.tokenize_words(str(record["text"]))
            documents[str(doc_id)] = str(record["text"])
        index_data = {
            "documents": documents,
            "index_data": doc_terms_map,
            "total_documents": len(lines),
            "total_terms": idx.term_count(),
        }
        output_path.write_text(json.dumps(index_data, ensure_ascii=False), encoding="utf-8")
    else:
        idx.save(str(output_path))
        docs_path = _native_docs_path(output_path)
        docs_path.write_text(
            json.dumps(collection.to_records(), ensure_ascii=False), encoding="utf-8"
        )
        documents_path = str(docs_path)

    if args.json:
        _json_out(
            {
                "command": "index",
                "action": "build",
                "file": path.name,
                "format": output_format,
                "documents": len(lines),
                "terms": idx.term_count(),
                "output": str(output_path),
                "documents_path": documents_path,
            }
        )
        return

    print(f"Built index from {path.name}")
    print(f"Format:    {output_format}")
    print(f"Documents: {len(lines)}")
    print(f"Terms:     {idx.term_count()}")
    print(f"Output:    {output_path}")
    if documents_path:
        print(f"Docs:      {documents_path}")


# ---------------------------------------------------------------------------
# Commands — Phase 2: Hashing & Dedup
# ---------------------------------------------------------------------------


def _cmd_hash(args: argparse.Namespace) -> None:
    """Compute fuzzy hash of a file."""
    from kaos_nlp_core.hashing import MinHasher, ctph_hash_str

    path = _validate_file(args.file)
    text = _read_text(path)

    if args.algorithm == "ctph":
        hash_value = ctph_hash_str(text, 64, 8, 4)
        if args.json:
            _json_out(
                {
                    "command": "hash",
                    "file": path.name,
                    "algorithm": "ctph",
                    "hash": hash_value,
                }
            )
            return
        print(f"File:      {path.name}")
        print("Algorithm: CTPH")
        print(f"Hash:      {hash_value}")

    else:  # minhash
        hasher = MinHasher(128, 42)
        sig = hasher.hash_char_shingles(text, 3)
        values = sig.values
        # Show summary: first/last few values and length
        preview = values[:8]
        if args.json:
            _json_out(
                {
                    "command": "hash",
                    "file": path.name,
                    "algorithm": "minhash",
                    "num_permutations": len(values),
                    "signature_preview": preview,
                }
            )
            return
        print(f"File:            {path.name}")
        print("Algorithm:       MinHash (128 permutations, k=3 shingles)")
        print(f"Signature[0:8]:  {preview}")


def _cmd_duplicates(args: argparse.Namespace) -> None:
    """Find near-duplicate text files in a directory."""
    from kaos_nlp_core.hashing import MinHasher, find_duplicates
    from kaos_nlp_core.tokenizer import tokenize_words

    dir_path = _validate_dir(args.directory)
    txt_files = sorted(dir_path.glob("*.txt"))

    if not txt_files:
        _error(f"No .txt files found in {dir_path}")

    hasher = MinHasher(128, 42)
    docs: list[tuple[int, list[str]]] = []
    file_map: dict[int, str] = {}

    for i, f in enumerate(txt_files):
        text = f.read_text(encoding="utf-8")
        terms = tokenize_words(text, lowercase=True)
        docs.append((i, terms))
        file_map[i] = f.name

    groups = find_duplicates(hasher, docs, shingle_size=2, threshold=args.threshold)

    if args.json:
        group_list = []
        for g in groups:
            group_list.append(
                {
                    "canonical": g.canonical_id,
                    "canonical_file": file_map.get(g.canonical_id, ""),
                    "duplicates": [
                        {"doc_id": did, "file": file_map.get(did, ""), "similarity": sim}
                        for did, sim in g.duplicates
                    ],
                }
            )
        _json_out(
            {
                "command": "duplicates",
                "directory": dir_path.name,
                "threshold": args.threshold,
                "files_scanned": len(txt_files),
                "groups": group_list,
            }
        )
        return

    if not groups:
        print(f"No duplicates found (threshold={args.threshold})")
        return

    print(f"Duplicate groups (threshold={args.threshold}):\n")
    for gi, g in enumerate(groups, 1):
        print(f"Group {gi}:")
        print(f"  Canonical: [{g.canonical_id}] {file_map.get(g.canonical_id, '?')}")
        for did, sim in g.duplicates:
            print(f"  Duplicate: [{did}] {file_map.get(did, '?')} (similarity: {sim:.4f})")
        print()


# ---------------------------------------------------------------------------
# Commands — Phase 3: Specialized
# ---------------------------------------------------------------------------


def _cmd_encode(args: argparse.Namespace) -> None:
    """Phonetic encoding of text."""
    from kaos_nlp_core.algorithms import metaphone_encode, soundex_encode

    encoders = {
        "soundex": soundex_encode,
        "metaphone": metaphone_encode,
    }

    encoding = encoders[args.algorithm](args.text)

    if args.json:
        _json_out(
            {
                "command": "encode",
                "text": args.text,
                "algorithm": args.algorithm,
                "encoding": encoding,
            }
        )
        return

    print(f"Text:      {args.text!r}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Encoding:  {encoding}")


def _cmd_vocab(args: argparse.Namespace) -> None:
    """Build vocabulary from a file."""
    from kaos_nlp_core.structures import BloomVocabulary, FrequencyVocabulary, IndexedVocabulary
    from kaos_nlp_core.tokenizer import tokenize_words

    path = _validate_file(args.file)
    text = _read_text(path)
    words = tokenize_words(text, lowercase=True)

    vocab_type = args.vocab_type
    top_n = args.top_n

    if vocab_type == "frequency":
        vocab = FrequencyVocabulary()
        for w in words:
            vocab.insert(w)
        unique_terms = len(vocab)
        total_terms = vocab.total_count()
        top_terms = vocab.top_n(top_n)

        if args.json:
            _json_out(
                {
                    "command": "vocab",
                    "action": "build",
                    "file": path.name,
                    "type": "frequency",
                    "total_terms": total_terms,
                    "unique_terms": unique_terms,
                    "top_terms": [{"term": t, "count": c} for t, c in top_terms],
                }
            )
            return

        print(f"File:         {path.name}")
        print("Type:         frequency")
        print(f"Total terms:  {total_terms}")
        print(f"Unique terms: {unique_terms}")
        print(f"\nTop {min(top_n, unique_terms)} terms:")
        for term, count in top_terms:
            print(f"  {term:<30s} {count:>8d}")

    elif vocab_type == "indexed":
        vocab_idx = IndexedVocabulary()
        for w in words:
            vocab_idx.insert(w)
        unique_terms = len(vocab_idx)

        if args.json:
            _json_out(
                {
                    "command": "vocab",
                    "action": "build",
                    "file": path.name,
                    "type": "indexed",
                    "total_terms": len(words),
                    "unique_terms": unique_terms,
                }
            )
            return

        print(f"File:         {path.name}")
        print("Type:         indexed")
        print(f"Total terms:  {len(words)}")
        print(f"Unique terms: {unique_terms}")

    else:  # bloom
        bloom = BloomVocabulary(max(1000, len(words)), 0.01)
        for w in words:
            bloom.insert(w)

        if args.json:
            _json_out(
                {
                    "command": "vocab",
                    "action": "build",
                    "file": path.name,
                    "type": "bloom",
                    "total_terms": len(words),
                    "approx_unique": bloom.approx_len(),
                }
            )
            return

        print(f"File:           {path.name}")
        print("Type:           bloom")
        print(f"Total terms:    {len(words)}")
        print(f"Approx unique:  {bloom.approx_len()}")


def _cmd_analyze(args: argparse.Namespace) -> None:
    """Comprehensive text analysis."""
    from kaos_nlp_core.segmentation import segment_lines, segment_paragraphs, segment_sentences
    from kaos_nlp_core.structures import FrequencyVocabulary
    from kaos_nlp_core.token_properties import classify_tokens
    from kaos_nlp_core.tokenizer import tokenize, tokenize_words

    path = _validate_file(args.file)
    text = _read_text(path)

    # Basic stats
    char_count = len(text)
    tokens = tokenize(text, lowercase=True)
    token_count = len(tokens)
    words = tokenize_words(text, lowercase=True)
    token_flags = classify_tokens(words)
    property_counts = {
        "abbreviations": sum(1 for flags in token_flags if flags.is_abbreviation),
        "numeric": sum(1 for flags in token_flags if flags.is_numeric_word),
        "alphanumeric": sum(1 for flags in token_flags if flags.is_alphanumeric_word),
        "uppercase": sum(1 for flags in token_flags if flags.is_uppercase_word),
        "emoji": sum(1 for flags in token_flags if flags.has_emoji),
    }

    # Segmentation
    sentences = segment_sentences(text)
    lines = segment_lines(text)
    paragraphs = segment_paragraphs(text)

    # Vocabulary richness
    vocab = FrequencyVocabulary()
    for w in words:
        vocab.insert(w)
    unique_terms = len(vocab)
    type_token_ratio = unique_terms / token_count if token_count > 0 else 0.0

    # Average sentence length
    avg_sentence_len = token_count / len(sentences) if sentences else 0.0

    # Top terms
    top_terms = vocab.top_n(10)

    if args.json:
        _json_out(
            {
                "command": "analyze",
                "file": path.name,
                "characters": char_count,
                "tokens": token_count,
                "unique_terms": unique_terms,
                "token_properties": property_counts,
                "sentences": len(sentences),
                "lines": len(lines),
                "paragraphs": len(paragraphs),
                "avg_sentence_length": round(avg_sentence_len, 2),
                "type_token_ratio": round(type_token_ratio, 4),
                "top_terms": [{"term": t, "count": c} for t, c in top_terms],
            }
        )
        return

    print(f"File:                {path.name}")
    print(f"Characters:          {char_count:,}")
    print(f"Tokens:              {token_count:,}")
    print(f"Unique terms:        {unique_terms:,}")
    print(f"Abbreviations:       {property_counts['abbreviations']:,}")
    print(f"Numeric tokens:      {property_counts['numeric']:,}")
    print(f"Sentences:           {len(sentences):,}")
    print(f"Lines:               {len(lines):,}")
    print(f"Paragraphs:          {len(paragraphs):,}")
    print(f"Avg sentence length: {avg_sentence_len:.1f} tokens")
    print(f"Type-token ratio:    {type_token_ratio:.4f}")
    print("\nTop 10 terms:")
    for term, count in top_terms:
        print(f"  {term:<30s} {count:>8d}")
