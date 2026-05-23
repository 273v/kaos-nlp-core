"""MCP tool definitions for kaos-nlp-core.

Provides 17 MCP tools for high-performance NLP primitives: tokenization,
segmentation, string comparison, pattern matching, BM25 search, fuzzy
hashing, near-duplicate detection, text analysis, text quality scoring,
lexicon lookup/expansion, token frequency, concept extraction, line
labeling, and document outline. All tools operate on raw text (no
runtime/context required) and are read-only except build-index.

All kaos-core imports are lazy (inside register_nlp_tools) so that
kaos-nlp-core can be used standalone without kaos-core installed.

## Sandbox model for file-touching tools (F3)

`kaos-nlp-build-index` is the only tool that reads or writes the local
filesystem. Its inputs are confined to a workspace root:

- `KAOS_NLP_WORKSPACE_ROOT` defaults to the current working directory.
- All `corpus_path` / `output_path` inputs are resolved against this root
  and rejected if the resulting absolute path escapes it (path traversal
  guard).
- The corpus file size is capped by `KAOS_NLP_MAX_CORPUS_BYTES`
  (default 256 MiB).

These constraints apply equally to stdio and HTTP transports. See
`kaos-nlp-serve --http` for the operator-acknowledgement gate that
controls when the HTTP transport may even start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_MODULE = "kaos-nlp-core"


def _resolve_version() -> str:
    """Return the installed `kaos-nlp-core` version, or `unknown` if absent.

    Derived via `importlib.metadata.version` so the MCP tool metadata always
    matches the wheel/sdist version. We can't import `kaos_nlp_core.__version__`
    here because this module is imported from `kaos_nlp_core/__init__.py`
    (`register_nlp_tools`), so the package's `__version__` attribute isn't
    bound yet at this import point.

    A hardcoded literal here drifted in earlier releases —
    audit-04/kaos-nlp-core.md F-001 caught the test pinning the stale
    "0.1.0a1" value while the package had moved to 0.1.1.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        return _version("kaos-nlp-core")
    except PackageNotFoundError:
        return "unknown"


_VERSION = _resolve_version()


def _workspace_root_from_settings(settings: Any) -> Path:
    """Resolve the active workspace root from a ``KaosNlpSettings``-like object.

    ``settings.workspace_root is None`` falls back to ``Path.cwd().resolve()``
    so the historical "default to CWD" behaviour is preserved when the
    operator has not configured one. ``settings`` is duck-typed: anything
    with a ``workspace_root`` attribute works.
    """
    raw = getattr(settings, "workspace_root", None) if settings is not None else None
    return Path(raw).resolve() if raw else Path.cwd().resolve()


def _resolve_within_root(path_str: str, root: Path) -> Path:
    """Resolve ``path_str`` and enforce that the result is inside ``root``.

    Raises ``ValueError`` on traversal outside the root.
    """
    resolved = Path(path_str).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"path '{path_str}' resolves to '{resolved}', which is outside the "
            f"workspace root '{root}'. Set KAOS_NLP_WORKSPACE_ROOT to widen the "
            "allowed area, or pass a path inside the current root."
        ) from exc
    return resolved


# Algorithms supported by kaos-nlp-compare
_COMPARE_ALGORITHMS = [
    "levenshtein",
    "damerau_levenshtein",
    "jaro",
    "jaro_winkler",
    "soundex",
    "metaphone",
    "lcs",
    "ngram_jaccard",
    "sorensen_dice",
]


def register_nlp_tools(runtime: Any) -> int:
    """Register kaos-nlp-core MCP tools with a KaosRuntime.

    All 17 tools are defined inside this function to keep kaos-core imports
    lazy. Returns the count of registered tools.

    Args:
        runtime: A ``KaosRuntime`` instance.

    Returns:
        Number of tools registered.

    Raises:
        ImportError: If kaos-core is not installed.
    """
    try:
        from kaos_core.base.context import KaosContext
        from kaos_core.base.tool import KaosTool
        from kaos_core.types.annotations import ToolAnnotations
        from kaos_core.types.enums import ToolCapability, ToolCategory
        from kaos_core.types.metadata import ToolMetadata
        from kaos_core.types.parameters import ParameterSchema
        from kaos_core.types.results import ToolResult
    except ImportError as e:
        raise ImportError(
            "kaos-core is required for MCP tool registration. Install with: pip install kaos-core"
        ) from e

    def _settings_for(context: KaosContext | None) -> Any:
        """Settings instance scoped to the current MCP context.

        Threads ``KaosContext._config`` (per-request ``_meta.kaos_config``)
        through to ``KaosNlpSettings`` so MCP callers can override the
        workspace root + corpus cap without env-var edits. Standalone
        callers (``register_nlp_tools(runtime)`` outside an MCP request)
        get plain env-var defaults.
        """
        from kaos_nlp_core.settings import KaosNlpSettings

        return KaosNlpSettings.from_context(context)

    # Shared annotations
    _NLP_RO_ANNOTATIONS = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
    _NLP_WR_ANNOTATIONS = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    # ── 1. kaos-nlp-tokenize ─────────────────────────────────────────

    class TokenizeTool(KaosTool):
        """Tokenize text into words with character offsets."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-tokenize",
                display_name="Tokenize Text",
                description=(
                    "Tokenize text into words with character offsets. "
                    "Uses Unicode-aware whitespace and punctuation splitting. "
                    "Returns tokens with start/end character positions. "
                    "For string similarity, use kaos-nlp-compare. "
                    "For text statistics, use kaos-nlp-analyze-text."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text",
                        type="string",
                        description="Text to tokenize.",
                    ),
                    ParameterSchema(
                        name="lowercase",
                        type="boolean",
                        description="Lowercase all tokens (default: false).",
                        required=False,
                        default=False,
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.tokenizer import tokenize

            text = inputs.get("text")
            if not text:
                return ToolResult.create_error(
                    "Parameter 'text' is required and must be non-empty. "
                    "Provide the text you want to tokenize."
                )

            lowercase = inputs.get("lowercase", False)

            try:
                tokens = tokenize(text, lowercase=lowercase)
            except Exception as exc:
                return ToolResult.create_error(
                    f"Tokenization failed: {exc}. Ensure the input is valid text."
                )

            output = {
                "tokens": [{"text": t.text, "start": t.start, "end": t.end} for t in tokens],
                "count": len(tokens),
            }
            return ToolResult.create_success(
                output, summary=f"Tokenized into {len(tokens)} token(s)"
            )

    # ── 2. kaos-nlp-segment-sentences ────────────────────────────────

    class SegmentSentencesTool(KaosTool):
        """Segment text into sentences with character offsets."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-segment-sentences",
                display_name="Segment Sentences",
                description=(
                    "Segment text into sentences using a Punkt tokenizer "
                    "trained on legal/formal text. Returns sentences with "
                    "character offsets. For paragraph segmentation, use "
                    "kaos-nlp-segment-paragraphs. For BM25 search within "
                    "sentences, use kaos-nlp-search-text."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text",
                        type="string",
                        description="Text to segment into sentences.",
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.segmentation import segment_sentences

            text = inputs.get("text")
            if not text:
                return ToolResult.create_error(
                    "Parameter 'text' is required and must be non-empty. "
                    "Provide the text you want to segment."
                )

            try:
                segments = segment_sentences(text)
            except Exception as exc:
                return ToolResult.create_error(
                    f"Sentence segmentation failed: {exc}. Ensure the input is valid text."
                )

            output = {
                "segments": [{"text": s.text, "start": s.start, "end": s.end} for s in segments],
                "count": len(segments),
            }
            return ToolResult.create_success(
                output, summary=f"Segmented into {len(segments)} sentence(s)"
            )

    # ── 3. kaos-nlp-segment-paragraphs ───────────────────────────────

    class SegmentParagraphsTool(KaosTool):
        """Segment text into paragraphs with character offsets."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-segment-paragraphs",
                display_name="Segment Paragraphs",
                description=(
                    "Segment text into paragraphs using blank-line detection. "
                    "Returns paragraphs with character offsets. For sentence "
                    "segmentation, use kaos-nlp-segment-sentences. For BM25 "
                    "search within paragraphs, use kaos-nlp-search-text with "
                    "level='paragraphs'."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text",
                        type="string",
                        description="Text to segment into paragraphs.",
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.segmentation import segment_paragraphs_simple

            text = inputs.get("text")
            if not text:
                return ToolResult.create_error(
                    "Parameter 'text' is required and must be non-empty. "
                    "Provide the text you want to segment."
                )

            try:
                segments = segment_paragraphs_simple(text)
            except Exception as exc:
                return ToolResult.create_error(
                    f"Paragraph segmentation failed: {exc}. Ensure the input is valid text."
                )

            output = {
                "segments": [{"text": s.text, "start": s.start, "end": s.end} for s in segments],
                "count": len(segments),
            }
            return ToolResult.create_success(
                output, summary=f"Segmented into {len(segments)} paragraph(s)"
            )

    # ── 4. kaos-nlp-compare ──────────────────────────────────────────

    class CompareTool(KaosTool):
        """Compare two strings using distance/similarity algorithms."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-compare",
                display_name="Compare Strings",
                description=(
                    "Compare two strings using a distance/similarity algorithm. "
                    "Supported algorithms: levenshtein, damerau_levenshtein, "
                    "jaro, jaro_winkler (default), soundex, metaphone, lcs, "
                    "ngram_jaccard, sorensen_dice. Returns distance (0=identical), "
                    "normalized distance (0-1), and similarity (1=identical). "
                    "For near-duplicate detection across many texts, use "
                    "kaos-nlp-find-duplicates."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text1",
                        type="string",
                        description="First string to compare.",
                    ),
                    ParameterSchema(
                        name="text2",
                        type="string",
                        description="Second string to compare.",
                    ),
                    ParameterSchema(
                        name="algorithm",
                        type="string",
                        description="Comparison algorithm (default: jaro_winkler).",
                        required=False,
                        default="jaro_winkler",
                        constraints={"enum": _COMPARE_ALGORITHMS},
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core import algorithms as algo

            text1 = inputs.get("text1")
            text2 = inputs.get("text2")
            if text1 is None or text2 is None:
                return ToolResult.create_error(
                    "Both 'text1' and 'text2' are required. Provide two strings to compare."
                )

            algorithm = inputs.get("algorithm", "jaro_winkler")

            algo_map: dict[str, Any] = {
                "levenshtein": algo.levenshtein,
                "damerau_levenshtein": algo.damerau_levenshtein,
                "jaro": algo.jaro,
                "jaro_winkler": algo.jaro_winkler,
                "soundex": algo.soundex_distance,
                "metaphone": algo.metaphone_distance,
                "lcs": algo.lcs_distance,
                "ngram_jaccard": algo.ngram_jaccard,
                "sorensen_dice": algo.sorensen_dice,
            }

            func = algo_map.get(algorithm)
            if func is None:
                return ToolResult.create_error(
                    f"Unknown algorithm '{algorithm}'. Supported: {', '.join(_COMPARE_ALGORITHMS)}"
                )

            try:
                result = func(text1, text2)
            except Exception as exc:
                return ToolResult.create_error(
                    f"Comparison failed: {exc}. "
                    "Ensure both inputs are valid strings. "
                    "For hamming distance, strings must be equal length."
                )

            output = {
                "distance": result.distance,
                "normalized": result.normalized,
                "similarity": result.similarity,
                "algorithm": algorithm,
            }
            return ToolResult.create_success(
                output,
                summary=(
                    f"{algorithm}: similarity={result.similarity:.4f}, "
                    f"distance={result.distance:.4f}"
                ),
            )

    # ── 5. kaos-nlp-find-pattern ─────────────────────────────────────

    class FindPatternTool(KaosTool):
        """Find pattern occurrences in text."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-find-pattern",
                display_name="Find Pattern",
                description=(
                    "Find all occurrences of a pattern in text. Supports two "
                    "modes: 'substring' (SIMD-accelerated literal match) and "
                    "'regex' (Rust regex engine). Returns matches with character "
                    "offsets. For BM25-ranked search, use kaos-nlp-search-text."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.QUERY,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text",
                        type="string",
                        description="Text to search in.",
                    ),
                    ParameterSchema(
                        name="pattern",
                        type="string",
                        description="Pattern to search for (literal or regex).",
                    ),
                    ParameterSchema(
                        name="mode",
                        type="string",
                        description="Search mode (default: substring).",
                        required=False,
                        default="substring",
                        constraints={"enum": ["substring", "regex"]},
                    ),
                    ParameterSchema(
                        name="case_insensitive",
                        type="boolean",
                        description=(
                            "Case-insensitive search (default: false). "
                            "Only applies to substring mode."
                        ),
                        required=False,
                        default=False,
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            text = inputs.get("text")
            pattern = inputs.get("pattern")
            if not text or not pattern:
                return ToolResult.create_error(
                    "Both 'text' and 'pattern' are required. "
                    "Provide the text to search and the pattern to find."
                )

            mode = inputs.get("mode", "substring")
            case_insensitive = inputs.get("case_insensitive", False)

            try:
                if mode == "regex":
                    from kaos_nlp_core.matching import RegexMatcher

                    matcher = RegexMatcher(pattern)
                    matches = matcher.find_all(text)
                    match_list = [{"text": m.text, "start": m.start, "end": m.end} for m in matches]
                else:
                    from kaos_nlp_core.matching import (
                        substring_find_all,
                        substring_find_all_case_insensitive,
                    )

                    if case_insensitive:
                        matches = substring_find_all_case_insensitive(text, pattern)
                    else:
                        matches = substring_find_all(text, pattern)
                    match_list = [{"text": m.text, "start": m.start, "end": m.end} for m in matches]
            except Exception as exc:
                return ToolResult.create_error(
                    f"Pattern search failed: {exc}. "
                    + (
                        "Check that the regex pattern is valid. "
                        if mode == "regex"
                        else "Ensure both text and pattern are valid strings. "
                    )
                    + "For BM25-ranked search, use kaos-nlp-search-text."
                )

            output = {
                "matches": match_list,
                "count": len(match_list),
            }
            return ToolResult.create_success(
                output,
                summary=(f"Found {len(match_list)} match(es) for '{pattern}' ({mode})"),
            )

    # ── 6. kaos-nlp-search-text ──────────────────────────────────────

    class SearchTextTool(KaosTool):
        """BM25 search within text at sentence or paragraph level."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-search-text",
                display_name="Search Text",
                description=(
                    "Search within a text document using BM25 ranking. "
                    "Segments text into sentences or paragraphs, then "
                    "ranks segments by relevance to the query. Returns "
                    "top-k results with scores and character offsets. "
                    "For literal pattern matching, use kaos-nlp-find-pattern."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.QUERY,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text",
                        type="string",
                        description="Text document to search within.",
                    ),
                    ParameterSchema(
                        name="query",
                        type="string",
                        description="Search query (natural language).",
                    ),
                    ParameterSchema(
                        name="level",
                        type="string",
                        description=("Segmentation level for search (default: sentences)."),
                        required=False,
                        default="sentences",
                        constraints={"enum": ["sentences", "paragraphs"]},
                    ),
                    ParameterSchema(
                        name="top_k",
                        type="integer",
                        description="Maximum results to return (default: 10).",
                        required=False,
                        default=10,
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.search import search_paragraphs, search_sentences

            text = inputs.get("text")
            query = inputs.get("query")
            if not text or not query:
                return ToolResult.create_error(
                    "Both 'text' and 'query' are required. "
                    "Provide the document text and search query."
                )

            level = inputs.get("level", "sentences")
            top_k = inputs.get("top_k", 10)

            try:
                if level == "paragraphs":
                    hits = search_paragraphs(text, query, top_k=top_k)
                else:
                    hits = search_sentences(text, query, top_k=top_k)
            except Exception as exc:
                return ToolResult.create_error(
                    f"Search failed: {exc}. Ensure the text and query are valid strings."
                )

            output = {
                "results": [
                    {
                        "text": h.text,
                        "start": h.start,
                        "end": h.end,
                        "score": h.score,
                    }
                    for h in hits
                ],
                "total": len(hits),
            }
            return ToolResult.create_success(
                output,
                summary=f"Found {len(hits)} result(s) for '{query}' ({level})",
            )

    # ── 7. kaos-nlp-build-index ──────────────────────────────────────

    class BuildIndexTool(KaosTool):
        """Build a BM25 inverted index from a corpus file."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-build-index",
                display_name="Build Index",
                description=(
                    "Build a BM25 inverted index from a text corpus file "
                    "(one document per line). Saves the index to disk for "
                    "later use with kaos-nlp-search-text or the CLI "
                    "'kaos-nlp search' command. "
                    "For searching within a single document without building "
                    "an index, use kaos-nlp-search-text."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.TRANSFORM,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_WR_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="corpus_path",
                        type="string",
                        description=("Path to a text file with one document per line."),
                    ),
                    ParameterSchema(
                        name="output_path",
                        type="string",
                        description=(
                            "Output path for the index file "
                            "(default: index.kncidx in same directory)."
                        ),
                        required=False,
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            import json

            from kaos_nlp_core.documents import DocumentCollection
            from kaos_nlp_core.tokenizer import Tokenizer

            corpus_path_str = inputs.get("corpus_path")
            if not corpus_path_str:
                return ToolResult.create_error(
                    "Parameter 'corpus_path' is required. "
                    "Provide a path to a text file (one document per line)."
                )

            # F3 — confine I/O to KAOS_NLP_WORKSPACE_ROOT (default: CWD) and
            # enforce a configurable corpus size cap. Settings are threaded
            # through the MCP context so per-request _meta.kaos_config can
            # override workspace_root / max_corpus_bytes (audit follow-up
            # parity with kaos-graph A2-followup-#3).
            settings = _settings_for(context)
            root = _workspace_root_from_settings(settings)
            try:
                corpus_path = _resolve_within_root(corpus_path_str, root)
            except ValueError as exc:
                return ToolResult.create_error(str(exc))

            if not corpus_path.is_file():
                return ToolResult.create_error(
                    f"File not found: {corpus_path}. Verify the path exists "
                    f"and is readable inside the workspace root '{root}'."
                )

            max_bytes = int(settings.max_corpus_bytes)
            corpus_size = corpus_path.stat().st_size
            if corpus_size > max_bytes:
                return ToolResult.create_error(
                    f"Corpus file is {corpus_size} bytes; exceeds the "
                    f"KAOS_NLP_MAX_CORPUS_BYTES cap of {max_bytes} bytes. "
                    "Split the corpus or raise the cap explicitly."
                )

            output_path_str = inputs.get("output_path")
            if output_path_str:
                try:
                    output_path = _resolve_within_root(output_path_str, root)
                except ValueError as exc:
                    return ToolResult.create_error(str(exc))
            else:
                output_path = corpus_path.with_suffix(".kncidx")

            try:
                text = corpus_path.read_text(encoding="utf-8")
                lines = [line.strip() for line in text.splitlines() if line.strip()]

                if not lines:
                    return ToolResult.create_error(
                        f"Corpus file '{corpus_path.name}' is empty or contains "
                        "only blank lines. Each non-blank line is one document."
                    )

                tokenizer = Tokenizer(lowercase=True)
                records = [{"id": doc_id, "text": line} for doc_id, line in enumerate(lines)]
                collection = DocumentCollection.from_records(records)
                idx = collection.build_index(tokenizer=tokenizer)

                idx.save(str(output_path))

                # Save sidecar docs file inside the same directory; that
                # directory was already validated to be inside the root.
                docs_path = output_path.with_name(f"{output_path.name}.docs.json")
                docs_path.write_text(
                    json.dumps(collection.to_records(), ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                return ToolResult.create_error(
                    f"Index build failed: {exc}. "
                    "Ensure the file is UTF-8 text with one document per line."
                )

            output = {
                "index_path": str(output_path),
                "doc_count": len(lines),
                "term_count": idx.term_count(),
            }
            return ToolResult.create_success(
                output,
                summary=(
                    f"Built index: {len(lines)} docs, "
                    f"{idx.term_count()} terms -> {output_path.name}"
                ),
            )

    # ── 8. kaos-nlp-hash ─────────────────────────────────────────────

    class HashTool(KaosTool):
        """Compute a fuzzy hash of text."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-hash",
                display_name="Hash Text",
                description=(
                    "Compute a fuzzy hash of text. Supports CTPH "
                    "(context-triggered piecewise hashing) and MinHash "
                    "(locality-sensitive hashing). CTPH produces a single "
                    "hash string for similarity comparison. MinHash produces "
                    "a signature array for near-duplicate detection. "
                    "For finding duplicates across many texts, use "
                    "kaos-nlp-find-duplicates."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text",
                        type="string",
                        description="Text to hash.",
                    ),
                    ParameterSchema(
                        name="algorithm",
                        type="string",
                        description="Hash algorithm (default: ctph).",
                        required=False,
                        default="ctph",
                        constraints={"enum": ["ctph", "minhash"]},
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.hashing import MinHasher, ctph_hash_str

            text = inputs.get("text")
            if not text:
                return ToolResult.create_error(
                    "Parameter 'text' is required and must be non-empty. "
                    "Provide the text you want to hash."
                )

            algorithm = inputs.get("algorithm", "ctph")

            try:
                if algorithm == "ctph":
                    hash_value = ctph_hash_str(text, 64, 8, 4)
                    output: dict[str, Any] = {
                        "hash": hash_value,
                        "algorithm": "ctph",
                    }
                    summary = f"CTPH hash: {hash_value[:40]}..."
                else:
                    hasher = MinHasher(128, 42)
                    sig = hasher.hash_char_shingles(text, 3)
                    values = sig.values
                    output = {
                        "signature": values[:32],
                        "algorithm": "minhash",
                        "num_permutations": len(values),
                    }
                    summary = f"MinHash: {len(values)} permutations"
            except Exception as exc:
                return ToolResult.create_error(
                    f"Hashing failed: {exc}. Ensure the input is valid text."
                )

            return ToolResult.create_success(output, summary=summary)

    # ── 9. kaos-nlp-find-duplicates ──────────────────────────────────

    class FindDuplicatesTool(KaosTool):
        """Find near-duplicate texts using MinHash/LSH."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-find-duplicates",
                display_name="Find Duplicates",
                description=(
                    "Find near-duplicate texts in a list using MinHash "
                    "locality-sensitive hashing. Groups similar texts by "
                    "a configurable similarity threshold. Returns groups "
                    "with a canonical text index and duplicate indices with "
                    "similarity scores. For comparing two specific strings, "
                    "use kaos-nlp-compare."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="texts",
                        type="array",
                        description="List of text documents to compare.",
                        constraints={"items": {"type": "string"}},
                    ),
                    ParameterSchema(
                        name="threshold",
                        type="number",
                        description=(
                            "Similarity threshold (0-1, default: 0.5). Higher means more similar."
                        ),
                        required=False,
                        default=0.5,
                        constraints={"minimum": 0.0, "maximum": 1.0},
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.hashing import MinHasher, find_duplicates
            from kaos_nlp_core.tokenizer import tokenize_words

            texts = inputs.get("texts")
            if not texts or not isinstance(texts, list):
                return ToolResult.create_error(
                    "Parameter 'texts' is required and must be a non-empty "
                    "list of strings. Provide at least 2 texts to compare."
                )
            if len(texts) < 2:
                return ToolResult.create_error(
                    "At least 2 texts are required for duplicate detection. "
                    "For comparing exactly 2 strings, use kaos-nlp-compare."
                )

            threshold = inputs.get("threshold", 0.5)

            try:
                hasher = MinHasher(128, 42)
                docs: list[tuple[int, list[str]]] = []
                for i, t in enumerate(texts):
                    terms = tokenize_words(str(t), lowercase=True)
                    docs.append((i, terms))

                groups = find_duplicates(hasher, docs, shingle_size=2, threshold=threshold)
            except Exception as exc:
                return ToolResult.create_error(
                    f"Duplicate detection failed: {exc}. "
                    "Ensure all items in 'texts' are valid strings."
                )

            group_list = [
                {
                    "canonical": g.canonical_id,
                    "duplicates": [{"index": did, "similarity": sim} for did, sim in g.duplicates],
                }
                for g in groups
            ]

            output = {
                "groups": group_list,
                "total_groups": len(group_list),
            }
            return ToolResult.create_success(
                output,
                summary=(f"Found {len(group_list)} duplicate group(s) at threshold={threshold}"),
            )

    # ── 10. kaos-nlp-analyze-text ────────────────────────────────────

    class AnalyzeTextTool(KaosTool):
        """Comprehensive text analysis: stats, vocabulary, structure."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-analyze-text",
                display_name="Analyze Text",
                description=(
                    "Compute comprehensive text statistics: character count, "
                    "word count, sentence count, paragraph count, unique tokens, "
                    "type-token ratio, and top terms by frequency. "
                    "For tokenization only, use kaos-nlp-tokenize. "
                    "For segmentation only, use kaos-nlp-segment-sentences."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text",
                        type="string",
                        description="Text to analyze.",
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.segmentation import (
                segment_paragraphs_simple,
                segment_sentences,
            )
            from kaos_nlp_core.structures import FrequencyVocabulary
            from kaos_nlp_core.tokenizer import tokenize

            text = inputs.get("text")
            if not text:
                return ToolResult.create_error(
                    "Parameter 'text' is required and must be non-empty. "
                    "Provide the text you want to analyze."
                )

            try:
                # Basic counts
                char_count = len(text)
                tokens = tokenize(text, lowercase=True)
                token_count = len(tokens)

                # Segmentation
                sentences = segment_sentences(text)
                paragraphs = segment_paragraphs_simple(text)

                # Vocabulary
                vocab = FrequencyVocabulary()
                for t in tokens:
                    vocab.insert(t.text)
                unique_terms = len(vocab)
                type_token_ratio = unique_terms / token_count if token_count > 0 else 0.0

                # Averages
                avg_sentence_len = token_count / len(sentences) if sentences else 0.0

                # Top terms
                top_terms = vocab.top_n(10)
            except Exception as exc:
                return ToolResult.create_error(
                    f"Analysis failed: {exc}. Ensure the input is valid text."
                )

            output = {
                "characters": char_count,
                "tokens": token_count,
                "unique_terms": unique_terms,
                "sentences": len(sentences),
                "paragraphs": len(paragraphs),
                "avg_sentence_length": round(avg_sentence_len, 2),
                "type_token_ratio": round(type_token_ratio, 4),
                "top_terms": [{"term": t, "count": c} for t, c in top_terms],
            }
            return ToolResult.create_success(
                output,
                summary=(
                    f"Text: {char_count} chars, {token_count} tokens, "
                    f"{len(sentences)} sentences, TTR={type_token_ratio:.4f}"
                ),
            )

    # ── 11. kaos-nlp-score-quality ──────────────────────────────────

    class ScoreQualityTool(KaosTool):
        """Score text quality using 18 metrics with anomaly detection."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-score-quality",
                display_name="Score Text Quality",
                description=(
                    "Compute 18 text quality metrics (whitespace ratio, "
                    "line/paragraph length, alphanumeric ratio, capitalization, "
                    "punctuation, symbol, word length, type-token ratio, "
                    "token/char entropy, repetition rate, max frequency ratio, "
                    "format-token ratio, and in-lexicon ratio) and score "
                    "deviations from expected ranges as a weighted anomaly. "
                    "Lower scores are better — zero means all metrics fall "
                    "within the expected range. The in-lexicon ratio is the "
                    "strongest OCR / extraction-quality signal: scrambled "
                    "text bombs lexicon hit-rate even when other ratios look "
                    "fine. Domain presets: 'general' (default, wider ranges, "
                    "calibrated on Project Gutenberg + EDGAR) or 'legal' "
                    "(USC-calibrated). Set 'use_lexicon' false for non-English "
                    "text. For basic text statistics, use kaos-nlp-analyze-text."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text",
                        type="string",
                        description="Text to score.",
                    ),
                    ParameterSchema(
                        name="domain",
                        type="string",
                        description=(
                            "Expected-range preset: 'general' (default, wider "
                            "ranges) or 'legal' (USC-calibrated, tighter)."
                        ),
                        required=False,
                        default="general",
                        constraints={"enum": ["general", "legal"]},
                    ),
                    ParameterSchema(
                        name="use_lexicon",
                        type="boolean",
                        description=(
                            "Include the in-lexicon-token ratio metric — the "
                            "strongest OCR / extraction-quality signal. Loads "
                            "the bundled English wordset (~2 MB FST, ~382k "
                            "headwords from OpenGloss) on first call. Set "
                            "false for non-English text."
                        ),
                        required=False,
                        default=True,
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.quality import quality_report

            text = inputs.get("text")
            if not text:
                return ToolResult.create_error(
                    "Parameter 'text' is required and must be non-empty. "
                    "Provide the text you want to score for quality."
                )

            domain = inputs.get("domain", "general")
            use_lexicon = inputs.get("use_lexicon", True)

            try:
                report = quality_report(
                    text,
                    domain=domain,
                    use_default_lexicon=bool(use_lexicon),
                )
            except ValueError as exc:
                return ToolResult.create_error(f"Invalid domain: {exc}. Use 'general' or 'legal'.")
            except Exception as exc:
                return ToolResult.create_error(
                    f"Quality scoring failed: {exc}. Ensure the input is valid text."
                )

            metrics = report.metrics.to_dict()
            output = {
                "score": report.score.score,
                "domain": report.score.domain,
                "metrics": {
                    k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()
                },
                "deviations": {k: c.to_dict() for k, c in report.score.components.items()},
            }

            n_dev = len(report.score.components)
            lex_ratio = report.metrics.ratio_in_lexicon
            ocr_hint = ""
            if lex_ratio is not None and lex_ratio < 0.5:
                ocr_hint = f", ratio_in_lexicon={lex_ratio:.2f} — likely OCR/extraction artifact"
            elif lex_ratio is not None:
                ocr_hint = f", ratio_in_lexicon={lex_ratio:.2f}"

            return ToolResult.create_success(
                output,
                summary=(
                    f"Quality score: {report.score.score:.4f} "
                    f"({report.score.domain} domain, {n_dev} deviation(s){ocr_hint})"
                ),
            )

    # ── 12. kaos-nlp-lexicon-related ─────────────────────────────────

    _LEXICON_RELATIONS = [
        "synonym",
        "antonym",
        "hypernym",
        "hyponym",
        "inflection",
        "collocation",
        "derivation_noun",
        "derivation_verb",
        "derivation_adjective",
        "derivation_adverb",
        "cognate",
        "etymology_parent",
    ]

    class LexiconRelatedTool(KaosTool):
        """Look up related terms (synonyms, hypernyms, inflections, ...) for a word."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-lexicon-related",
                display_name="Lexicon — Related Terms",
                description=(
                    "Look up related terms for a single word in the bundled "
                    "OpenGloss v1.3 lexicon (~206k headwords across legal, "
                    "medical, scientific, financial, technical, and general "
                    "domains). Supports synonym, antonym, hypernym, hyponym, "
                    "inflection, collocation, derivation, cognate, and "
                    "etymology_parent relations. Optionally filter by part of "
                    "speech and sense index to avoid cross-sense pollution "
                    "(e.g. 'contract' as legal noun vs. shrink verb). For "
                    "multi-term query expansion suitable for retrieval, use "
                    "kaos-nlp-lexicon-expand-query."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="word",
                        type="string",
                        description="The word to look up.",
                    ),
                    ParameterSchema(
                        name="relation",
                        type="string",
                        description=(
                            "Relation type. 'synonym' for similar meanings, "
                            "'hypernym' for broader categories, 'hyponym' for "
                            "narrower instances, 'inflection' for morphological "
                            "forms (plurals, conjugations)."
                        ),
                        required=False,
                        default="synonym",
                        constraints={"enum": _LEXICON_RELATIONS},
                    ),
                    ParameterSchema(
                        name="pos",
                        type="string",
                        description=(
                            "Part of speech filter (e.g. 'noun', 'verb', "
                            "'adjective'). When set with sense_index, "
                            "restricts the lookup to that specific word sense."
                        ),
                        required=False,
                        default=None,
                    ),
                    ParameterSchema(
                        name="sense_index",
                        type="integer",
                        description=(
                            "Sense index for sense-aware lookup (0-based). "
                            "Use kaos-nlp-lexicon-related with relation='synonym' "
                            "and inspect the results across senses to discover "
                            "available indices, or call get_senses() in code."
                        ),
                        required=False,
                        default=None,
                        constraints={"minimum": 0},
                    ),
                    ParameterSchema(
                        name="max_results",
                        type="integer",
                        description="Maximum number of related terms to return.",
                        required=False,
                        default=50,
                        constraints={"minimum": 1, "maximum": 1000},
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.lexicon import default_opengloss_lexicon

            word = inputs.get("word")
            if not word or not isinstance(word, str):
                return ToolResult.create_error(
                    "Parameter 'word' is required and must be a non-empty string. "
                    "Provide the word you want to look up."
                )

            relation = inputs.get("relation", "synonym")
            pos = inputs.get("pos")
            sense_index = inputs.get("sense_index")
            max_results = int(inputs.get("max_results", 50))

            try:
                lex = default_opengloss_lexicon()
            except FileNotFoundError as exc:
                return ToolResult.create_error(str(exc))

            if not lex.contains(word):
                # Try lowercased fallback before erroring — most lexicon
                # entries are lower-cased headwords.
                lowered = word.lower()
                if lowered != word and lex.contains(lowered):
                    word = lowered
                else:
                    return ToolResult.create_error(
                        f"Word '{word}' not found in the OpenGloss lexicon "
                        f"({len(lex)} entries). Check the spelling, try the "
                        f"lower-cased form, or use kaos-nlp-find-pattern with "
                        f"a regex to discover similar terms."
                    )

            try:
                terms = lex.related_typed(word, relation, pos=pos, sense_index=sense_index)
            except Exception as exc:
                return ToolResult.create_error(
                    f"Lexicon lookup failed: {exc}. Verify relation is one of {_LEXICON_RELATIONS}."
                )

            truncated = terms[:max_results]
            output = {
                "word": word,
                "relation": relation,
                "pos": pos,
                "sense_index": sense_index,
                "related": [
                    {
                        "text": t.text,
                        "relation": t.relation,
                        "pos": t.pos,
                        "sense_index": t.sense_index,
                    }
                    for t in truncated
                ],
                "count": len(truncated),
                "total_matches": len(terms),
                "has_more": len(terms) > max_results,
            }
            return ToolResult.create_success(
                output,
                summary=(
                    f"{len(truncated)} {relation}(s) for '{word}'"
                    + (f" (truncated from {len(terms)})" if len(terms) > max_results else "")
                ),
            )

    # ── 13. kaos-nlp-lexicon-expand-query ────────────────────────────

    class LexiconExpandQueryTool(KaosTool):
        """Expand a multi-term query with synonyms / inflections / hypernyms."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-lexicon-expand-query",
                display_name="Lexicon — Expand Query",
                description=(
                    "Expand a list of query terms with related terms drawn "
                    "from the bundled OpenGloss v1.3 lexicon. Useful before "
                    "retrieval over an external corpus (BM25 / TF-IDF / "
                    "vector search) to improve recall on terminology-heavy "
                    "queries. Default expansion uses synonyms + inflections, "
                    "which is conservative; add 'hypernym' for broader "
                    "category-level recall. max_depth controls BFS hops "
                    "(depth=2 expands synonyms-of-synonyms — use sparingly, "
                    "results explode). For single-word lookups with sense "
                    "filtering, use kaos-nlp-lexicon-related instead. For "
                    "BM25 search where expansion happens internally, use "
                    "kaos-nlp-search-text."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="terms",
                        type="array",
                        description="Query terms to expand.",
                        constraints={"items": {"type": "string"}},
                    ),
                    ParameterSchema(
                        name="relations",
                        type="array",
                        description=(
                            "Relation types to walk. Default ['synonym', "
                            "'inflection']. Add 'hypernym' for broader recall "
                            "or 'hyponym' for narrower."
                        ),
                        required=False,
                        default=None,
                        constraints={"items": {"type": "string", "enum": _LEXICON_RELATIONS}},
                    ),
                    ParameterSchema(
                        name="max_depth",
                        type="integer",
                        description=(
                            "BFS expansion depth. 1 (default) walks one hop; "
                            "2 includes synonyms-of-synonyms (recall jumps, "
                            "precision drops); 3 is rarely useful."
                        ),
                        required=False,
                        default=1,
                        constraints={"minimum": 1, "maximum": 3},
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.lexicon import default_opengloss_lexicon

            raw_terms = inputs.get("terms")
            if not raw_terms or not isinstance(raw_terms, list):
                return ToolResult.create_error(
                    "Parameter 'terms' is required and must be a non-empty list of query strings."
                )
            terms = [str(t).strip() for t in raw_terms if str(t).strip()]
            if not terms:
                return ToolResult.create_error(
                    "All entries in 'terms' were empty after stripping. "
                    "Provide at least one non-empty query term."
                )

            relations = inputs.get("relations") or ["synonym", "inflection"]
            if not isinstance(relations, list) or not relations:
                return ToolResult.create_error(
                    "Parameter 'relations' must be a non-empty list of relation "
                    f"types. Allowed values: {_LEXICON_RELATIONS}."
                )
            unknown = [r for r in relations if r not in _LEXICON_RELATIONS]
            if unknown:
                return ToolResult.create_error(
                    f"Unknown relation type(s): {unknown}. Allowed values: {_LEXICON_RELATIONS}."
                )

            max_depth = int(inputs.get("max_depth", 1))
            if max_depth < 1 or max_depth > 3:
                return ToolResult.create_error(
                    "Parameter 'max_depth' must be between 1 and 3. "
                    "Use 1 (default) for direct relations, 2 for second-hop "
                    "expansion (results explode quickly)."
                )

            try:
                lex = default_opengloss_lexicon()
            except FileNotFoundError as exc:
                return ToolResult.create_error(str(exc))

            try:
                expanded = lex.expand_query(terms, relations, max_depth=max_depth)
            except Exception as exc:
                return ToolResult.create_error(
                    f"Query expansion failed: {exc}. Verify all relation types "
                    f"are valid: {_LEXICON_RELATIONS}."
                )

            output = {
                "original_terms": terms,
                "expanded_terms": list(expanded),
                "original_count": len(terms),
                "expanded_count": len(expanded),
                "expansion_factor": (round(len(expanded) / len(terms), 3) if terms else 0.0),
                "relations": relations,
                "max_depth": max_depth,
            }
            return ToolResult.create_success(
                output,
                summary=(
                    f"Expanded {len(terms)} term(s) -> {len(expanded)} "
                    f"(x{output['expansion_factor']}) via {','.join(relations)} "
                    f"depth={max_depth}"
                ),
            )

    # ── 14. kaos-nlp-token-frequency ─────────────────────────────────

    class TokenFrequencyTool(KaosTool):
        """Term-frequency table for a document, optionally filtered by lexicon."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-token-frequency",
                display_name="Token Frequency",
                description=(
                    "Compute a per-document term frequency table with optional "
                    "lexicon filtering. Without 'lexicon', counts every token "
                    "(useful for raw vocab analysis). With lexicon='english' "
                    "(the bundled ~382k OpenGloss-derived wordset), only "
                    "in-vocabulary terms are counted — the result is a clean "
                    "frequency table over real English words and the "
                    "'coverage' field becomes the in-vocab-token ratio "
                    "(strong OCR / extraction-quality signal). Returns terms "
                    "ranked by frequency descending. For broader text "
                    "statistics use kaos-nlp-analyze-text. For knowledge-graph "
                    "concept extraction (hypernyms / hyponyms) use "
                    "kaos-nlp-extract-concepts."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text",
                        type="string",
                        description="Document text to count.",
                    ),
                    ParameterSchema(
                        name="lexicon",
                        type="string",
                        description=(
                            "Filter mode: 'none' counts every token (default); "
                            "'english' filters to the bundled ~382k-key English "
                            "wordset (OpenGloss-derived, ships in the wheel)."
                        ),
                        required=False,
                        default="none",
                        constraints={"enum": ["none", "english"]},
                    ),
                    ParameterSchema(
                        name="top_k",
                        type="integer",
                        description="Maximum number of terms to return.",
                        required=False,
                        default=50,
                        constraints={"minimum": 1, "maximum": 5000},
                    ),
                    ParameterSchema(
                        name="min_count",
                        type="integer",
                        description="Drop terms with fewer than this many occurrences.",
                        required=False,
                        default=1,
                        constraints={"minimum": 1},
                    ),
                    ParameterSchema(
                        name="lowercase",
                        type="boolean",
                        description="Lowercase tokens before counting (default true).",
                        required=False,
                        default=True,
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.quality import default_english_wordset
            from kaos_nlp_core.vocabulary import token_frequency

            text = inputs.get("text")
            if not text:
                return ToolResult.create_error(
                    "Parameter 'text' is required and must be non-empty. "
                    "Provide the document text you want to count."
                )

            lex_mode = inputs.get("lexicon", "none")
            top_k = int(inputs.get("top_k", 50))
            min_count = int(inputs.get("min_count", 1))
            lowercase = bool(inputs.get("lowercase", True))

            lex_obj: Any = None
            if lex_mode == "english":
                try:
                    lex_obj = default_english_wordset()
                except FileNotFoundError as exc:
                    return ToolResult.create_error(
                        f"English wordset unavailable: {exc}. Set lexicon='none' "
                        "to count every token, or install the bundled wordset."
                    )
            elif lex_mode != "none":
                return ToolResult.create_error(
                    f"Unknown lexicon mode '{lex_mode}'. Use 'none' or 'english'."
                )

            try:
                result = token_frequency(
                    text,
                    lexicon=lex_obj,
                    lowercase=lowercase,
                    min_count=min_count,
                    top_k=top_k,
                )
            except Exception as exc:
                return ToolResult.create_error(
                    f"Token frequency failed: {exc}. Verify the input is valid text."
                )

            kept = result.kept_tokens
            output = {
                "terms": [
                    {
                        "text": tc.text,
                        "count": tc.count,
                        "share": round(tc.count / kept, 6) if kept else 0.0,
                    }
                    for tc in result.counts
                ],
                "total_tokens": result.total_tokens,
                "kept_tokens": result.kept_tokens,
                "unique_terms": result.unique_terms,
                "coverage": round(result.coverage, 6),
                "lexicon_mode": lex_mode,
            }
            cov_hint = f", coverage={result.coverage:.2f}" if lex_mode != "none" else ""
            return ToolResult.create_success(
                output,
                summary=(
                    f"{result.unique_terms} unique term(s) over "
                    f"{result.kept_tokens}/{result.total_tokens} tokens"
                    f"{cov_hint}"
                ),
            )

    # ── 15. kaos-nlp-extract-concepts ────────────────────────────────

    class ExtractConceptsTool(KaosTool):
        """Surface document concepts via the OpenGloss hypergraph."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-extract-concepts",
                display_name="Extract Concepts (OpenGloss Graph)",
                description=(
                    "Surface concepts in a document by aggregating term "
                    "frequencies up the hypernym graph (default) or down the "
                    "hyponym graph of the bundled OpenGloss lexicon. "
                    "'hypernym' direction asks 'what is this document ABOUT' "
                    "(broader categories — e.g. terms like 'plaintiff, "
                    "summons, complaint, judgment' aggregate to 'legal "
                    "action', 'litigant', 'legal document'). 'hyponym' "
                    "direction asks 'what specific things does this mention' "
                    "(narrower instances). 'both' returns both lists. "
                    "Each concept carries the source terms that contributed "
                    "to its score so the result is auditable. "
                    "WARNING: tagging tool, NOT a retrieval-expansion tool. "
                    "Feeding extracted concepts back into a BM25 query "
                    "reproduces a benchmarked anti-pattern (-18% to -22% "
                    "NDCG@10 on BEIR; see adaptive-retrieval-roadmap.md). "
                    "Hyponym output has known noise (~34% irrelevant on the "
                    "EDGAR review fixture) — treat as exploratory."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text",
                        type="string",
                        description="Document text to extract concepts from.",
                    ),
                    ParameterSchema(
                        name="direction",
                        type="string",
                        description=(
                            "'hypernym' (default, broader concepts), 'hyponym' "
                            "(narrower instances), or 'both' (concatenated)."
                        ),
                        required=False,
                        default="hypernym",
                        constraints={"enum": ["hypernym", "hyponym", "both"]},
                    ),
                    ParameterSchema(
                        name="top_k",
                        type="integer",
                        description=(
                            "Maximum concepts per direction. With direction='both' "
                            "the response can contain up to 2*top_k records."
                        ),
                        required=False,
                        default=20,
                        constraints={"minimum": 1, "maximum": 200},
                    ),
                    ParameterSchema(
                        name="max_depth",
                        type="integer",
                        description=(
                            "BFS hops along the chosen relation. 1 (default) is "
                            "direct relations only. 2-3 walks deeper but pulls "
                            "in noise; same anti-pattern as multi-hop query "
                            "expansion in retrieval."
                        ),
                        required=False,
                        default=1,
                        constraints={"minimum": 1, "maximum": 3},
                    ),
                    ParameterSchema(
                        name="weight",
                        type="string",
                        description=(
                            "Score aggregation. 'log' (default) uses "
                            "log1p(frequency) so one high-frequency term "
                            "doesn't dominate; 'linear' uses raw counts."
                        ),
                        required=False,
                        default="log",
                        constraints={"enum": ["log", "linear"]},
                    ),
                    ParameterSchema(
                        name="min_term_count",
                        type="integer",
                        description=(
                            "Skip source terms with fewer than this many "
                            "occurrences. Useful on very long docs to "
                            "suppress spurious singletons."
                        ),
                        required=False,
                        default=1,
                        constraints={"minimum": 1},
                    ),
                    ParameterSchema(
                        name="extra_stop_terms",
                        type="array",
                        description=(
                            "Additional concept terms to filter (augments the "
                            "calibrated default stop-list). Use to suppress "
                            "domain-specific noise."
                        ),
                        required=False,
                        default=None,
                        constraints={"items": {"type": "string"}},
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.concepts import extract_concepts

            text = inputs.get("text")
            if not text:
                return ToolResult.create_error(
                    "Parameter 'text' is required and must be non-empty. "
                    "Provide the document text you want to analyze."
                )

            direction = inputs.get("direction", "hypernym")
            top_k = int(inputs.get("top_k", 20))
            max_depth = int(inputs.get("max_depth", 1))
            weight = inputs.get("weight", "log")
            min_term_count = int(inputs.get("min_term_count", 1))

            extra_raw = inputs.get("extra_stop_terms")
            extra_stop_terms: list[str] | None
            if extra_raw is None:
                extra_stop_terms = None
            elif isinstance(extra_raw, list):
                extra_stop_terms = [str(t) for t in extra_raw]
            else:
                return ToolResult.create_error(
                    "Parameter 'extra_stop_terms' must be a list of strings."
                )

            try:
                concepts = extract_concepts(
                    text,
                    direction=direction,
                    top_k=top_k,
                    max_depth=max_depth,
                    weight=weight,
                    min_term_count=min_term_count,
                    extra_stop_terms=extra_stop_terms,
                )
            except FileNotFoundError as exc:
                return ToolResult.create_error(str(exc))
            except ValueError as exc:
                return ToolResult.create_error(
                    f"Invalid argument: {exc}. Check direction/weight/max_depth values."
                )
            except Exception as exc:
                return ToolResult.create_error(
                    f"Concept extraction failed: {exc}. Verify input and lexicon availability."
                )

            output = {
                "concepts": [
                    {
                        "term": c.term,
                        "direction": c.direction,
                        "score": c.score,
                        "frequency": c.frequency,
                        "source_terms": list(c.source_terms),
                    }
                    for c in concepts
                ],
                "count": len(concepts),
                "direction": direction,
                "max_depth": max_depth,
            }
            return ToolResult.create_success(
                output,
                summary=(f"{len(concepts)} {direction} concept(s) at depth={max_depth}"),
            )

    # ── 16. kaos-nlp-label-lines ─────────────────────────────────────

    class LabelLinesTool(KaosTool):
        """Label every line of a document with one of seven structural labels."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-label-lines",
                display_name="Label Document Lines",
                description=(
                    "Run the document-structure pipeline (P7): per-line "
                    "feature extraction, Viterbi sequence decoding, and "
                    "heading-stack inference. Returns one of seven labels "
                    "per line: blank, heading, body, list_item, table_row, "
                    "metadata, boilerplate. For each heading line, also "
                    "returns score, hierarchy depth, and enumerator kind. "
                    "Lexicons (heading/hierarchy/enumerator) and weights "
                    "are configurable. For tokenization, use kaos-nlp-tokenize."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text",
                        type="string",
                        description="Document text to label.",
                    ),
                    ParameterSchema(
                        name="enum_lexicon",
                        type="string",
                        description=(
                            "Word-prefix lexicon for enumerator parsing. "
                            "Options: english_legal_us (default), french_legal, "
                            "german_legal, spanish_legal, italian_legal, "
                            "portuguese_legal, markdown_atx."
                        ),
                        required=False,
                        default=None,
                    ),
                    ParameterSchema(
                        name="heading_lexicon",
                        type="string",
                        description=(
                            "Canonical-heading lexicon. Options: english_legal_us "
                            "(default), english_academic, english_software, "
                            "french_legal, german_legal, spanish_legal, "
                            "italian_legal, portuguese_legal, none."
                        ),
                        required=False,
                        default=None,
                    ),
                    ParameterSchema(
                        name="hierarchy_lexicon",
                        type="string",
                        description=(
                            "Hierarchy-keyword lexicon. Options: english_legal_us "
                            "(default), french_legal, german_legal, spanish_legal, "
                            "italian_legal, portuguese_legal, markdown_atx, none."
                        ),
                        required=False,
                        default=None,
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.structure import label_lines

            text = inputs.get("text")
            if not text:
                return ToolResult.create_error(
                    "Parameter 'text' is required and must be non-empty. "
                    "Provide the document text you want to label."
                )
            enum_lex = inputs.get("enum_lexicon")
            heading_lex = inputs.get("heading_lexicon")
            hier_lex = inputs.get("hierarchy_lexicon")
            scoring: dict[str, Any] = {}
            if heading_lex is not None:
                scoring["heading_lexicon"] = heading_lex
            if hier_lex is not None:
                scoring["hierarchy_lexicon"] = hier_lex
            try:
                result = label_lines(
                    text,
                    enum_lexicon=enum_lex,
                    scoring=scoring or None,
                )
            except ValueError as exc:
                return ToolResult.create_error(
                    f"Invalid lexicon: {exc}. Check that enum_lexicon, "
                    "heading_lexicon, and hierarchy_lexicon are one of the "
                    "supported names listed in the tool description."
                )
            except Exception as exc:
                return ToolResult.create_error(
                    f"Line labeling failed: {exc}. Ensure the input is valid UTF-8 text."
                )
            output = {
                "labels": result.labels,
                "candidates": [
                    {
                        "line_index": int(c.line_index),
                        "score": float(c.score),
                        "hierarchy_level": int(c.hierarchy_level),
                        "numeric_depth": int(c.numeric_depth),
                        "atx_depth": int(c.atx_depth),
                        "enumerator_kind": c.enumerator_kind,
                        "picked_depth": int(c.picked_depth()),
                    }
                    for c in result.candidates
                ],
                "label_counts": {
                    label: result.labels.count(label)
                    for label in (
                        "blank",
                        "heading",
                        "body",
                        "list_item",
                        "table_row",
                        "metadata",
                        "boilerplate",
                    )
                },
            }
            return ToolResult.create_success(
                output,
                summary=(
                    f"Labeled {len(result.labels)} line(s); "
                    f"{len(result.candidates)} heading candidate(s)"
                ),
            )

    # ── 13. kaos-nlp-outline ─────────────────────────────────────────

    class OutlineTool(KaosTool):
        """Build a document outline tree + structure summary in one shot."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-outline",
                display_name="Document Outline",
                description=(
                    "Build a heading-hierarchy tree (outline) from a document "
                    "and return a structure summary the caller can use for "
                    "triage. Each outline node includes the heading text, "
                    "its detected depth, the line indices it owns "
                    "(section_start/section_end), and any nested subheadings. "
                    "Summary fields tell you whether the document looks like "
                    "a form, contract, regulation, or prose. Use this when "
                    "you need a high-level view of a document; use "
                    "kaos-nlp-label-lines for line-by-line classification."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_NLP_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="text",
                        type="string",
                        description="Document text.",
                    ),
                    ParameterSchema(
                        name="enum_lexicon",
                        type="string",
                        description=(
                            "P3 enumerator lexicon: english_legal_us (default), "
                            "french_legal, german_legal, spanish_legal, "
                            "italian_legal, portuguese_legal, markdown_atx."
                        ),
                        required=False,
                        default=None,
                    ),
                    ParameterSchema(
                        name="hierarchy_lexicon",
                        type="string",
                        description=("Hierarchy keyword lexicon (same options as enum_lexicon)."),
                        required=False,
                        default=None,
                    ),
                    ParameterSchema(
                        name="heading_lexicon",
                        type="string",
                        description=(
                            "Canonical-heading lexicon: english_legal_us "
                            "(default), english_academic, english_software, "
                            "french_legal, german_legal, spanish_legal, "
                            "italian_legal, portuguese_legal, none."
                        ),
                        required=False,
                        default=None,
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.structure import build_outline, summarize_structure

            text = inputs.get("text")
            if not text:
                return ToolResult.create_error(
                    "Parameter 'text' is required and must be non-empty."
                )
            enum_lex = inputs.get("enum_lexicon")
            hier_lex = inputs.get("hierarchy_lexicon")
            heading_lex = inputs.get("heading_lexicon")
            scoring: dict[str, Any] = {}
            if hier_lex is not None:
                scoring["hierarchy_lexicon"] = hier_lex
            if heading_lex is not None:
                scoring["heading_lexicon"] = heading_lex
            try:
                outline = build_outline(text, enum_lexicon=enum_lex, scoring=scoring or None)
                summary = summarize_structure(text, enum_lexicon=enum_lex, scoring=scoring or None)
            except ValueError as exc:
                return ToolResult.create_error(f"Invalid lexicon: {exc}. Check parameter names.")
            except Exception as exc:
                return ToolResult.create_error(
                    f"Outline build failed: {exc}. Ensure input is valid UTF-8 text."
                )

            def _node_to_dict(n: Any) -> dict[str, Any]:
                return {
                    "line_index": n.line_index,
                    "text": n.text,
                    "depth": n.depth,
                    "score": round(n.score, 3),
                    "section_start": n.section_start,
                    "section_end": n.section_end,
                    "enumerator_kind": n.enumerator_kind,
                    "children": [_node_to_dict(c) for c in n.children],
                }

            output = {
                "outline": [_node_to_dict(n) for n in outline],
                "summary": {
                    "n_lines": summary.n_lines,
                    "label_counts": summary.label_counts,
                    "n_headings": summary.n_headings,
                    "max_depth": summary.max_depth,
                    "has_metadata_block": summary.has_metadata_block,
                    "has_boilerplate": summary.has_boilerplate,
                    "has_table_rows": summary.has_table_rows,
                    "looks_like_form": summary.looks_like_form,
                    "dominant_label": summary.dominant_label,
                    "body_ratio": round(summary.body_ratio, 3),
                },
            }
            return ToolResult.create_success(
                output,
                summary=(
                    f"Outline: {len(outline)} top-level heading(s); "
                    f"{summary.n_headings} total; "
                    f"shape={summary.dominant_label}"
                ),
            )

    # ── Registration ─────────────────────────────────────────────────

    tool_classes = [
        TokenizeTool,
        SegmentSentencesTool,
        SegmentParagraphsTool,
        CompareTool,
        FindPatternTool,
        SearchTextTool,
        BuildIndexTool,
        HashTool,
        FindDuplicatesTool,
        AnalyzeTextTool,
        ScoreQualityTool,
        LexiconRelatedTool,
        LexiconExpandQueryTool,
        TokenFrequencyTool,
        ExtractConceptsTool,
        LabelLinesTool,
        OutlineTool,
    ]

    count = 0
    for cls in tool_classes:
        runtime.tools.register_tool(cls())
        count += 1
    return count
