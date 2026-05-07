//! PyO3 bindings for the Lexicon semantic knowledge graph.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::bindings::util::{
    bincode_getstate, bincode_setstate, decode_from_slice, encode_to_vec, read_knc_header,
    zstd_decompress_capped, zstd_level, KNC_HEADER_LEN, KNC_MAGIC, KNC_VERSION_INTERN,
    KNC_VERSION_RAW, KNC_VERSION_ZSTD,
};
use crate::core::lexicon::{Edge, LexemeEntry, Lexicon, LexiconCompact, RelationType, Sense};

/// Parse a Python dict into a LexemeEntry.
fn dict_to_entry(d: &Bound<'_, PyDict>) -> PyResult<LexemeEntry> {
    let word: String = d
        .get_item("word")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'word'"))?
        .extract()?;

    // Parse senses
    let senses: Vec<Sense> = if let Some(senses_obj) = d.get_item("senses")? {
        let senses_list: &Bound<'_, PyList> = senses_obj.cast()?;
        senses_list
            .iter()
            .map(|s| {
                let sd: &Bound<'_, PyDict> = s.cast()?;
                Ok(Sense {
                    part_of_speech: sd
                        .get_item("part_of_speech")?
                        .map(|v| v.extract())
                        .transpose()?
                        .unwrap_or_default(),
                    sense_index: sd
                        .get_item("sense_index")?
                        .map(|v| v.extract())
                        .transpose()?
                        .unwrap_or(0),
                    definition: sd
                        .get_item("definition")?
                        .map(|v| v.extract())
                        .transpose()?
                        .unwrap_or_default(),
                    synonyms: extract_string_list(sd, "synonyms")?,
                    antonyms: extract_string_list(sd, "antonyms")?,
                    hypernyms: extract_string_list(sd, "hypernyms")?,
                    hyponyms: extract_string_list(sd, "hyponyms")?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?
    } else {
        vec![]
    };

    // Parse edges
    let edges: Vec<Edge> = if let Some(edges_obj) = d.get_item("edges")? {
        let edges_list: &Bound<'_, PyList> = edges_obj.cast()?;
        edges_list
            .iter()
            .map(|e| {
                let ed: &Bound<'_, PyDict> = e.cast()?;
                let rel_str: String = ed
                    .get_item("relationship_type")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();
                Ok(Edge {
                    relationship_type: RelationType::parse(&rel_str),
                    target: ed
                        .get_item("target")?
                        .map(|v| v.extract())
                        .transpose()?
                        .unwrap_or_default(),
                    source_pos: ed.get_item("source_pos")?.and_then(|v| v.extract().ok()),
                    sense_index: ed.get_item("sense_index")?.and_then(|v| v.extract().ok()),
                })
            })
            .collect::<PyResult<Vec<_>>>()?
    } else {
        vec![]
    };

    Ok(LexemeEntry {
        word,
        senses,
        edges,
        all_synonyms: extract_string_list(d, "all_synonyms")?,
        all_antonyms: extract_string_list(d, "all_antonyms")?,
        all_hypernyms: extract_string_list(d, "all_hypernyms")?,
        all_hyponyms: extract_string_list(d, "all_hyponyms")?,
        all_inflections: extract_string_list(d, "all_inflections")?,
        all_derivations: extract_string_list(d, "all_derivations")?,
        all_collocations: extract_string_list(d, "all_collocations")?,
    })
}

fn extract_string_list(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<String>> {
    Ok(d.get_item(key)?
        .map(|v| v.extract::<Vec<String>>())
        .transpose()?
        .unwrap_or_default())
}

fn parse_relation(s: &str) -> PyResult<RelationType> {
    let r = RelationType::parse(s);
    Ok(r)
}

fn parse_relations(relations: Vec<String>) -> PyResult<Vec<RelationType>> {
    relations.iter().map(|s| parse_relation(s)).collect()
}

// ─── KNC v3 (Lexicon-specific) save/load ────────────────────────────────────

/// Save a Lexicon to disk using KNC v3 (string-interned + dropped definitions
/// + zstd-compressed postcard).
///
/// Use this for any Lexicon save — it is strictly smaller than KNC v2 for the
/// same data.
///
/// Layout: `KNC1` magic, u16 LE version=3, zstd(postcard(LexiconCompact)).
fn save_lexicon_to_path(lex: &Lexicon, path: &str) -> Result<(), String> {
    let compact = lex.to_compact();
    let raw = encode_to_vec(&compact).map_err(|e| e.to_string())?;
    let compressed =
        zstd::encode_all(raw.as_slice(), zstd_level()).map_err(|e| format!("zstd encode: {e}"))?;
    let mut buf = Vec::with_capacity(KNC_HEADER_LEN + compressed.len());
    buf.extend_from_slice(KNC_MAGIC);
    buf.extend_from_slice(&KNC_VERSION_INTERN.to_le_bytes());
    buf.extend_from_slice(&compressed);
    std::fs::write(path, buf).map_err(|e| e.to_string())
}

/// Load a Lexicon from disk. Accepts:
/// - KNC v1 (raw postcard of the runtime `Lexicon`)
/// - KNC v2 (zstd(postcard(runtime `Lexicon`)))
/// - KNC v3 (zstd(postcard(`LexiconCompact`)) — string-interned, definitions
///   dropped; rehydrates `Sense::definition` as `""`)
fn load_lexicon_from_path(path: &str) -> Result<Lexicon, String> {
    let (version, payload) = read_knc_header(path)?;
    decode_lexicon_payload(version, &payload, path)
}

/// Decode an in-memory KNC payload into a runtime Lexicon. Shared between the
/// file-on-disk path (`load_lexicon_from_path`) and the wheel-embedded path
/// (`Lexicon::default_embedded`). `source_label` is the user-facing string
/// used in error messages — `path` for files, `"<embedded OpenGloss>"` for
/// the bundled lexicon.
fn decode_lexicon_payload(
    version: u16,
    payload: &[u8],
    source_label: &str,
) -> Result<Lexicon, String> {
    match version {
        KNC_VERSION_RAW => decode_from_slice::<Lexicon>(payload).map_err(|e| e.to_string()),
        KNC_VERSION_ZSTD => {
            let decompressed = zstd_decompress_capped(payload, source_label)?;
            decode_from_slice::<Lexicon>(&decompressed).map_err(|e| e.to_string())
        }
        KNC_VERSION_INTERN => {
            let decompressed = zstd_decompress_capped(payload, source_label)?;
            let compact: LexiconCompact =
                decode_from_slice(&decompressed).map_err(|e| e.to_string())?;
            Lexicon::from_compact(compact)
        }
        _ => Err(format!(
            "{source_label}: unsupported KNC format version {version} for Lexicon"
        )),
    }
}

// ─── Wheel-embedded OpenGloss lexicon ──────────────────────────────────────
//
// The canonical KNC v3 OpenGloss binary is baked into the compiled `_rust`
// shared object via `include_bytes!`. End-users get a working
// `default_opengloss_lexicon()` after `pip install kaos-nlp-core` with no
// filesystem state, no network call, no env var. The override path (env var
// `KAOS_NLP_LEXICON_PATH` / explicit `Lexicon.load(...)`) still works via the
// file-loading code above.
//
// Source-of-truth: regenerate via
//   `KAOS_NLP_ZSTD_LEVEL=19 uv run python scripts/build_opengloss_lexicon.py
//    --output python/kaos_nlp_core/data/opengloss-v1.3.lexicon.bin --compression-level 19`
// after which `cargo build --release` re-bakes the bytes into _rust.so.

const EMBEDDED_OPENGLOSS_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/python/kaos_nlp_core/data/opengloss-v1.3.lexicon.bin"
);

/// Compile-time embedded OpenGloss v1.3 KNC v3 binary. ~32 MB at zstd level 19.
const EMBEDDED_OPENGLOSS_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/python/kaos_nlp_core/data/opengloss-v1.3.lexicon.bin"
));

/// Decode and return the lexicon embedded in this `_rust` shared object.
fn load_embedded_opengloss() -> Result<Lexicon, String> {
    if EMBEDDED_OPENGLOSS_BYTES.len() < KNC_HEADER_LEN
        || &EMBEDDED_OPENGLOSS_BYTES[0..4] != KNC_MAGIC
    {
        return Err(format!(
            "embedded OpenGloss bytes are corrupt or empty (len={}, expected file at {})",
            EMBEDDED_OPENGLOSS_BYTES.len(),
            EMBEDDED_OPENGLOSS_PATH
        ));
    }
    let version = u16::from_le_bytes([EMBEDDED_OPENGLOSS_BYTES[4], EMBEDDED_OPENGLOSS_BYTES[5]]);
    decode_lexicon_payload(
        version,
        &EMBEDDED_OPENGLOSS_BYTES[KNC_HEADER_LEN..],
        "<embedded OpenGloss>",
    )
}

// ─── PyLexicon ──────────────────────────────────────────────────────────────

/// Semantic knowledge graph for query expansion.
///
/// Generic — works with OpenGloss or any dataset providing words, senses,
/// and typed semantic edges. Supports dynamic loading, subset loading,
/// and pickle for multiprocessing.
///
/// Example:
///     lex = Lexicon()
///     lex.add_entry({"word": "contract", "all_synonyms": ["agreement"], ...})
///     lex.synonyms("contract")  # → ["agreement"]
///     expanded = lex.expand_query(["contract"], ["synonym", "inflection"])
#[pyclass(name = "Lexicon", module = "kaos_nlp_core._rust.lexicon")]
pub struct PyLexicon {
    pub(crate) inner: Lexicon,
}

impl PyLexicon {
    /// Borrow the underlying Lexicon. Used by sibling binding modules.
    pub fn inner_ref(&self) -> &Lexicon {
        &self.inner
    }
}

#[pymethods]
impl PyLexicon {
    #[new]
    fn new() -> Self {
        Self {
            inner: Lexicon::new(),
        }
    }

    /// Load a Lexicon from disk.
    ///
    /// **Trusted-source only.** The on-disk format uses postcard behind a
    /// `KNC1` magic + u16 version header with a configurable size cap
    /// (`KAOS_NLP_MAX_LOAD_BYTES`, default 256 MiB). The header guards
    /// against truncated / mistyped files and unbounded allocations,
    /// but does not protect against an adversarial author of the file.
    ///
    /// Supports three versions:
    ///
    /// - **v1** — raw postcard of the runtime Lexicon. Legacy.
    /// - **v2** — zstd-compressed postcard of the runtime Lexicon.
    /// - **v3** — zstd-compressed postcard of a string-interned compact
    ///   form. KNC v3 *intentionally drops* `Sense.definition` to save
    ///   50–150 MB on the OpenGloss lexicon; runtime APIs
    ///   (`synonyms`/`hypernyms`/`hyponyms`/`antonyms`/`related`/
    ///   `expand_query`) never read it. After loading a v3 file,
    ///   `get_senses()` returns `definition=""` for every sense.
    ///
    /// `Lexicon.save()` always writes v3.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_owned = path.to_string();
        let inner: Lexicon = py
            .detach(|| load_lexicon_from_path(&path_owned))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Load the OpenGloss v1.3 lexicon embedded in this `_rust` shared
    /// object. ~32 MB KNC v3 binary baked in at build time via
    /// `include_bytes!`. Returns the same fully-rehydrated `Lexicon` as
    /// `Lexicon.load(<path>)` would for the same source bytes — except
    /// `Sense.definition` is `""` (KNC v3 drops definitions; see
    /// `Lexicon.load` for the rationale).
    ///
    /// This is the path `default_opengloss_lexicon()` calls when no env-var
    /// override is set, giving a working lexicon out of the box after
    /// `pip install kaos-nlp-core` with no filesystem state required.
    #[staticmethod]
    fn default_embedded(py: Python<'_>) -> PyResult<Self> {
        let inner: Lexicon = py
            .detach(load_embedded_opengloss)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bincode_getstate(py, &self.inner)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode_setstate(state)?;
        Ok(())
    }

    /// Save this Lexicon to disk in KNC v3 format (string-interned + zstd).
    ///
    /// **KNC v3 drops `Sense.definition`** by design — that field is never
    /// read by the runtime relation/expansion APIs. If you reload a saved
    /// lexicon, `get_senses()[i]["definition"]` will be `""`. See `load()`
    /// for the full version compatibility matrix.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let path_owned = path.to_string();
        let inner = &self.inner;
        py.detach(|| save_lexicon_to_path(inner, &path_owned))
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Add an entry from a dict. Accepts the OpenGloss schema.
    ///
    /// Required key: "word" (str).
    /// Optional keys: "senses" (list[dict]), "edges" (list[dict]),
    ///   "all_synonyms", "all_antonyms", "all_hypernyms", "all_hyponyms",
    ///   "all_inflections", "all_derivations", "all_collocations" (list[str]).
    fn add_entry(&mut self, entry: &Bound<'_, PyDict>) -> PyResult<()> {
        let e = dict_to_entry(entry)?;
        self.inner.add_entry(e);
        Ok(())
    }

    /// Add multiple entries from a list of dicts.
    fn add_entries(&mut self, entries: &Bound<'_, PyList>) -> PyResult<()> {
        let py = entries.py();
        // Phase 1: extract all Python dicts to Rust structs (requires GIL)
        let parsed: Vec<LexemeEntry> = entries
            .iter()
            .map(|item| {
                let d: &Bound<'_, PyDict> = item.cast()?;
                dict_to_entry(d)
            })
            .collect::<PyResult<Vec<_>>>()?;
        // Phase 2: add all entries without GIL
        let inner = &mut self.inner;
        py.detach(|| {
            for e in parsed {
                inner.add_entry(e);
            }
        });
        Ok(())
    }

    /// Number of entries in the lexicon.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if a word exists.
    fn __contains__(&self, word: &str) -> bool {
        self.inner.contains(word)
    }

    fn contains(&self, word: &str) -> bool {
        self.inner.contains(word)
    }

    /// Get synonyms (all senses).
    fn synonyms(&self, word: &str) -> Vec<String> {
        self.inner.synonyms(word)
    }

    /// Get antonyms (all senses).
    fn antonyms(&self, word: &str) -> Vec<String> {
        self.inner.antonyms(word)
    }

    /// Get hypernyms (all senses).
    fn hypernyms(&self, word: &str) -> Vec<String> {
        self.inner.hypernyms(word)
    }

    /// Get hyponyms (all senses).
    fn hyponyms(&self, word: &str) -> Vec<String> {
        self.inner.hyponyms(word)
    }

    /// Get inflections.
    fn inflections(&self, word: &str) -> Vec<String> {
        self.inner.inflections(word)
    }

    /// Get collocations.
    fn collocations(&self, word: &str) -> Vec<String> {
        self.inner.collocations(word)
    }

    /// Get related terms by relationship type.
    ///
    /// Args:
    ///     word: The word to look up.
    ///     relation: Relationship type string (e.g., "synonym", "hypernym").
    ///     pos: Part of speech filter (optional, e.g., "noun").
    ///     sense_index: Sense index filter (optional, e.g., 0).
    #[pyo3(signature = (word, relation, pos=None, sense_index=None))]
    fn related(
        &self,
        word: &str,
        relation: &str,
        pos: Option<&str>,
        sense_index: Option<u32>,
    ) -> PyResult<Vec<String>> {
        Ok(self
            .inner
            .related(word, parse_relation(relation)?, pos, sense_index))
    }

    /// Expand query terms using specified relation types.
    ///
    /// Args:
    ///     terms: List of query terms.
    ///     relations: List of relation type strings (e.g., ["synonym", "inflection"]).
    ///     max_depth: Maximum expansion hops (default 1).
    ///
    /// Returns a list of expanded terms (including originals).
    #[pyo3(signature = (terms, relations, max_depth=1))]
    fn expand_query(
        &self,
        py: Python<'_>,
        terms: Vec<String>,
        relations: Vec<String>,
        max_depth: usize,
    ) -> PyResult<Vec<String>> {
        let rels = parse_relations(relations)?;
        let inner = &self.inner;
        let expanded = py.detach(|| {
            let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
            inner.expand_query(&refs, &rels, max_depth)
        });
        Ok(expanded.into_iter().collect())
    }

    /// Sense-aware query expansion.
    ///
    /// Args:
    ///     terms: List of (word, pos, sense_index) tuples.
    ///         pos and sense_index can be None for all-sense expansion.
    ///     relations: List of relation type strings.
    ///
    /// Returns a list of expanded terms.
    fn expand_query_sense_aware(
        &self,
        py: Python<'_>,
        terms: Vec<(String, Option<String>, Option<u32>)>,
        relations: Vec<String>,
    ) -> PyResult<Vec<String>> {
        let rels = parse_relations(relations)?;
        let inner = &self.inner;
        let expanded = py.detach(|| {
            let refs: Vec<(&str, Option<&str>, Option<u32>)> = terms
                .iter()
                .map(|(w, p, s)| (w.as_str(), p.as_deref(), *s))
                .collect();
            inner.expand_query_sense_aware(&refs, &rels)
        });
        Ok(expanded.into_iter().collect())
    }

    /// Build an FST-backed word set from this lexicon.
    ///
    /// Includes every headword and (when `include_inflections` is True)
    /// every aggregated inflection, lower-cased and de-duplicated. Use
    /// the result for fast in-vocabulary membership checks (e.g. as the
    /// `lexicon=` argument to `kaos_nlp_core.quality.compute_metrics`).
    #[pyo3(signature = (include_inflections=true))]
    fn to_fst_set(
        &self,
        py: Python<'_>,
        include_inflections: bool,
    ) -> PyResult<crate::bindings::matching::PyFstSet> {
        let inner = &self.inner;
        let fst = py
            .detach(|| inner.to_fst_set(include_inflections))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(crate::bindings::matching::PyFstSet { inner: fst })
    }

    /// Get senses for a word as a list of dicts.
    fn get_senses(&self, py: Python<'_>, word: &str) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        if let Some(entry) = self.inner.get(word) {
            for s in &entry.senses {
                let d = PyDict::new(py);
                d.set_item("part_of_speech", &s.part_of_speech)?;
                d.set_item("sense_index", s.sense_index)?;
                d.set_item("definition", &s.definition)?;
                d.set_item("synonyms", &s.synonyms)?;
                d.set_item("antonyms", &s.antonyms)?;
                d.set_item("hypernyms", &s.hypernyms)?;
                d.set_item("hyponyms", &s.hyponyms)?;
                list.append(d)?;
            }
        }
        Ok(list.into())
    }
}

/// Register lexicon submodule.
pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "lexicon")?;
    m.add_class::<PyLexicon>()?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.lexicon", &m)?;

    Ok(())
}

// ─── KNC v1/v2/v3 file-format tests ─────────────────────────────────────────
//
// These tests exercise the Lexicon-specific save/load path and the legacy
// version compatibility. They live in the bindings layer because they use
// the binding-side wire helpers (zstd, KNC framing, postcard). Pure-core
// `to_compact` / `from_compact` round-trip tests live in
// `rust/core/lexicon.rs` so they run under `cargo test --no-default-features`.
#[cfg(test)]
mod format_tests {
    use super::*;
    use std::io::Write;

    fn build_test_lexicon() -> Lexicon {
        let mut lex = Lexicon::new();
        lex.add_entry(LexemeEntry {
            word: "contract".to_string(),
            senses: vec![Sense {
                part_of_speech: "noun".to_string(),
                sense_index: 0,
                definition: "A legally binding agreement (DROPPED ON SAVE)".to_string(),
                synonyms: vec!["agreement".to_string(), "pact".to_string()],
                antonyms: vec!["breach".to_string()],
                hypernyms: vec!["legal document".to_string()],
                hyponyms: vec!["employment contract".to_string()],
            }],
            edges: vec![],
            all_synonyms: vec!["agreement".to_string(), "pact".to_string()],
            all_antonyms: vec!["breach".to_string()],
            all_hypernyms: vec!["legal document".to_string()],
            all_hyponyms: vec!["employment contract".to_string()],
            all_inflections: vec!["contracts".to_string()],
            all_derivations: vec!["contractor".to_string()],
            all_collocations: vec!["sign a contract".to_string()],
        });
        lex.add_entry(LexemeEntry {
            word: "agreement".to_string(),
            senses: vec![Sense {
                part_of_speech: "noun".to_string(),
                sense_index: 0,
                definition: "Mutual understanding (DROPPED ON SAVE)".to_string(),
                synonyms: vec!["contract".to_string(), "accord".to_string()],
                antonyms: vec!["disagreement".to_string()],
                hypernyms: vec!["arrangement".to_string()],
                hyponyms: vec!["treaty".to_string()],
            }],
            edges: vec![],
            all_synonyms: vec!["contract".to_string(), "accord".to_string()],
            all_antonyms: vec!["disagreement".to_string()],
            all_hypernyms: vec!["arrangement".to_string()],
            all_hyponyms: vec!["treaty".to_string()],
            all_inflections: vec![],
            all_derivations: vec![],
            all_collocations: vec![],
        });
        lex
    }

    fn tmpfile(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("knc_test_{pid}_{nanos}_{name}"));
        p
    }

    /// Synthesize a KNC v1 (raw postcard, uncompressed) file on disk using the
    /// runtime `Lexicon` schema. Mimics what an old 0.x prerelease wrote.
    fn write_v1_file(lex: &Lexicon, path: &std::path::Path) {
        let bytes = postcard::to_allocvec(lex).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(KNC_MAGIC).unwrap();
        f.write_all(&KNC_VERSION_RAW.to_le_bytes()).unwrap();
        f.write_all(&bytes).unwrap();
    }

    /// Synthesize a KNC v2 (zstd-compressed postcard of the runtime Lexicon)
    /// file on disk. Mimics what `save_bincode_to_path` produces.
    fn write_v2_file(lex: &Lexicon, path: &std::path::Path) {
        let raw = postcard::to_allocvec(lex).unwrap();
        let compressed = zstd::encode_all(raw.as_slice(), 3).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(KNC_MAGIC).unwrap();
        f.write_all(&KNC_VERSION_ZSTD.to_le_bytes()).unwrap();
        f.write_all(&compressed).unwrap();
    }

    #[test]
    fn test_v3_smaller_than_v2() {
        // KNC v3's reason for existing: smaller files on real lexicons.
        // Even on a tiny 2-entry fixture, dropping the definition strings +
        // interning the repeated "noun"/"agreement"/"contract" tokens should
        // beat or at least match v2.
        let lex = build_test_lexicon();

        let v2_path = tmpfile("v2.bin");
        write_v2_file(&lex, &v2_path);
        let v2_size = std::fs::metadata(&v2_path).unwrap().len();

        let v3_path = tmpfile("v3.bin");
        save_lexicon_to_path(&lex, v3_path.to_str().unwrap()).unwrap();
        let v3_size = std::fs::metadata(&v3_path).unwrap().len();

        assert!(
            v3_size < v2_size,
            "v3 ({v3_size} bytes) should be smaller than v2 ({v2_size} bytes)"
        );

        // Confirm v3 file actually has the v3 marker.
        let bytes = std::fs::read(&v3_path).unwrap();
        assert_eq!(&bytes[0..4], KNC_MAGIC);
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, KNC_VERSION_INTERN);

        let _ = std::fs::remove_file(&v2_path);
        let _ = std::fs::remove_file(&v3_path);
    }

    #[test]
    fn test_load_v1_legacy() {
        let lex = build_test_lexicon();
        let path = tmpfile("v1.bin");
        write_v1_file(&lex, &path);

        let loaded = load_lexicon_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.len(), lex.len());
        assert_eq!(loaded.synonyms("contract"), lex.synonyms("contract"));
        // v1 preserves definitions (no compact step).
        let def = &loaded.get("contract").unwrap().senses[0].definition;
        assert!(def.contains("DROPPED ON SAVE"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_v2_legacy() {
        let lex = build_test_lexicon();
        let path = tmpfile("v2.bin");
        write_v2_file(&lex, &path);

        let loaded = load_lexicon_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.len(), lex.len());
        assert_eq!(loaded.synonyms("contract"), lex.synonyms("contract"));
        // v2 also preserves definitions.
        let def = &loaded.get("contract").unwrap().senses[0].definition;
        assert!(def.contains("DROPPED ON SAVE"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_v3_drops_definitions() {
        // Save (which uses v3) → load → confirm definitions are empty even
        // though the in-memory original had populated definitions.
        let lex = build_test_lexicon();
        let path = tmpfile("v3_drop.bin");
        save_lexicon_to_path(&lex, path.to_str().unwrap()).unwrap();

        let loaded = load_lexicon_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.len(), lex.len());

        // Relations must round-trip.
        assert_eq!(loaded.synonyms("contract"), lex.synonyms("contract"));
        assert_eq!(loaded.hypernyms("agreement"), lex.hypernyms("agreement"));

        // Definitions must be empty.
        for word in loaded.words() {
            for sense in &loaded.get(word).unwrap().senses {
                assert_eq!(
                    sense.definition, "",
                    "v3 must drop definitions; got non-empty for {word}"
                );
            }
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_v3_header_layout() {
        // Sanity: v3 file has exactly the documented 6-byte header.
        let lex = build_test_lexicon();
        let path = tmpfile("v3_hdr.bin");
        save_lexicon_to_path(&lex, path.to_str().unwrap()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > KNC_HEADER_LEN);
        assert_eq!(&bytes[0..4], KNC_MAGIC);
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), KNC_VERSION_INTERN);
        let _ = std::fs::remove_file(&path);
    }
}
