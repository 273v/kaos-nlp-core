//! PyO3 bindings for the Lexicon semantic knowledge graph.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::bindings::util::{
    bincode_getstate, bincode_setstate, load_bincode_from_path, save_bincode_to_path,
};
use crate::core::lexicon::{Edge, LexemeEntry, Lexicon, RelationType, Sense};

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
    /// **Trusted-source only.** The on-disk format is bincode behind a
    /// magic + version header with a configurable size cap
    /// (`KAOS_NLP_MAX_LOAD_BYTES`, default 256 MiB). The header guards
    /// against truncated / mistyped files and unbounded allocations,
    /// but does not protect against an adversarial author of the file.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_owned = path.to_string();
        let inner: Lexicon = py
            .detach(|| load_bincode_from_path(&path_owned))
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

    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let path_owned = path.to_string();
        let inner = &self.inner;
        py.detach(|| save_bincode_to_path(inner, &path_owned))
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
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
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
