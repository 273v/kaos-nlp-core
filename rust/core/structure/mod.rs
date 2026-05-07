//! Document-structure analysis layer (P7).
//!
//! Three components, in order of dependency:
//!
//! 1. [`scoring`] — per-line feature extractor (`HeadingFeatureVector`).
//!    Pure forward-only; consumes P1 (`LineRecord`), P3 (`Enumerator`)
//!    and P5 (`BoilerplateRun`) outputs.
//! 2. [`decoder`] — Viterbi sequence decoder over the 7-state line-label
//!    set (`blank, heading, body, list_item, table_row, metadata,
//!    boilerplate`).
//! 3. [`hierarchy`] — heading-stack inferencer that emits parent / depth
//!    relationships for the `heading`-labeled lines.
//!
//! [`lexicon`] supplies the heading-canonicalisation and
//! hierarchy-keyword registries (G4 + G5 of the generality contract).
//!
//! Boundary: this module operates on `&str` and primitive types only.
//! It does not consult AST nodes, `node_ref`s, or any kaos-content
//! type. The kaos-content wrapper composes (1)+(2)+(3) with
//! `DocumentView` / `Annotation` / `Block` and is a separate package.
//! See `docs/INTEGRATION_BOUNDARIES.md`.

pub mod decoder;
pub mod hierarchy;
pub mod lexicon;
pub mod scoring;

pub use decoder::{decode_line_labels, DecoderOptions, EmissionCosts, LineLabel, TransitionMatrix};
pub use hierarchy::{infer_hierarchy, HeadingCandidate, HierarchyOptions};
pub use lexicon::{CustomHeadingLexicon, CustomHierarchyLexicon, HeadingLexicon, HierarchyLexicon};
pub use scoring::{
    citation_density, score_heading_features, HeadingFeatureVector, ScoringOptions, ScoringWeights,
};
