pub mod inverted_index;
pub mod similarity_matrix;
pub mod span_index;
pub mod sparse_term_matrix;
pub mod vocabulary;

pub use span_index::{LabeledSpan, SpanIndex, SpanIndexError};
