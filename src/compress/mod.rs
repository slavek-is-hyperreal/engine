pub mod lift;
pub mod serial;
pub mod decompress;
pub mod json_loader;

// Re-export the key public API so callers can write `use eml_trs::compress::serialize_grammar`
// instead of `use eml_trs::compress::serial::serialize_grammar`.
pub use serial::{serialize_grammar, deserialize_grammar};
pub use decompress::rebuild_tree;

