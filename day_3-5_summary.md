# Summary: Day 3-5 — DAG to TSLP Grammar Conversion

## Accomplishments
- Implemented `src/tslp/grammar.rs` for formal TSLP grammar extraction.
- Used pointer identity (`Arc::as_ptr`) to ensure shared nodes in the DAG become shared non-terminals in the grammar.
- Implemented `compression_ratio` metric (Unique Nodes / Grammar Rules).
- Verified extraction correctness with unit tests.
- Exported functionality in `src/tslp/mod.rs`.

## Technical Details
The grammar extraction provides a structured representation of the EML expression:
- **Internal nodes:** `NodeId → eml(NodeId, NodeId)`
- **Leaf nodes:** `NodeId → LeafKind`

This representation confirms that the EML DAG is a valid TSLP, which is the prerequisite for Theorem C3's depth balancing.

## Next Steps
- **Phase 3 (Days 6-7):** Implement Rake-and-Compress heuristic in `src/tslp/rake_compress.rs` to optimize depth for general structures like residual connections.
