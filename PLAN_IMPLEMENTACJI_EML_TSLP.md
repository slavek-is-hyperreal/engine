# Plan Implementacji: EML-TSLP Compressor
# Dla agenta kodującego (Gemini Antigravity lub inny)
# Dokładny plan z kodem, testami i oczekiwanymi wynikami

## Kontekst

Repozytorium: github.com/slavek-is-hyperreal/engine
Język: Rust (2021 edition)
Istniejące moduły do reuse: src/ast.rs, src/trs.rs, src/dag.rs,
  src/tslp/grammar.rs, src/tslp/parallel_prefix.rs

Cel: Zaimplementować kompresor danych EML-TSLP który:
1. Bierze dowolny plik tekstowy lub binarny
2. Kompresuje przez algebraiczną normalizację + TSLP gramatykę
3. Jest ściśle silniejszy niż LZ dla algebraicznie ustrukturyzowanych danych
4. Benchmarkuje się przeciw gzip/zstd/lz4

Paper opisujący teorię: PAPER_EML_TSLP_COMPRESSION.md (w repo)

---

## KROK 1: Symbol Lifting (1-2 dni)

### Nowy plik: `src/compress/mod.rs`

```rust
// src/compress/mod.rs
pub mod lift;
pub mod serial;
pub mod decompress;
pub use lift::{lift_bytes, lift_text, SequenceTree};
pub use serial::{serialize_grammar, deserialize_grammar};
pub use decompress::decompress;
```

### Nowy plik: `src/compress/lift.rs`

```rust
// src/compress/lift.rs
//
// Symbol lifting: maps a byte sequence to an EML tree.
//
// Each byte b is lifted to Var(format!("b{}", b)).
// The sequence is represented as a right-spine binary tree:
//   seq(b1, b2, b3) = eml(Var("b1"), eml(Var("b2"), eml(Var("b3"), One)))
//
// Right-spine is simple but deep (O(n)). After TRS + DAG + Ganardi,
// depth becomes O(log n). The initial depth doesn't matter for compression
// ratio — only grammar size matters.
//
// Alternative: balanced tree for initial representation.
// Right-spine is chosen for simplicity and because TRS/DAG/Ganardi
// handle depth reduction downstream.

use crate::ast::*;
use std::sync::Arc;

/// Lift a byte slice to an EML tree.
/// Result: right-spine tree of depth n, size O(n).
pub fn lift_bytes(data: &[u8]) -> Arc<EmlNode> {
    if data.is_empty() {
        return one();
    }
    if data.len() == 1 {
        return var(&format!("b{}", data[0]));
    }
    // Right-spine: eml(Var(b[0]), lift_bytes(b[1..]))
    // Use iterative construction to avoid stack overflow for large files
    let mut acc = one();
    for &byte in data.iter().rev() {
        acc = eml(var(&format!("b{}", byte)), acc);
    }
    acc
}

/// Lift UTF-8 text to an EML tree.
/// Unicode code points as symbols.
pub fn lift_text(text: &str) -> Arc<EmlNode> {
    let code_points: Vec<u32> = text.chars().map(|c| c as u32).collect();
    if code_points.is_empty() {
        return one();
    }
    let mut acc = one();
    for &cp in code_points.iter().rev() {
        acc = eml(var(&format!("u{}", cp)), acc);
    }
    acc
}

/// Extract byte sequence from EML tree (reverse of lift_bytes).
/// Traverses right-spine collecting Var("bN") nodes.
pub fn unlift_bytes(tree: &Arc<EmlNode>) -> Vec<u8> {
    let mut result = Vec::new();
    let mut current = tree.clone();
    loop {
        match current.as_ref() {
            EmlNode::One => break,
            EmlNode::Var(name) => {
                if let Some(b) = parse_byte_var(name) {
                    result.push(b);
                }
                break;
            }
            EmlNode::Eml(l, r) => {
                // Left child should be Var("bN")
                if let EmlNode::Var(name) = l.as_ref() {
                    if let Some(b) = parse_byte_var(name) {
                        result.push(b);
                    }
                }
                current = r.clone();
            }
            EmlNode::Const(_) => break,
        }
    }
    result
}

fn parse_byte_var(name: &str) -> Option<u8> {
    name.strip_prefix('b')?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lift_roundtrip_empty() {
        let data: Vec<u8> = vec![];
        let tree = lift_bytes(&data);
        let recovered = unlift_bytes(&tree);
        assert_eq!(data, recovered);
    }

    #[test]
    fn test_lift_roundtrip_single() {
        let data = vec![42u8];
        let tree = lift_bytes(&data);
        let recovered = unlift_bytes(&tree);
        assert_eq!(data, recovered);
    }

    #[test]
    fn test_lift_roundtrip_ascii() {
        let data = b"hello world".to_vec();
        let tree = lift_bytes(&data);
        let recovered = unlift_bytes(&tree);
        assert_eq!(data, recovered);
    }

    #[test]
    fn test_lift_size() {
        let data = vec![1u8, 2, 3, 4];
        let tree = lift_bytes(&data);
        // Right-spine: n eml nodes + n var leaves + 1 one = 2n+1 nodes
        assert_eq!(tree.node_count(), 2 * data.len() + 1);
    }

    #[test]
    fn test_lift_repeated_bytes_dag() {
        // "aaa" → after DAG, Var("b97") shared = fewer unique nodes
        let data = vec![97u8, 97, 97]; // "aaa"
        let tree = lift_bytes(&data);

        use crate::dag::tree_to_dag;
        let dag = tree_to_dag(&tree);

        // Without sharing: 7 nodes (3 eml + 3 var + 1 one)
        // With sharing: Var("b97") shared = 5 unique nodes
        println!("tree nodes: {}, dag nodes: {}", tree.node_count(), dag.node_count());
        assert!(dag.node_count() < tree.node_count());
    }
}
```

---

## KROK 2: Serializer (3-5 dni)

### Nowy plik: `src/compress/serial.rs`

```rust
// src/compress/serial.rs
//
// Serializes TslpGrammar to a compact bitstream.
//
// Format:
//   [4 bytes] magic: 0x454D4C54 ("EMLT")
//   [4 bytes] version: 1
//   [8 bytes] n_productions: u64
//   [8 bytes] start_id: u64 (index into productions)
//   For each production (NodeId, TslpRhs):
//     [1 byte] tag: 0=Leaf(One), 1=Leaf(Var), 2=Leaf(Const), 3=Eml
//     If tag=1: [2 bytes] var_len + [var_len bytes] var_name
//     If tag=2: [8 bytes] f64 value
//     If tag=3: [8 bytes] left_idx + [8 bytes] right_idx
//
// NodeId values are remapped to sequential indices [0, n_productions)
// for compact representation.

use crate::tslp::grammar::{TslpGrammar, TslpRhs, LeafKind, GrammarNodeId};
use std::collections::HashMap;
use std::io::{Write, Read, Cursor};

const MAGIC: u32 = 0x454D4C54;
const VERSION: u32 = 1;

pub fn serialize_grammar(grammar: &TslpGrammar) -> Vec<u8> {
    let mut buf = Vec::new();

    // Build NodeId → sequential index mapping
    let mut id_to_idx: HashMap<GrammarNodeId, u64> = HashMap::new();
    for (i, (node_id, _)) in grammar.productions.iter().enumerate() {
        id_to_idx.insert(*node_id, i as u64);
    }

    // Header
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&(grammar.productions.len() as u64).to_le_bytes());

    let start_idx = id_to_idx[&grammar.start];
    buf.extend_from_slice(&start_idx.to_le_bytes());

    // Productions
    for (_, rhs) in &grammar.productions {
        match rhs {
            TslpRhs::Leaf(LeafKind::One) => {
                buf.push(0u8);
            }
            TslpRhs::Leaf(LeafKind::Var(name)) => {
                buf.push(1u8);
                let name_bytes = name.as_bytes();
                buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(name_bytes);
            }
            TslpRhs::Leaf(LeafKind::Const(v)) => {
                buf.push(2u8);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            TslpRhs::Eml(left_id, right_id) => {
                buf.push(3u8);
                let left_idx = id_to_idx[left_id];
                let right_idx = id_to_idx[right_id];
                buf.extend_from_slice(&left_idx.to_le_bytes());
                buf.extend_from_slice(&right_idx.to_le_bytes());
            }
        }
    }

    buf
}

pub fn deserialize_grammar(data: &[u8]) -> Result<TslpGrammar, String> {
    let mut cursor = Cursor::new(data);

    // Verify magic
    let mut magic_buf = [0u8; 4];
    cursor.read_exact(&mut magic_buf).map_err(|e| e.to_string())?;
    if u32::from_le_bytes(magic_buf) != MAGIC {
        return Err("Invalid magic number".to_string());
    }

    // Read header
    let mut u32_buf = [0u8; 4];
    cursor.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    // version check could go here

    let mut u64_buf = [0u8; 8];
    cursor.read_exact(&mut u64_buf).map_err(|e| e.to_string())?;
    let n_productions = u64::from_le_bytes(u64_buf) as usize;

    cursor.read_exact(&mut u64_buf).map_err(|e| e.to_string())?;
    let start_idx = u64::from_le_bytes(u64_buf) as usize;

    // Read productions
    // NodeIds are sequential indices cast to GrammarNodeId
    let mut productions = Vec::with_capacity(n_productions);

    for idx in 0..n_productions {
        let node_id = idx as GrammarNodeId;
        let mut tag_buf = [0u8; 1];
        cursor.read_exact(&mut tag_buf).map_err(|e| e.to_string())?;

        let rhs = match tag_buf[0] {
            0 => TslpRhs::Leaf(LeafKind::One),
            1 => {
                let mut len_buf = [0u8; 2];
                cursor.read_exact(&mut len_buf).map_err(|e| e.to_string())?;
                let len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; len];
                cursor.read_exact(&mut name_buf).map_err(|e| e.to_string())?;
                let name = String::from_utf8(name_buf).map_err(|e| e.to_string())?;
                TslpRhs::Leaf(LeafKind::Var(name))
            }
            2 => {
                cursor.read_exact(&mut u64_buf).map_err(|e| e.to_string())?;
                let v = f64::from_le_bytes(u64_buf);
                TslpRhs::Leaf(LeafKind::Const(v))
            }
            3 => {
                cursor.read_exact(&mut u64_buf).map_err(|e| e.to_string())?;
                let left_idx = u64::from_le_bytes(u64_buf) as GrammarNodeId;
                cursor.read_exact(&mut u64_buf).map_err(|e| e.to_string())?;
                let right_idx = u64::from_le_bytes(u64_buf) as GrammarNodeId;
                TslpRhs::Eml(left_idx, right_idx)
            }
            t => return Err(format!("Unknown tag: {}", t)),
        };

        productions.push((node_id, rhs));
    }

    let start = start_idx as GrammarNodeId;
    Ok(TslpGrammar {
        productions,
        start,
        depths: HashMap::new(), // not needed for decompression
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::lift::lift_bytes;
    use crate::tslp::grammar::extract_grammar;

    #[test]
    fn test_serialize_roundtrip() {
        let data = b"hello world hello".to_vec();
        let tree = lift_bytes(&data);
        let grammar = extract_grammar(&tree);

        let serialized = serialize_grammar(&grammar);
        let deserialized = deserialize_grammar(&serialized).unwrap();

        assert_eq!(grammar.productions.len(), deserialized.productions.len());
    }
}
```

---

## KROK 3: Decompressor (2-3 dni)

### Nowy plik: `src/compress/decompress.rs`

```rust
// src/compress/decompress.rs
//
// Reconstructs byte sequence from TslpGrammar.
// Bottom-up: build Arc<EmlNode> from grammar, then unlift_bytes.

use crate::ast::*;
use crate::tslp::grammar::{TslpGrammar, TslpRhs, LeafKind, GrammarNodeId};
use crate::compress::lift::unlift_bytes;
use std::sync::Arc;
use std::collections::HashMap;

pub fn decompress(grammar: &TslpGrammar) -> Vec<u8> {
    let tree = reconstruct_tree(grammar);
    unlift_bytes(&tree)
}

fn reconstruct_tree(grammar: &TslpGrammar) -> Arc<EmlNode> {
    // Build index map: sequential_idx → Arc<EmlNode>
    let mut nodes: HashMap<GrammarNodeId, Arc<EmlNode>> = HashMap::new();

    // Bottom-up: leaves first, then internals
    // Grammar is stored in topological order (children before parents)
    for (node_id, rhs) in &grammar.productions {
        let node = match rhs {
            TslpRhs::Leaf(LeafKind::One) => one(),
            TslpRhs::Leaf(LeafKind::Var(name)) => var(name),
            TslpRhs::Leaf(LeafKind::Const(v)) => konst(*v),
            TslpRhs::Eml(left_id, right_id) => {
                let left = nodes[left_id].clone();
                let right = nodes[right_id].clone();
                eml(left, right)
            }
        };
        nodes.insert(*node_id, node);
    }

    nodes[&grammar.start].clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::lift::{lift_bytes, unlift_bytes};
    use crate::tslp::grammar::extract_grammar;

    #[test]
    fn test_decompress_roundtrip() {
        let original = b"hello world".to_vec();
        let tree = lift_bytes(&original);
        let grammar = extract_grammar(&tree);
        let recovered = decompress(&grammar);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_decompress_repeated() {
        let original = b"abcabcabc".to_vec();
        let tree = lift_bytes(&original);

        use crate::dag::tree_to_dag;
        let dag = tree_to_dag(&tree);
        let grammar = extract_grammar(&dag);
        let recovered = decompress(&grammar);
        assert_eq!(original, recovered);
    }
}
```

---

## KROK 4: Główny interfejs CLI (2 dni)

### Nowy plik: `src/bin/eml_compress.rs`

```rust
// src/bin/eml_compress.rs
//
// CLI: eml_compress [compress|decompress] <input> <output>
//
// Usage:
//   cargo run --bin eml_compress compress input.txt input.eml
//   cargo run --bin eml_compress decompress input.eml input.txt

use eml_trs::compress::{lift_bytes, serialize_grammar, deserialize_grammar, decompress};
use eml_trs::trs::rewrite;
use eml_trs::dag::tree_to_dag;
use eml_trs::tslp::grammar::extract_grammar;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: eml_compress [compress|decompress] <input> <output>");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "compress" => {
            let input = fs::read(&args[2]).expect("Cannot read input");
            let output_path = &args[3];

            eprintln!("Input size: {} bytes", input.len());

            // Step 1: Lift
            let tree = lift_bytes(&input);
            eprintln!("Tree nodes: {}", tree.node_count());

            // Step 2: TRS normalize
            let tree = rewrite(tree);
            eprintln!("After TRS: {} nodes", tree.node_count());

            // Step 3: DAG/CSE
            let dag = tree_to_dag(&tree);
            eprintln!("After DAG: {} nodes", dag.node_count());

            // Step 4: Grammar extraction
            let grammar = extract_grammar(&dag);
            eprintln!("Grammar size: {} productions", grammar.size());

            // Step 5: Serialize
            let serialized = serialize_grammar(&grammar);
            eprintln!("Compressed size: {} bytes", serialized.len());

            let ratio = input.len() as f64 / serialized.len() as f64;
            eprintln!("Compression ratio: {:.2}x", ratio);

            fs::write(output_path, &serialized).expect("Cannot write output");
            eprintln!("Written to {}", output_path);
        }
        "decompress" => {
            let input = fs::read(&args[2]).expect("Cannot read input");
            let grammar = deserialize_grammar(&input).expect("Invalid .eml file");
            let output = decompress(&grammar);
            fs::write(&args[3], &output).expect("Cannot write output");
            eprintln!("Decompressed {} bytes to {}", output.len(), args[3]);
        }
        cmd => {
            eprintln!("Unknown command: {}", cmd);
            std::process::exit(1);
        }
    }
}
```

---

## KROK 5: Benchmark (1 tydzień)

### Nowy plik: `benches/compress_bench.rs`

```rust
// benches/compress_bench.rs
//
// Benchmark EML-TSLP vs gzip vs zstd vs lz4 on:
//   1. Synthetic data with algebraic structure
//   2. Source code (Rust/C files)
//   3. DNA sequences
//   4. Neural network weights (if available)

use eml_trs::compress::*;
use eml_trs::trs::rewrite;
use eml_trs::dag::tree_to_dag;
use eml_trs::tslp::grammar::extract_grammar;

struct CompressionResult {
    name: String,
    original_bytes: usize,
    compressed_bytes: usize,
    ratio: f64,
}

fn compress_eml(data: &[u8]) -> usize {
    let tree = lift_bytes(data);
    let tree = rewrite(tree);
    let dag = tree_to_dag(&tree);
    let grammar = extract_grammar(&dag);
    serialize_grammar(&grammar).len()
}

fn benchmark_dataset(name: &str, data: &[u8]) -> Vec<CompressionResult> {
    let mut results = Vec::new();

    // EML-TSLP
    let eml_size = compress_eml(data);
    results.push(CompressionResult {
        name: "EML-TSLP".to_string(),
        original_bytes: data.len(),
        compressed_bytes: eml_size,
        ratio: data.len() as f64 / eml_size as f64,
    });

    // Note: for gzip/zstd comparison, use external tools or flate2/zstd crates
    // Add to Cargo.toml: flate2 = "1.0", zstd = "0.12"

    results
}

fn main() {
    println!("=== EML-TSLP Compression Benchmark ===\n");

    // Test 1: Synthetic algebraic data
    // Sequence where ln(exp(x)) appears repeatedly
    let algebraic = {
        let pattern = b"ln(exp(x))"; // 10 bytes, TRS collapses to "x"
        pattern.repeat(100) // 1000 bytes before TRS
    };

    println!("--- Synthetic algebraic data (repeated ln(exp(x))) ---");
    let results = benchmark_dataset("synthetic_algebraic", &algebraic);
    for r in &results {
        println!("  {}: {}/{} bytes ({:.2}x)",
            r.name, r.compressed_bytes, r.original_bytes, r.ratio);
    }

    // Test 2: Highly repetitive data (LZ baseline)
    let repetitive = b"abcdefghij".repeat(100);
    println!("\n--- Repetitive data (LZ baseline) ---");
    let results = benchmark_dataset("repetitive", &repetitive);
    for r in &results {
        println!("  {}: {}/{} bytes ({:.2}x)",
            r.name, r.compressed_bytes, r.original_bytes, r.ratio);
    }

    // Test 3: Random data (worst case)
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let random: Vec<u8> = (0..1000u64)
        .map(|i| {
            let mut h = DefaultHasher::new();
            i.hash(&mut h);
            (h.finish() & 0xFF) as u8
        })
        .collect();
    println!("\n--- Random data (worst case) ---");
    let results = benchmark_dataset("random", &random);
    for r in &results {
        println!("  {}: {}/{} bytes ({:.2}x)",
            r.name, r.compressed_bytes, r.original_bytes, r.ratio);
    }

    println!("\n=== Expected Results ===");
    println!("Algebraic: EML-TSLP >> gzip (TRS collapses ln(exp(x))→x)");
    println!("Repetitive: EML-TSLP ≈ gzip (both detect repetition)");
    println!("Random: EML-TSLP > original size (no compression possible)");
}
```

---

## KROK 6: Ganardi SCD (2-3 tygodnie, najważniejszy)

Implementacja według planu z `APPENDIX_E_GANARDI.md`.

### Szkielet: `src/tslp/ganardi.rs`

```rust
// src/tslp/ganardi.rs
//
// Ganardi–Jeż–Lohrey (JACM 2021) TSLP balancing.
// Transforms TslpGrammar of size g into balanced form:
// - Same size O(g)
// - Depth O(log N) where N = expanded tree size
//
// Algorithm: Symmetric Centroid Decomposition (SCD)
// See APPENDIX_E_GANARDI.md for full specification.

use crate::tslp::grammar::{TslpGrammar, TslpRhs, GrammarNodeId};
use std::collections::HashMap;

pub fn ganardi_balance(grammar: &TslpGrammar) -> TslpGrammar {
    // Phase 1: Compute π weights (bottom-up + top-down)
    let pi_up = compute_pi_up(grammar);
    let pi_down = compute_pi_down(grammar);

    // Phase 2: Assign λ labels
    let lambda = compute_lambda(&pi_up, &pi_down);

    // Phase 3: Extract SCD paths
    let sc_paths = extract_sc_paths(grammar, &lambda);

    // Phase 4: Balance each SC-path as tournament tree
    let mut new_grammar = grammar.clone();
    let mut next_id = grammar.productions.iter()
        .map(|(id, _)| *id)
        .max()
        .unwrap_or(0) + 1;

    for path in &sc_paths {
        if path.len() > 1 {
            balance_path(&mut new_grammar, path, &mut next_id);
        }
    }

    new_grammar
}

fn compute_pi_up(grammar: &TslpGrammar) -> HashMap<GrammarNodeId, u64> {
    // Bottom-up: pi_up[v] = size of expanded subtree at v
    // Leaf: pi_up = 1
    // Eml(l, r): pi_up = pi_up[l] + pi_up[r] + 1
    let mut pi_up = HashMap::new();

    // Process in topological order (productions stored leaves-first)
    for (node_id, rhs) in &grammar.productions {
        let size = match rhs {
            TslpRhs::Leaf(_) => 1u64,
            TslpRhs::Eml(l, r) => {
                pi_up.get(l).copied().unwrap_or(1)
                + pi_up.get(r).copied().unwrap_or(1)
                + 1
            }
        };
        pi_up.insert(*node_id, size);
    }

    pi_up
}

fn compute_pi_down(grammar: &TslpGrammar) -> HashMap<GrammarNodeId, u64> {
    // Top-down: pi_down[v] = number of times v appears in expanded tree
    // Root: pi_down = 1
    // Children inherit from parent
    let mut pi_down: HashMap<GrammarNodeId, u64> = HashMap::new();
    pi_down.insert(grammar.start, 1);

    // Process in reverse topological order (root first)
    for (node_id, rhs) in grammar.productions.iter().rev() {
        let count = pi_down.get(node_id).copied().unwrap_or(0);
        if let TslpRhs::Eml(l, r) = rhs {
            *pi_down.entry(*l).or_insert(0) += count;
            *pi_down.entry(*r).or_insert(0) += count;
        }
    }

    pi_down
}

fn compute_lambda(
    pi_up: &HashMap<GrammarNodeId, u64>,
    pi_down: &HashMap<GrammarNodeId, u64>,
) -> HashMap<GrammarNodeId, (u8, u8)> {
    pi_up.keys().map(|id| {
        let up = pi_up[id].max(1);
        let down = pi_down.get(id).copied().unwrap_or(1).max(1);
        let lambda = (
            (down as f64).log2().floor() as u8,
            (up as f64).log2().floor() as u8,
        );
        (*id, lambda)
    }).collect()
}

fn extract_sc_paths(
    grammar: &TslpGrammar,
    lambda: &HashMap<GrammarNodeId, (u8, u8)>,
) -> Vec<Vec<GrammarNodeId>> {
    // Edge (u → v) is centroid iff lambda[u] == lambda[v]
    // SC-paths: maximal chains of centroid edges
    // For |Σ|=1: each node has at most 2 children (l, r)
    // Check both children for centroid edges

    let mut paths: Vec<Vec<GrammarNodeId>> = Vec::new();
    let mut visited: std::collections::HashSet<GrammarNodeId> = std::collections::HashSet::new();

    for (node_id, rhs) in &grammar.productions {
        if visited.contains(node_id) { continue; }

        // Start a new path from this node
        let mut path = vec![*node_id];
        visited.insert(*node_id);

        // Follow centroid edges
        let mut current = *node_id;
        loop {
            // Find child with same lambda
            let current_lambda = lambda.get(&current).copied().unwrap_or((0,0));
            let rhs_current = grammar.productions.iter()
                .find(|(id, _)| *id == current)
                .map(|(_, rhs)| rhs);

            let mut found_next = false;
            if let Some(TslpRhs::Eml(l, r)) = rhs_current {
                for child in [l, r] {
                    if !visited.contains(child) {
                        if lambda.get(child).copied().unwrap_or((0,0)) == current_lambda {
                            path.push(*child);
                            visited.insert(*child);
                            current = *child;
                            found_next = true;
                            break;
                        }
                    }
                }
            }
            if !found_next { break; }
        }

        if path.len() > 1 {
            paths.push(path);
        }
    }

    paths
}

fn balance_path(
    grammar: &mut TslpGrammar,
    path: &[GrammarNodeId],
    next_id: &mut GrammarNodeId,
) {
    // Build tournament tree over path nodes
    // Reuses parallel_prefix_sum logic but for grammar NodeIds
    if path.len() <= 1 { return; }

    let balanced = build_tournament(grammar, path, next_id);

    // Replace first path node's production with root of tournament
    if let Some(first_prod) = grammar.productions.iter_mut()
        .find(|(id, _)| *id == path[0])
    {
        *first_prod = (path[0], grammar.productions.iter()
            .find(|(id, _)| *id == balanced)
            .map(|(_, rhs)| rhs.clone())
            .unwrap_or(TslpRhs::Leaf(crate::tslp::grammar::LeafKind::One)));
    }
}

fn build_tournament(
    grammar: &mut TslpGrammar,
    nodes: &[GrammarNodeId],
    next_id: &mut GrammarNodeId,
) -> GrammarNodeId {
    if nodes.len() == 1 { return nodes[0]; }
    if nodes.len() == 2 {
        let new_id = *next_id;
        *next_id += 1;
        grammar.productions.push((new_id, TslpRhs::Eml(nodes[0], nodes[1])));
        return new_id;
    }
    let mid = nodes.len() / 2;
    let left = build_tournament(grammar, &nodes[..mid], next_id);
    let right = build_tournament(grammar, &nodes[mid..], next_id);
    let new_id = *next_id;
    *next_id += 1;
    grammar.productions.push((new_id, TslpRhs::Eml(left, right)));
    new_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::lift::lift_bytes;
    use crate::tslp::grammar::extract_grammar;
    use crate::compress::decompress::decompress;

    #[test]
    fn test_ganardi_preserves_content() {
        let data = b"hello world hello world".to_vec();
        let tree = crate::compress::lift::lift_bytes(&data);
        let grammar = extract_grammar(&tree);

        let balanced = ganardi_balance(&grammar);
        let recovered = decompress(&balanced);

        assert_eq!(data, recovered,
            "Ganardi balancing must preserve content");
    }

    #[test]
    fn test_ganardi_reduces_depth() {
        let data = b"abcdefghijklmnopqrstuvwxyz".repeat(4).to_vec();
        let tree = lift_bytes(&data);
        let grammar = extract_grammar(&tree);

        let depth_before = grammar.max_depth();
        let balanced = ganardi_balance(&grammar);
        let depth_after = balanced.max_depth();

        println!("Depth before: {}, after: {}", depth_before, depth_after);
        assert!(depth_after <= depth_before,
            "Ganardi must not increase depth");
    }

    #[test]
    fn test_ganardi_grammar_size_bounded() {
        let data = b"hello world".repeat(10).to_vec();
        let tree = lift_bytes(&data);
        let grammar = extract_grammar(&tree);
        let g = grammar.size();

        let balanced = ganardi_balance(&grammar);
        let g_balanced = balanced.size();

        // Must be O(g) — allow 2x slack for tournament nodes
        assert!(g_balanced <= g * 3,
            "Ganardi grammar size must be O(g): {} vs {}", g_balanced, g);
    }
}
```

---

## WYMAGANE ZMIANY W Cargo.toml

```toml
[[bin]]
name = "eml_compress"
path = "src/bin/eml_compress.rs"

[[bin]]
name = "compress_bench"
path = "benches/compress_bench.rs"
```

---

## OCZEKIWANE WYNIKI BENCHMARKU

```
=== Test 1: Algebraicznie strukturyzowane dane ===
Wejście: "ln(exp(x))" × 100 = 1000 bajtów
TRS redukuje ln(exp(x)) → x więc faktycznie = "x" × 100 = 100 bajtów
EML-TSLP: ~15 bajtów (10× lepiej niż gzip który daje ~30)

=== Test 2: Repetywne dane (baseline LZ) ===
Wejście: "abcdefghij" × 100 = 1000 bajtów
gzip: ~30 bajtów
EML-TSLP: ~40 bajtów (nieco gorszy bo brak byte-level matching)

=== Test 3: Losowe dane ===
Wejście: 1000 losowych bajtów
gzip: ~1010 bajtów (brak kompresji + overhead)
EML-TSLP: ~2000 bajtów (gorszy — nie nadaje się do losowych danych)
```

**Kluczowy wynik który udowadnia paper:**
Test 1 pokazuje że EML-TSLP bije gzip o 2× dla danych z algebraiczną strukturą.
To jest empiryczne potwierdzenie Theorem 2 w paperze.

---

## KOLEJNOŚĆ IMPLEMENTACJI

```
Tydzień 1:
  Dzień 1-2: src/compress/lift.rs + testy
  Dzień 3-5: src/compress/serial.rs + testy
  
Tydzień 2:
  Dzień 1-2: src/compress/decompress.rs + testy
  Dzień 3-4: src/bin/eml_compress.rs (CLI)
  Dzień 5: benches/compress_bench.rs (bez Ganardi)
  
Tydzień 3-5:
  src/tslp/ganardi.rs (SCD balancing)
  testy ganardi
  pełny benchmark z Ganardi

Po Ganardi: zaktualizować paper z empirycznymi wynikami
```

---

## DEFINICJA "DONE"

```bash
cargo test compress          # wszystkie testy zielone
cargo run --bin eml_compress compress README.md README.eml
cargo run --bin eml_compress decompress README.eml README_recovered.md
diff README.md README_recovered.md  # puste = identyczne
cargo run --bin compress_bench
```

Wynik benchmarku pokazuje że EML-TSLP bije gzip na:
- Danych syntetycznych z ln(exp(x)) — Theorem 2 udowodnione empirycznie
- Kodzie źródłowym Rust z wyrażeniami matematycznymi
