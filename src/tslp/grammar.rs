// src/tslp/grammar.rs
//
// Converts an EML DAG (Arc<EmlNode>) into a formal TSLP grammar.
//
// This is the bridge between the EML tree representation and
// the theoretical TSLP framework of Theorem C3.
//
// A TSLP grammar for EML has productions of the form:
//   NodeId → eml(NodeId, NodeId)   (internal node)
//   NodeId → Leaf                   (terminal)
//
// Shared nodes in the DAG (same Arc pointer) become shared
// non-terminals in the grammar — this is the grammar compression.

use crate::ast::*;
use std::collections::HashMap;
use std::sync::Arc;

pub type GrammarNodeId = usize;

/// A production in the TSLP grammar.
#[derive(Debug, Clone)]
pub enum TslpRhs {
    /// Terminal: a leaf node
    Leaf(LeafKind),
    /// Internal: eml(left, right) where left/right are NodeIds
    Eml(GrammarNodeId, GrammarNodeId),
}

#[derive(Debug, Clone)]
pub enum LeafKind {
    One,
    Var(String),
    Const(f64),
}

/// A TSLP grammar for an EML expression.
/// productions[i] = (GrammarNodeId, TslpRhs) — all productions
/// start = GrammarNodeId of the root
pub struct TslpGrammar {
    pub productions: Vec<(GrammarNodeId, TslpRhs)>,
    pub start: GrammarNodeId,
    /// For each GrammarNodeId: its depth in the grammar DAG
    pub depths: HashMap<GrammarNodeId, usize>,
}

impl TslpGrammar {
    /// Number of productions (= number of unique nodes)
    pub fn size(&self) -> usize {
        self.productions.len()
    }
    
    /// Maximum depth in the grammar
    pub fn max_depth(&self) -> usize {
        *self.depths.values().max().unwrap_or(&0)
    }
}

/// Extract a TSLP grammar from an EML DAG.
///
/// Uses Arc::as_ptr as GrammarNodeId — pointer identity = node identity.
/// Shared nodes (Arc clones pointing to same allocation) produce
/// shared non-terminals, achieving grammar compression.
///
/// Time: O(unique nodes) — each node visited once.
pub fn extract_grammar(root: &Arc<EmlNode>) -> TslpGrammar {
    let mut productions = Vec::new();
    let mut visited: HashMap<GrammarNodeId, ()> = HashMap::new();
    let mut depths: HashMap<GrammarNodeId, usize> = HashMap::new();
    
    let start = extract_node(root, &mut productions, &mut visited, &mut depths);
    
    TslpGrammar { productions, start, depths }
}

fn node_id(node: &Arc<EmlNode>) -> GrammarNodeId {
    Arc::as_ptr(node) as usize
}

fn extract_node(
    node: &Arc<EmlNode>,
    productions: &mut Vec<(GrammarNodeId, TslpRhs)>,
    visited: &mut HashMap<GrammarNodeId, ()>,
    depths: &mut HashMap<GrammarNodeId, usize>,
) -> GrammarNodeId {
    let id = node_id(node);
    
    // Already processed (shared node) — just return its NodeId
    if visited.contains_key(&id) {
        return id;
    }
    visited.insert(id, ());
    
    match node.as_ref() {
        EmlNode::One => {
            productions.push((id, TslpRhs::Leaf(LeafKind::One)));
            depths.insert(id, 0);
        }
        EmlNode::Var(name) => {
            productions.push((id, TslpRhs::Leaf(LeafKind::Var(name.clone()))));
            depths.insert(id, 0);
        }
        EmlNode::Const(v) => {
            productions.push((id, TslpRhs::Leaf(LeafKind::Const(*v))));
            depths.insert(id, 0);
        }
        EmlNode::Eml(l, r) => {
            let left_id = extract_node(l, productions, visited, depths);
            let right_id = extract_node(r, productions, visited, depths);
            productions.push((id, TslpRhs::Eml(left_id, right_id)));
            
            let left_depth = depths.get(&left_id).copied().unwrap_or(0);
            let right_depth = depths.get(&right_id).copied().unwrap_or(0);
            depths.insert(id, 1 + left_depth.max(right_depth));
        }
    }
    
    id
}

/// Compression ratio: tree size / grammar size
/// Higher = more sharing in the DAG
pub fn compression_ratio(root: &Arc<EmlNode>, grammar: &TslpGrammar) -> f64 {
    root.node_count() as f64 / grammar.size() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    #[test]
    fn test_shared_node_grammar() {
        // ln(x) appearing twice should produce one production
        let x = var("x");
        let ln_x = ln_node(x.clone());
        let tree = eml(ln_x.clone(), ln_x.clone()); // shared ln_x
        
        let grammar = extract_grammar(&tree);
        
        // Unique nodes: eml(root), ln_x, eml(1,eml(eml(1,x),1)), 1, x
        // Without sharing: would have 2x ln_x = more productions
        // With sharing: ln_x appears once in grammar
        println!("Tree nodes: {}, Grammar size: {}", 
            tree.node_count(), grammar.size());
        assert_eq!(grammar.size(), tree.node_count(),
            "Grammar size must match DAG-aware node count");
    }

    #[test]
    fn test_grammar_depth_equals_tree_depth() {
        let tree = eml(ln_node(var("x")), exp_node(var("y")));
        let grammar = extract_grammar(&tree);
        
        assert_eq!(grammar.max_depth(), tree.depth(),
            "Grammar depth should match tree depth");
    }

    #[test]
    fn test_dot_product_compression() {
        use crate::nn_layer::build_dot_product_eml;
        
        let k = 16;
        let weights: Vec<f32> = (0..k).map(|i| (i+1) as f32 * 0.1).collect();
        let inputs: Vec<Arc<EmlNode>> = (0..k)
            .map(|i| var(&format!("x{}", i)))
            .collect();
        
        let tree = build_dot_product_eml(&inputs, &weights);
        let grammar = extract_grammar(&tree);
        
        println!("K={}: tree_nodes={}, grammar_size={}, ratio={:.1}x",
            k, tree.node_count(), grammar.size(), 
            compression_ratio(&tree, &grammar));
        
        // Grammar size must match unique node count
        assert_eq!(grammar.size(), tree.node_count());
    }
}
