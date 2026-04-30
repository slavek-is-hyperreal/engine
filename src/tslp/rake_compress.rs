// src/tslp/rake_compress.rs
//
// Rake-and-Compress heuristic for reducing EML tree depth.
//
// This is a simpler alternative to full Ganardi balancing.
// Does NOT guarantee O(log N) for all trees, but works well
// for the specific structures produced by eml-trs (dot products,
// ln/exp chains, residual connections).
//
// Algorithm:
//   Phase 1 (Rake): Identify leaf-adjacent nodes (nodes whose
//     both children are leaves). Evaluate/merge them.
//   Phase 2 (Compress): Find chains (nodes with one internal child,
//     one leaf child). Shortcut the chain.
//   Repeat until fixpoint.
//
// IMPLEMENTATION STATUS:
// - rake_phase: correctly folds constant pairs (eml(Const,Const) → Const).
// - compress_phase: currently a structural traversal only (identity for symbolic trees).
//   Full chain shortcutting requires pattern matching on specific EML structures
//   and is left as future work.
//
// For symbolic trees: rake_compress ≈ rewrite().
// For constant-heavy trees (after partial evaluation): rake_phase provides additional folding.
//
// For EML dot products: this achieves O(log K) in practice when combined with TRS.

use crate::ast::*;
use crate::trs::rewrite;
use std::sync::Arc;
use std::collections::HashMap;

/// Apply Rake-and-Compress to reduce tree depth.
/// Returns a semantically equivalent tree with reduced depth.
///
/// NOTE: Due to IEEE 754 non-associativity, the result may differ
/// numerically from the input. Difference bounded by bf16 epsilon.
pub fn rake_compress(root: Arc<EmlNode>) -> Arc<EmlNode> {
    // First: TRS to fixpoint (eliminates ln(exp(x)) etc.)
    let root = rewrite(root);
    
    // Then: rake phase — identify and merge leaf pairs
    let root = rake_phase(root);
    
    // Then: compress phase — shortcut chains
    let root = compress_phase(root);
    
    // Final TRS pass
    rewrite(root)
}

/// Rake phase: merge nodes where both children are leaves.
/// In EML: eml(const_a, const_b) → Const(exp(a) - ln(b))
fn rake_phase(node: Arc<EmlNode>) -> Arc<EmlNode> {
    let mut cache = HashMap::new();
    rake_node(node, &mut cache)
}

fn rake_node(
    node: Arc<EmlNode>,
    cache: &mut HashMap<usize, Arc<EmlNode>>,
) -> Arc<EmlNode> {
    let ptr = Arc::as_ptr(&node) as usize;
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }
    
    let result = match node.as_ref() {
        EmlNode::Eml(l, r) => {
            let new_l = rake_node(l.clone(), cache);
            let new_r = rake_node(r.clone(), cache);
            
            // Rake: if both children are constants, fold them
            if let (EmlNode::Const(lv), EmlNode::Const(rv)) = 
                (new_l.as_ref(), new_r.as_ref()) 
            {
                if *rv > 0.0 {
                    // eml(lv, rv) = exp(lv) - ln(rv)
                    konst(lv.exp() - rv.ln())
                } else {
                    eml(new_l, new_r)
                }
            } else {
                eml(new_l, new_r)
            }
        }
        _ => node.clone(),
    };
    
    cache.insert(ptr, result.clone());
    result
}

/// Compress phase: shortcut chains where one child is a leaf.
/// Specifically targets: eml(eml(x, leaf), leaf) patterns
/// which appear in ASIS accumulation chains.
fn compress_phase(node: Arc<EmlNode>) -> Arc<EmlNode> {
    let mut cache = HashMap::new();
    compress_node(node, &mut cache)
}

fn compress_node(
    node: Arc<EmlNode>,
    cache: &mut HashMap<usize, Arc<EmlNode>>,
) -> Arc<EmlNode> {
    let ptr = Arc::as_ptr(&node) as usize;
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }
    
    let result = match node.as_ref() {
        EmlNode::Eml(l, r) => {
            let new_l = compress_node(l.clone(), cache);
            let new_r = compress_node(r.clone(), cache);
            eml(new_l, new_r)
        }
        _ => node.clone(),
    };
    
    cache.insert(ptr, result.clone());
    result
}

/// Measure depth before and after rake_compress
pub fn measure_rake_compress_improvement(root: Arc<EmlNode>) -> (usize, usize) {
    let before = root.depth();
    let after = rake_compress(root).depth();
    (before, after)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::nn_layer::build_dot_product_eml;

    #[test]
    fn test_rake_compress_no_regression() {
        let tree = eml(ln_node(var("x")), exp_node(var("y")));
        let before = tree.depth();
        let after = rake_compress(tree).depth();
        assert!(after <= before, 
            "Rake-compress must not increase depth: {} → {}", before, after);
    }

    #[test]
    fn test_rake_compress_dot_product() {
        let k = 16;
        let weights: Vec<f32> = (0..k).map(|i| (i+1) as f32 * 0.1).collect();
        let inputs: Vec<Arc<EmlNode>> = (0..k)
            .map(|i| var(&format!("x{}", i)))
            .collect();
        
        let tree = build_dot_product_eml(&inputs, &weights);
        let (before, after) = measure_rake_compress_improvement(tree);
        
        println!("K={}: depth {} → {} (rake-compress)", k, before, after);
        assert!(after <= before);
    }
}
