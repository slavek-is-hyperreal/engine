// src/compress/decompress.rs
//
// Decompression: Reconstructing original data from a TSLP grammar.
//
// Strategy:
// 1. Rebuild the EML tree by evaluating the TSLP productions.
// 2. Unlift the resulting tree to recover the byte sequence.

use crate::ast::*;
use crate::tslp::grammar::{TslpGrammar, TslpRhs, LeafKind, GrammarNodeId};
use crate::compress::lift::unlift_bytes;
use std::collections::HashMap;
use std::sync::Arc;

/// Decompresses data from a TSLP grammar.
pub fn decompress(grammar: &TslpGrammar) -> Vec<u8> {
    let tree = rebuild_tree(grammar);
    unlift_bytes(&tree)
}

/// Reconstructs the EML tree from grammar productions.
pub fn rebuild_tree(grammar: &TslpGrammar) -> Arc<EmlNode> {
    let mut cache: HashMap<GrammarNodeId, Arc<EmlNode>> = HashMap::new();
    
    // Convert productions to a map for easy lookup
    let prod_map: HashMap<GrammarNodeId, &TslpRhs> = grammar.productions.iter()
        .map(|(id, rhs)| (*id, rhs))
        .collect();
    
    rebuild_node(grammar.start, &prod_map, &mut cache)
}

fn rebuild_node(
    id: GrammarNodeId,
    prod_map: &HashMap<GrammarNodeId, &TslpRhs>,
    cache: &mut HashMap<GrammarNodeId, Arc<EmlNode>>,
) -> Arc<EmlNode> {
    if let Some(node) = cache.get(&id) {
        return node.clone();
    }
    
    let rhs = prod_map.get(&id).expect("Production ID not found");
    
    let node = match rhs {
        TslpRhs::Leaf(LeafKind::One) => one(),
        TslpRhs::Leaf(LeafKind::Var(name)) => var(name),
        TslpRhs::Leaf(LeafKind::Const(v)) => konst(*v),
        TslpRhs::Eml(l, r) => {
            let left = rebuild_node(*l, prod_map, cache);
            let right = rebuild_node(*r, prod_map, cache);
            eml(left, right)
        }
    };
    
    cache.insert(id, node.clone());
    node
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::lift::lift_bytes;
    use crate::tslp::grammar::extract_grammar;

    #[test]
    fn test_full_roundtrip() {
        let data = b"EML compression is algebraic!";
        let tree = lift_bytes(data);
        let grammar = extract_grammar(&tree);
        
        let restored = decompress(&grammar);
        assert_eq!(data.to_vec(), restored);
    }
}
