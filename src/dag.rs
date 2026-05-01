// src/dag.rs
//
// CONVENTION: eml_count() = number of internal Eml(l,r) nodes
//             node_count() = eml_count() + number of leaves
// Costs from exhaustive search (Odrzywołek paper) = node_count()
// Costs in this project's tests = eml_count() (internal only)

use crate::ast::*;
use std::sync::Arc;
use std::collections::HashMap;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// DAG Node — can be shared
#[derive(Debug)]
pub struct DagNode {
    pub node: Arc<EmlNode>,
    pub id: usize,
    /// Number of occurrences of this node in the full unoptimized tree
    pub ref_count: usize,
}

/// DAG graph with node table
pub struct EmlDag {
    nodes: Vec<DagNode>,
    /// Mapping of structural hash → node id (for deduplication)
    hash_map: HashMap<u64, usize>,
    /// Cache for structural hashes to ensure O(n) hashing complexity.
    /// Key is Arc pointer address.
    hash_cache: HashMap<usize, u64>,
}

impl EmlDag {
    pub fn new() -> Self {
        Self { 
            nodes: Vec::new(), 
            hash_map: HashMap::new(),
            hash_cache: HashMap::new(),
        }
    }

    /// Computes the structural hash of a tree (for deduplication).
    /// Uses caching to prevent O(n^2) complexity on deep trees.
    fn structural_hash(&mut self, node: &Arc<EmlNode>) -> u64 {
        let ptr = Arc::as_ptr(node) as usize;
        if let Some(&h) = self.hash_cache.get(&ptr) {
            return h;
        }

        let h = match node.as_ref() {
            EmlNode::One => {
                let mut s = DefaultHasher::new();
                0u8.hash(&mut s);
                s.finish()
            },
            EmlNode::Var(name) => {
                let mut s = DefaultHasher::new();
                1u8.hash(&mut s);
                name.hash(&mut s);
                s.finish()
            },
            EmlNode::Const(v) => {
                let mut s = DefaultHasher::new();
                2u8.hash(&mut s);
                v.to_bits().hash(&mut s);
                s.finish()
            },
            EmlNode::Eml(l, r) => {
                let hl = self.structural_hash(l);
                let hr = self.structural_hash(r);
                let mut s = DefaultHasher::new();
                3u8.hash(&mut s);
                hl.hash(&mut s);
                hr.hash(&mut s);
                s.finish()
            },
        };

        self.hash_cache.insert(ptr, h);
        h
    }

    /// Adds a node to the DAG, reuses if an identical one already exists
    pub fn add_node(&mut self, node: Arc<EmlNode>) -> usize {
        let h = self.structural_hash(&node);
        if let Some(&id) = self.hash_map.get(&h) {
            self.nodes[id].ref_count += 1;
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(DagNode { node, id, ref_count: 1 });
        self.hash_map.insert(h, id);
        id
    }

    /// Total number of nodes in the DAG (without duplicates)
    pub fn unique_node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of nodes in the tree (with duplicates)
    pub fn tree_node_count(&self) -> usize {
        self.nodes.iter().map(|n| n.ref_count).sum()
    }

    /// Savings from sharing
    pub fn sharing_savings(&self) -> usize {
        self.tree_node_count().saturating_sub(self.unique_node_count())
    }

    /// Clears the structural hash cache. Should be called after integrating a large tree
    /// to free up memory from temporary pointer hashes that won't be reused.
    pub fn clear_hash_cache(&mut self) {
        self.hash_cache.clear();
    }
}

/// Converts a tree to a DAG
pub fn tree_to_dag(root: Arc<EmlNode>) -> EmlDag {
    let mut dag = EmlDag::new();
    add_to_dag(&mut dag, root);
    dag
}

fn add_to_dag(dag: &mut EmlDag, node: Arc<EmlNode>) -> usize {
    if let EmlNode::Eml(l, r) = node.as_ref() {
        add_to_dag(dag, l.clone());
        add_to_dag(dag, r.clone());
    }
    dag.add_node(node)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_subtree() {
        let x = var("x");
        let ln_x = ln_node(x.clone());
        let tree = eml(ln_x.clone(), ln_x.clone());
        let dag = tree_to_dag(tree);
        assert!(dag.unique_node_count() < 15);
    }

    #[test]
    fn test_tree_node_count_correct() {
        let ln_x = ln_node(var("x"));
        let tree = eml(ln_x.clone(), ln_x.clone());
        let dag = tree_to_dag(tree.clone());
        // Deduplication in EMLDag via structural hashing collapses identical subtrees.
        // Result is 6 unique nodes in the DAG.
        assert_eq!(dag.unique_node_count(), 6);
    }
}
