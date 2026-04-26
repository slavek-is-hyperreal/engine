// src/dag.rs
//
// KONWENCJA: eml_count() = liczba węzłów wewnętrznych Eml(l,r)
//            node_count() = eml_count() + liczba liści
// Koszty z exhaustive search (paper Odrzywołka) = node_count()
// Koszty w testach tego projektu = eml_count() (tylko wewnętrzne)

use crate::ast::*;
use std::sync::Arc;
use std::collections::HashMap;

/// Węzeł DAG — może być współdzielony
#[derive(Debug)]
pub struct DagNode {
    pub node: Arc<EmlNode>,
    pub id: usize,
    pub ref_count: usize,  // ile razy jest reużywany
}

/// Graf DAG z tabelą węzłów
pub struct EmlDag {
    nodes: Vec<DagNode>,
    /// Mapowanie hash drzewa → id węzła (dla deduplicacji)
    hash_map: HashMap<String, usize>,
}

impl EmlDag {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), hash_map: HashMap::new() }
    }

    /// Oblicza hash strukturalny drzewa (dla deduplicacji)
    fn structural_hash(node: &EmlNode) -> String {
        match node {
            EmlNode::One => "1".to_string(),
            EmlNode::Var(s) => format!("v:{}", s),
            EmlNode::Const(v) => format!("c:{:.15}", v),
            EmlNode::Eml(l, r) => format!(
                "eml({},{})",
                Self::structural_hash(l),
                Self::structural_hash(r)
            ),
        }
    }

    /// Dodaje węzeł do DAG, reużywa jeśli identyczny już istnieje
    pub fn add_node(&mut self, node: Arc<EmlNode>) -> usize {
        let hash = Self::structural_hash(&node);
        if let Some(&id) = self.hash_map.get(&hash) {
            self.nodes[id].ref_count += 1;
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(DagNode { node, id, ref_count: 1 });
        self.hash_map.insert(hash, id);
        id
    }

    /// Całkowita liczba węzłów w DAG (bez duplikatów)
    pub fn unique_node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Całkowita liczba węzłów w drzewie (z duplikatami)
    pub fn tree_node_count(&self) -> usize {
        // Zgodnie z konwencją node_count(), całkowita liczba węzłów w drzewie to suma
        // wszystkich wystąpień (ref_count) poszczególnych unikalnych węzłów DAG.
        self.nodes.iter().map(|n| n.ref_count).sum()
    }

    /// Oszczędność przez współdzielenie
    pub fn sharing_savings(&self) -> usize {
        self.tree_node_count().saturating_sub(self.unique_node_count())
    }
}

/// Konwertuje drzewo do DAG
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
        // Drzewo gdzie ln(x) pojawia się dwa razy
        let x = var("x");
        let ln_x = ln_node(x.clone());
        // eml(ln(x), ln(x)) — ln(x) powinno być współdzielone
        let tree = eml(ln_x.clone(), ln_x.clone());
        let dag = tree_to_dag(tree);
        // Bez DAG: 7 + 7 + 1 = 15 węzłów
        // Z DAG: ln(x) raz (7) + eml (1) = 8 unikalnych
        assert!(dag.unique_node_count() < 15);
    }

    #[test]
    fn test_tree_node_count_correct() {
        let ln_x = ln_node(var("x"));
        let tree = eml(ln_x.clone(), ln_x.clone());
        let dag = tree_to_dag(tree.clone());
        assert_eq!(dag.tree_node_count(), tree.node_count());
        assert!(dag.unique_node_count() < tree.node_count());
    }
}
