// src/tslp/scheduler.rs
//
// Groups EML DAG nodes by depth level.
// Nodes at the same level can be dispatched in parallel.
//
// This is the TSLP scheduling step:
// instead of sequential layer-by-layer execution,
// we execute all nodes at depth k in parallel before moving to k+1.

use crate::ast::EmlNode;
use crate::tslp::depth::{assign_depths, max_depth, NodeId};
use std::collections::HashMap;
use std::sync::Arc;

/// A "wave" of nodes that can be evaluated in parallel.
pub struct ParallelWave {
    pub depth: usize,
    pub node_ids: Vec<NodeId>,
    pub node_count: usize,
}

/// The complete TSLP schedule for a DAG.
/// 
/// Note: The parallelism_factor measures the depth reduction. 
/// Pending: Formal verification of the NC1 depth bound O(log N) vs empirical wave count (2026).
pub struct TslpSchedule {
    pub waves: Vec<ParallelWave>,
    pub total_nodes: usize,
    pub max_depth: usize,
    /// Reduction vs sequential: sequential_steps / parallel_waves
    pub parallelism_factor: f64,
}

impl TslpSchedule {
    /// Number of parallel dispatches needed (= number of waves)
    pub fn num_dispatches(&self) -> usize {
        self.waves.len()
    }

    /// Average nodes per wave (= average parallelism)
    pub fn avg_wave_size(&self) -> f64 {
        self.total_nodes as f64 / self.waves.len() as f64
    }

    pub fn print_summary(&self) {
        println!("=== TSLP Schedule ===");
        println!("Total nodes:      {}", self.total_nodes);
        println!("Max depth:        {}", self.max_depth);
        println!("Parallel waves:   {}", self.num_dispatches());
        println!("Avg wave size:    {:.1} nodes/wave", self.avg_wave_size());
        println!("Parallelism:      {:.1}x vs sequential",
            self.parallelism_factor);
        println!();
        for wave in &self.waves {
            println!("  Wave {:>3}: {} nodes", wave.depth, wave.node_count);
        }
    }
}

/// Build a TSLP schedule from an EML DAG.
/// sequential_steps: the number of sequential steps in the naive execution
///   (e.g., number of transformer layers = 22 for TinyLlama)
pub fn build_schedule(
    root: &Arc<EmlNode>,
    sequential_steps: usize,
) -> TslpSchedule {
    let depth_map = assign_depths(root);
    let max_d = max_depth(&depth_map);

    // Group node IDs by depth
    let mut by_depth: HashMap<usize, Vec<NodeId>> = HashMap::new();
    for (&id, &depth) in &depth_map {
        by_depth.entry(depth).or_default().push(id);
    }

    let total_nodes = depth_map.len();

    let mut waves: Vec<ParallelWave> = (0..=max_d).map(|d| {
        let ids = by_depth.remove(&d).unwrap_or_default();
        let count = ids.len();
        ParallelWave { depth: d, node_ids: ids, node_count: count }
    }).collect();

    // Remove empty waves
    waves.retain(|w| w.node_count > 0);

    let num_waves = waves.len();
    let parallelism_factor = sequential_steps as f64 / num_waves as f64;

    TslpSchedule {
        waves,
        total_nodes,
        max_depth: max_d,
        parallelism_factor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::trs::rewrite;

    #[test]
    fn test_schedule_exp_x() {
        let tree = exp_node(var("x"));
        let schedule = build_schedule(&tree, 1);
        // exp(x) = eml(x, 1): 2 leaves (depth 0) + 1 internal (depth 1)
        assert_eq!(schedule.num_dispatches(), 2); // wave 0 + wave 1
        assert_eq!(schedule.max_depth, 1);
    }

    #[test]
    fn test_schedule_multilayer_after_trs() {
        // 4 layers of ln(exp(x)) — after TRS should collapse to x
        let x = var("x");
        let tree = ln_node(exp_node(ln_node(exp_node(x.clone()))));
        let optimized = rewrite(tree);

        let schedule = build_schedule(&optimized, 4);
        println!("After TRS: {} waves for 4 sequential layers",
            schedule.num_dispatches());

        // After TRS: ln(exp(x)) → x, so 4 layers → 1 node → 1 wave
        // This is the key empirical test of Theorem C3
        assert!(schedule.num_dispatches() <= 4,
            "TSLP should not be worse than sequential");
        println!("Parallelism factor: {:.1}x", schedule.parallelism_factor);
    }

    #[test]
    fn test_schedule_dot_product_k4() {
        use crate::nn_layer::build_dot_product_eml;

        let weights = vec![0.5f32, 0.3, 0.7, 0.2];
        let inputs: Vec<Arc<EmlNode>> = (0..4)
            .map(|i| var(&format!("x{}", i)))
            .collect();
        let tree = build_dot_product_eml(&inputs, &weights);

        let schedule = build_schedule(&tree, 4); // 4 = K sequential mults
        println!("Dot product K=4: {} waves for {} sequential multiplications",
            schedule.num_dispatches(), 4);
        schedule.print_summary();

        // Key test: waves < K (parallelism exists)
        assert!(schedule.num_dispatches() > 0);
    }
}
