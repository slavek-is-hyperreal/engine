// src/trs.rs
//
// CONVENTION: eml_count() = number of internal Eml(l,r) nodes
//             node_count() = eml_count() + number of leaves
// Costs from exhaustive search (Odrzywołek paper) = node_count()
// Costs in this project's tests = eml_count() (internal only)

use crate::ast::*;
use std::sync::Arc;

/// Pattern match result.
// Bindings removed (was unused)

/// A single rewriting rule.
pub struct Rule {
    pub name: &'static str,
    pub apply: fn(&Arc<EmlNode>) -> Option<Arc<EmlNode>>,
}

/// Checks if the node is One
fn is_one(n: &EmlNode) -> bool { matches!(n, EmlNode::One) }

/// Check if the node is eml(x, 1) — which is exp(x)
pub fn is_exp_pattern(n: &EmlNode) -> Option<Arc<EmlNode>> {
    if let EmlNode::Eml(l, r) = n {
        if is_one(r) { return Some(l.clone()); }
    }
    None
}

/// Check if the node is ln(x) — pattern: eml(1, eml(eml(1, x), 1))
pub fn is_ln_pattern(n: &EmlNode) -> Option<Arc<EmlNode>> {
    if let EmlNode::Eml(l, r) = n {
        if is_one(l) {
            if let EmlNode::Eml(rl, rr) = r.as_ref() {
                if is_one(rr) {
                    if let EmlNode::Eml(rll, rlr) = rl.as_ref() {
                        if is_one(rll) {
                            return Some(rlr.clone());
                        }
                    }
                }
            }
        }
    }
    None
}

/// Catalog of TRS rules
pub fn get_rules() -> Vec<Rule> {
    vec![
        // RULE 3: ln(exp(x)) → x
        Rule {
            name: "ln_exp_cancel",
            apply: |node| {
                // Looking for pattern: eml(1, eml(eml(1, eml(x, 1)), 1))
                // which is ln(exp(x))
                if let Some(inner) = is_ln_pattern(node) {
                    if let Some(x) = is_exp_pattern(&inner) {
                        return Some(x);
                    }
                }
                None
            },
        },

        // RULE 4: exp(ln(x)) → x
        Rule {
            name: "exp_ln_cancel",
            apply: |node| {
                if let Some(inner) = is_exp_pattern(node) {
                    if let Some(x) = is_ln_pattern(&inner) {
                        return Some(x);
                    }
                }
                None
            },
        },

        // RULE 5: eml(ln(a), 1) → a
        Rule {
            name: "eml_ln_one_absorb",
            apply: |node| {
                if let EmlNode::Eml(l, r) = node.as_ref() {
                    if is_one(r) {
                        if let Some(a) = is_ln_pattern(l) {
                            return Some(a);
                        }
                    }
                }
                None
            },
        },

        // RULE 8: eml(1, 1) → Const(e)
        Rule {
            name: "constant_e",
            apply: |node| {
                if let EmlNode::Eml(l, r) = node.as_ref() {
                    if is_one(l) && is_one(r) {
                        return Some(Arc::new(EmlNode::Const(std::f64::consts::E)));
                    }
                }
                None
            },
        },

        // RULE 9: eml(ln(exp(x)), y) → eml(x, y)
        Rule {
            name: "left_absorb",
            apply: |node| {
                if let EmlNode::Eml(l, r) = node.as_ref() {
                    if let Some(exp_inner) = is_ln_pattern(l) {
                        if let Some(x) = is_exp_pattern(&exp_inner) {
                            return Some(eml(x, r.clone()));
                        }
                    }
                }
                None
            },
        },

        // RULE 10: eml(x, exp(ln(y))) → eml(x, y)
        Rule {
            name: "right_absorb",
            apply: |node| {
                if let EmlNode::Eml(l, r) = node.as_ref() {
                    if let Some(ln_inner) = is_exp_pattern(r) {
                        if let Some(y) = is_ln_pattern(&ln_inner) {
                            return Some(eml(l.clone(), y));
                        }
                    }
                }
                None
            },
        },

        // RULE 11: SwiGLU Fusion
        // Pattern: (gate * up) * sigmoid(gate)
        // Matches: mul_eml(mul_eml(gate, up), sigmoid(gate))
        // Or any structure that yields SiLU(gate) * up
        Rule {
            name: "swiglu_fusion",
            apply: |node| {
                // For now, we look for the specific structure produced by SiLU * up
                // eml(ln(gate * up), 1 + exp(-gate))
                if let EmlNode::Eml(l, r) = node.as_ref() {
                    if let Some(gate_times_up) = is_ln_pattern(l) {
                        // Check if r is 1 + exp(-gate)
                        // add_eml(one, exp(neg(gate)))
                        // In EML: eml(ln(1), exp(eml(ln(0), exp(exp(neg_gate)))))? No.
                        // Let's use a simpler heuristic for now or match the explicit 
                        // structure if we know it.
                        // Actually, let's just implement the fusion in the high-level 
                        // layer builder for now, and keep TRS for general reductions.
                    }
                }
                None
            },
        },
    ]
}

/// Main TRS function: bottom-up traversal to fixpoint
/// Applies rules until none can be applied.
/// 
/// IMPLEMENTATION NOTE: Uses a cache to handle DAG structures efficiently.
/// Without caching, tree-based recursion on shared nodes (Arc) leads to 
/// exponential explosion O(K^L).
pub fn rewrite(node: Arc<EmlNode>) -> Arc<EmlNode> {
    use std::collections::HashMap;
    let mut cache = HashMap::new();
    rewrite_internal(node, &mut cache)
}

fn rewrite_internal(
    node: Arc<EmlNode>, 
    cache: &mut std::collections::HashMap<usize, Arc<EmlNode>>
) -> Arc<EmlNode> {
    // Check cache using the pointer address of the Arc
    let ptr = Arc::as_ptr(&node) as usize;
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }

    // Base case: return leaves unchanged
    if node.is_leaf() {
        cache.insert(ptr, node.clone());
        return node;
    }

    // Bottom-up: first reduce children
    let reduced_children = if let EmlNode::Eml(l, r) = node.as_ref() {
        let new_l = rewrite_internal(l.clone(), cache);
        let new_r = rewrite_internal(r.clone(), cache);
        if !new_l.structural_eq(l) || !new_r.structural_eq(r) {
            eml(new_l, new_r)
        } else {
            node.clone()
        }
    } else {
        node.clone()
    };

    // Fixpoint: apply rules until no match
    let rules = get_rules();
    let mut current = reduced_children;
    let mut changed = true;

    while changed {
        changed = false;
        for rule in &rules {
            if let Some(reduced) = (rule.apply)(&current) {
                // Recursively rewrite the new node (might have shared structures)
                current = rewrite_internal(reduced, cache);
                changed = true;
                break; // reset rules from the beginning
            }
        }
    }

    cache.insert(ptr, current.clone());
    current
}

/// Reduction statistics
pub struct ReductionStats {
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub reduction_percent: f64,
}

pub fn rewrite_with_stats(node: Arc<EmlNode>) -> (Arc<EmlNode>, ReductionStats) {
    let before = node.eml_count();
    let result = rewrite(node);
    let after = result.eml_count();
    let pct = if before > 0 {
        (before - after) as f64 / before as f64 * 100.0
    } else { 0.0 };
    (result, ReductionStats {
        nodes_before: before,
        nodes_after: after,
        reduction_percent: pct,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ln_exp_cancel() {
        // ln(exp(x)) → x
        let x = var("x");
        let tree = ln_node(exp_node(x.clone()));
        let reduced = rewrite(tree);
        assert!(reduced.structural_eq(&x));
    }

    #[test]
    fn test_exp_ln_cancel() {
        // exp(ln(x)) → x
        let x = var("x");
        let tree = exp_node(ln_node(x.clone()));
        let reduced = rewrite(tree);
        assert!(reduced.structural_eq(&x));
    }

    #[test]
    fn test_no_increase() {
        // TRS must never increase the number of nodes
        let tree = eml(ln_node(exp_node(var("x"))), one());
        let before = tree.eml_count();
        let reduced = rewrite(tree);
        assert!(reduced.eml_count() <= before);
    }

    #[test]
    fn test_nested_reduction() {
        // Nested ln(exp()) should reduce
        let x = var("x");
        let tree = ln_node(exp_node(ln_node(exp_node(x.clone()))));
        let reduced = rewrite(tree);
        assert!(reduced.eml_count() < 10); // should be small
    }
}
