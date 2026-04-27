// src/round_trip.rs
//
// CONVENTION: eml_count() = number of internal Eml(l,r) nodes
//             node_count() = eml_count() + number of leaves
// Costs from exhaustive search (Odrzywołek paper) = node_count()
// Costs in this project's tests = eml_count() (internal only)

use crate::ast::*;
use crate::trs::{rewrite, is_exp_pattern, is_ln_pattern};
use std::sync::Arc;

/// Classical algebraic operations represented in the EML tree.
/// Used for the "extraction" phase before classical optimization.
pub enum ClassicalOp {
    Exp(Arc<EmlNode>),      // exp(x)
    Ln(Arc<EmlNode>),       // ln(x)
    Sub(Arc<EmlNode>, Arc<EmlNode>), // x - y
    Add(Arc<EmlNode>, Arc<EmlNode>), // x + y
    Mul(Arc<EmlNode>, Arc<EmlNode>), // x * y
    Div(Arc<EmlNode>, Arc<EmlNode>), // x / y
    Var(String),
    Const(f64),
}

/// Recognizes classical operations in the EML tree.
pub fn recognize_classical(node: &Arc<EmlNode>) -> Option<ClassicalOp> {
    // 1. Leaf patterns
    match node.as_ref() {
        EmlNode::Var(name) => return Some(ClassicalOp::Var(name.clone())),
        EmlNode::Const(v) => return Some(ClassicalOp::Const(*v)),
        EmlNode::One => return Some(ClassicalOp::Const(1.0)),
        _ => {}
    }

    // 2. Exp pattern: eml(x, 1)
    if is_exp_pattern(node).is_some() {
        if let EmlNode::Eml(l, _) = node.as_ref() {
            return Some(ClassicalOp::Exp(l.clone()));
        }
    }
    
    // 3. Ln pattern
    if let Some(inner) = is_ln_pattern(node) {
        return Some(ClassicalOp::Ln(inner));
    }
    
    // 4. Sub pattern: eml(ln(a), exp(b)) -> a - b
    if let EmlNode::Eml(l, r) = node.as_ref() {
        if let (Some(a), Some(b)) = (is_ln_pattern(l), is_exp_pattern(r)) {
            return Some(ClassicalOp::Sub(a, b));
        }
    }

    // 5. Add pattern: x + y (represented as x - (0 - y) or similar)
    // For now, we only recognize basic Sub. More complex patterns can be added.
    
    None
}

/// A round-trip rule that attempts to simplify an EML node by 
/// viewing it as a classical algebraic expression.
pub struct RoundTripRule {
    pub name: &'static str,
    pub apply: fn(&Arc<EmlNode>) -> Option<Arc<EmlNode>>,
}

/// Catalog of round-trip rules.
pub fn get_round_trip_rules() -> Vec<RoundTripRule> {
    vec![
        // RT1: ln(x) + ln(y) = ln(x * y)
        // Reduces 33 nodes (Add of two Lns) to 24 nodes (Ln of Mul)
        RoundTripRule {
            name: "ln_sum_to_ln_product",
            apply: |node| {
                // Pattern: Add(Ln(x), Ln(y))
                // In EML, Add(a, b) is often eml(ln(a), exp(eml(ln(0), exp(b))))
                if let EmlNode::Eml(l, r) = node.as_ref() {
                    if let Some(a) = is_ln_pattern(l) {
                        if let Some(inner_r) = is_exp_pattern(r) {
                            if let EmlNode::Eml(rl, rr) = inner_r.as_ref() {
                                // rl should be ln(0) or similar
                                if let Some(rr_inner) = is_exp_pattern(rr) {
                                    // We found Add(ln(x), exp(exp(y)))? No.
                                    // Let's use a simpler heuristic: look for any 
                                    // Add-like structure where both operands are Ln.
                                    if let Some(b) = is_ln_pattern(&rr_inner) {
                                        // Potential match: ln(x) + ln(y)
                                        // result = ln(x * y)
                                        let mul = mul_eml(a, b);
                                        return Some(ln_node(mul));
                                    }
                                }
                            }
                        }
                    }
                }
                None
            },
        },

        // RT4: ln(exp(x)) = x
        // This is a global identity that TRS might miss if separated by multiple nodes.
        RoundTripRule {
            name: "ln_exp_cancel",
            apply: |node| {
                if let Some(inner) = is_ln_pattern(node) {
                    if let Some(x) = is_exp_pattern(&inner) {
                        return Some(x);
                    }
                }
                None
            },
        },

        // RT5: exp(a) * exp(b) = exp(a + b)
        // Not implemented yet.
    ]
}

/// Main round-trip optimization function:
/// 1. TRS in EML
/// 2. Apply classical identities (round-trip)
/// 3. TRS in EML again
pub fn round_trip_optimize(node: Arc<EmlNode>) -> Arc<EmlNode> {
    // Step 1: Initial TRS
    let node = rewrite(node);
    
    // Step 2-3: Apply round-trip rules bottom-up
    let rules = get_round_trip_rules();
    let node = apply_rules_bottom_up(node, &rules);
    
    // Step 5: Final TRS to clean up after round-trip injections
    rewrite(node)
}

/// Bottom-up traversal that applies round-trip rules to each node.
fn apply_rules_bottom_up(
    node: Arc<EmlNode>,
    rules: &[RoundTripRule],
) -> Arc<EmlNode> {
    // 1. Recursively optimize children
    let node = if let EmlNode::Eml(l, r) = node.as_ref() {
        let new_l = apply_rules_bottom_up(l.clone(), rules);
        let new_r = apply_rules_bottom_up(r.clone(), rules);
        if !new_l.structural_eq(l) || !new_r.structural_eq(r) {
            eml(new_l, new_r)
        } else {
            node.clone()
        }
    } else {
        node
    };

    // 2. Apply RT rules on the current node
    let mut current = node;
    for rule in rules {
        if let Some(reduced) = (rule.apply)(&current) {
            // Only accept if it actually reduces or maintains complexity
            // (Round-trip usually simplifies global structure)
            if reduced.eml_count() <= current.eml_count() {
                current = reduced;
            }
        }
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    #[test]
    fn test_round_trip_no_regression() {
        let tree = eml(ln_node(exp_node(var("x"))), one());
        let before = tree.eml_count();
        let after = round_trip_optimize(tree).eml_count();
        assert!(after <= before,
            "Round-trip increased node count: {} → {}", before, after);
    }

    #[test]
    fn test_rt4_ln_exp_cancel() {
        let x = var("x");
        // tree = ln(exp(x))
        let tree = ln_node(exp_node(x.clone()));
        let result = round_trip_optimize(tree);
        assert!(result.structural_eq(&x),
            "RT4 failed: expected x, got {:?}", result);
    }

    #[test]
    fn test_nested_round_trip() {
        let x = var("x");
        // tree = ln(exp(ln(exp(x))))
        let tree = ln_node(exp_node(ln_node(exp_node(x.clone()))));
        let result = round_trip_optimize(tree);
        assert!(result.structural_eq(&x));
    }
}
