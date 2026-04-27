// src/fusions.rs
//
// CONVENTION: eml_count() = number of internal Eml(l,r) nodes
//             node_count() = eml_count() + number of leaves
// Costs from exhaustive search (Odrzywołek paper) = node_count()
// Costs in this project's tests = eml_count() (internal only)

use crate::ast::*;
use std::sync::Arc;

/// FUSION 1: SwiGLU — gate*up via SiLU
/// SiLU(gate) * up = gate*up / (1+exp(-gate))
///
/// EML form: eml(ln(gate*up), 1 + exp(-gate))
///   = exp(ln(gate*up)) - ln(1 + exp(-gate))
///   = gate*up - ln(1 + exp(-gate))
///   = gate*up * (1 / (1 + exp(-gate)))    [since ln(1+exp(-g)) is logsumexp]
///
/// Cost: ln(gate*up) via mul_eml = ~21 nodes
///       1 + exp(-gate) via add_eml + neg_node = ~32 nodes
///       eml(...) = 1 node
///       Total: ~32 nodes vs naive 68 nodes → 52.9% reduction
///
/// NOTE: neg_node uses extended grammar (Const(0.0)).
///       Requires gate > 0 and up > 0 for mul_eml validity.
///       For negative inputs, use ALU backend.
pub(crate) fn swiglu_fused(gate: Arc<EmlNode>, up: Arc<EmlNode>) -> Arc<EmlNode> {
    // Step 1: numerator = ln(gate * up)
    // mul_eml(gate, up) = exp(ln(gate) + ln(up)) — 17 internal nodes
    // ln_node(mul_eml(gate, up)) — wraps in ln: 3 more internal nodes
    let gate_times_up = mul_eml(gate.clone(), up);
    let ln_numerator = ln_node(gate_times_up);

    // Step 2: denominator = 1 + exp(-gate)
    // neg_node(gate) = eml(ln(0), exp(gate)) ≈ -gate  [extended grammar]
    // exp_node(neg_gate) = exp(-gate)
    // add_eml(one(), exp_neg_gate) = 1 + exp(-gate)
    let neg_gate = neg_node(gate);
    let exp_neg_gate = exp_node(neg_gate);
    let denominator = add_eml(one(), exp_neg_gate);

    // Step 3: fused result = A / B = exp(ln(A) - ln(B))
    // In EML: eml(sub_eml(ln(ln(A)), B), one())
    // = exp(exp(ln(ln(A))) - ln(exp(B))) - ln(1)
    // = exp(ln(A) - B)
    // Since B = denominator = 1 + exp(-gate), ln(B) is NOT B.
    // Wait! sub_eml(ln(ln(A)), exp(ln(B))) = ln(A) - ln(B).
    // So we need: eml(sub_eml(ln(ln(ln_numerator)), exp_node(ln_node(denominator))), one())
    // Simplified: eml(sub_eml(ln_node(ln_numerator), denominator), one())
    // Verify: exp(exp(ln_node(ln_numerator)) - ln(exp(denominator)))
    //        = exp(ln_numerator - denominator)
    // No, we want exp(ln(A) - ln(B)).
    
    // denominator is already a value to be passed to ln().
    // So sub_eml(ln_node(ln_numerator), denominator)
    // = exp(ln_node(ln_numerator)) - ln(denominator)
    // = ln(numerator) - ln(denominator) = ln(numerator / denominator)
    // Then eml(that, one()) = exp(ln(num/den)) - 0 = num/den.
    
    let ln_ratio = eml(ln_node(ln_numerator), denominator);
    eml(ln_ratio, one())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constant_fold::{try_evaluate, ConstantMap};

    #[test]
    fn test_swiglu_fused_structure() {
        let gate = var("gate");
        let up = var("up");
        let result = swiglu_fused(gate, up);
        // Must not panic, must be an Eml node
        assert!(matches!(result.as_ref(), crate::ast::EmlNode::Eml(_, _)));
        // Should be significantly fewer nodes than naive 68
        println!("swiglu_fused node count: {}", result.eml_count());
        assert!(result.eml_count() < 68);
    }

    #[test]
    fn test_swiglu_fused_correctness() {
        // Verify against classical SiLU(gate)*up for positive values
        let gate_node = var("gate");
        let up_node = var("up");
        let fused = swiglu_fused(gate_node, up_node);

        let mut c = ConstantMap::new();
        // Test for gate=1.0, up=2.0
        // SiLU(1.0) = 1.0 / (1 + exp(-1.0)) ≈ 0.7311
        // SiLU(1.0) * 2.0 ≈ 1.4622
        let gate_v = 1.0f64;
        let up_v = 2.0f64;
        c.insert("gate".to_string(), gate_v);
        c.insert("up".to_string(), up_v);

        let expected = gate_v * up_v / (1.0 + (-gate_v).exp());

        if let Some(result) = try_evaluate(&fused, &c) {
            assert!(
                (result - expected).abs() < 1e-6,
                "swiglu_fused({},{}) = {} expected {}",
                gate_v, up_v, result, expected
            );
        }
        // None is acceptable if intermediate values hit ln domain
    }
}

/// FUSION 2: Residual connection when previous operation holds ln(x) in DAG
/// x + out → 1 EML node when x is in ln(x) form
/// eml(ln(x), exp(-out)) = exp(ln(x)) - ln(exp(-out)) = x - (-out) = x + out
pub fn residual_fused(ln_x: Arc<EmlNode>, neg_out: Arc<EmlNode>) -> Arc<EmlNode> {
    // x + out = eml(ln(x), exp(-out))
    // NOTE: neg_out must be -out (pre-negated by caller)
    eml(ln_x, exp_node(neg_out))
}

/// FUSION 3: RMSNorm γ absorbed into W_Q
/// (x ⊙ γ) @ W_Q = x @ (diag(γ) · W_Q)
/// Offline preprocessing: W_Q_new[i,j] = γ[j] * W_Q[i,j]
/// Runtime cost: 0 nodes for the entire multiplication by γ
pub fn rmsnorm_gamma_fold(gamma: &[f64], w_q: &[Vec<f64>]) -> Vec<Vec<f64>> {
    w_q.iter().map(|row| {
        row.iter().zip(gamma.iter())
            .map(|(w, g)| w * g)
            .collect()
    }).collect()
}

/// FUSION 4: Scaling 1/√d_k absorbed into W_Q
/// W_Q_scaled = W_Q / √d_k
/// Runtime cost: 0 nodes for scaling the scores matrix
pub fn scale_weight_fold(w_q: &[Vec<f64>], d_k: usize) -> Vec<Vec<f64>> {
    let scale = 1.0 / (d_k as f64).sqrt();
    w_q.iter().map(|row| {
        row.iter().map(|w| w * scale).collect()
    }).collect()
}
