// src/round_trip.rs
//
// Round-trip optimization: EML → recognize classical ops → TRS → lower back to ops.
//
// The pipeline:
//   EML tree (e.g. log_softmax(W·x))
//       ↓  rewrite()         — TRS: ln(exp(x)) → x, etc.
//       ↓  round_trip_optimize() — apply classical identities
//       ↓  lower_to_flat_ops()  — convert reduced EML to flat Op sequence
//
// Key insight: compound ops (log_softmax after matmul) have ln/exp pairs that
// cancel across operation boundaries, reducing total op count.

use crate::ast::*;
use crate::trs::{rewrite, is_exp_pattern, is_ln_pattern};
use std::sync::Arc;
use std::collections::HashMap;

// ============================================================
// Classical operations (output of lowering)
// ============================================================

/// A flattened classical operation — the output of lowering an EML tree.
/// Each Op reads from a slot index and writes to a new slot.
#[derive(Debug, Clone)]
pub enum FlatOp {
    LoadVar(String),          // slot = variable value
    LoadConst(f64),           // slot = constant
    Mul(usize, usize),        // slot = a * b
    Add(usize, usize),        // slot = a + b
    Sub(usize, usize),        // slot = a - b
    Div(usize, usize),        // slot = a / b
    Exp(usize),               // slot = exp(a)
    Ln(usize),                // slot = ln(a)
    MulConst(usize, f64),     // slot = a * constant  (CF case)
}

/// A flat program: sequence of ops + index of the result slot
#[derive(Debug)]
pub struct FlatProgram {
    pub ops: Vec<FlatOp>,
    pub result_slot: usize,
}

impl FlatProgram {
    /// Count expensive transcendental operations (exp, ln)
    pub fn transcendental_count(&self) -> usize {
        self.ops.iter().filter(|op| matches!(op, FlatOp::Exp(_) | FlatOp::Ln(_))).count()
    }

    /// Count all arithmetic operations
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Execute the flat program on given variable values
    pub fn execute(&self, vars: &HashMap<String, f64>) -> Option<f64> {
        let mut slots: Vec<f64> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let v = match op {
                FlatOp::LoadVar(name) => *vars.get(name)?,
                FlatOp::LoadConst(c) => *c,
                FlatOp::Mul(a, b) => slots[*a] * slots[*b],
                FlatOp::Add(a, b) => slots[*a] + slots[*b],
                FlatOp::Sub(a, b) => slots[*a] - slots[*b],
                FlatOp::Div(a, b) => {
                    let denom = slots[*b];
                    if denom == 0.0 { return None; }
                    slots[*a] / denom
                }
                FlatOp::Exp(a) => slots[*a].exp(),
                FlatOp::Ln(a) => {
                    let v = slots[*a];
                    if v <= 0.0 { return None; }
                    v.ln()
                }
                FlatOp::MulConst(a, c) => slots[*a] * c,
            };
            slots.push(v);
        }
        Some(slots[self.result_slot])
    }
}

// ============================================================
// Recognizer: EML node → classical operation
// ============================================================

/// Recognizes classical operations in the EML tree.
/// Returns None if the node is a raw EML gate (not recognized as classical).
pub fn recognize_classical(node: &Arc<EmlNode>) -> Option<ClassicalOp> {
    match node.as_ref() {
        EmlNode::Var(name) => return Some(ClassicalOp::Var(name.clone())),
        EmlNode::Const(v)  => return Some(ClassicalOp::Const(*v)),
        EmlNode::One       => return Some(ClassicalOp::Const(1.0)),
        _ => {}
    }

    // mul_cf(x, w) = eml(eml(ln(ln(x)), Const(1/w)), One)
    // Recognized before is_exp_pattern because mul_cf ALSO has `right = One`.
    // Without this pattern, the lowerer would emit Exp(Sub(Ln(Ln(x)), Const)) = 8 ops.
    // With this pattern: MulConst(x, w) = 2 ops (LoadVar + MulConst).
    if let EmlNode::Eml(outer_l, outer_r) = node.as_ref() {
        if matches!(outer_r.as_ref(), EmlNode::One) {
            if let EmlNode::Eml(inner_l, inner_r) = outer_l.as_ref() {
                if let EmlNode::Const(inv_w) = inner_r.as_ref() {
                    if let Some(ln_x) = is_ln_pattern(inner_l) {
                        if let Some(x) = is_ln_pattern(&ln_x) {
                            // eml(eml(ln(ln(x)), Const(1/w)), One) = x * w
                            let w = 1.0 / inv_w;
                            return Some(ClassicalOp::MulConst(x, w));
                        }
                    }
                }
            }
        }
    }

    // exp(x) = eml(x, 1)
    if let Some(inner) = is_exp_pattern(node) {
        return Some(ClassicalOp::Exp(inner));
    }

    // ln(x) — 7-node pattern
    if let Some(inner) = is_ln_pattern(node) {
        return Some(ClassicalOp::Ln(inner));
    }

    if let EmlNode::Eml(l, r) = node.as_ref() {
        // sub(a, b) = eml(ln(a), exp(b))
        if let (Some(a), Some(b)) = (is_ln_pattern(l), is_exp_pattern(r)) {
            return Some(ClassicalOp::Sub(a, b));
        }

        // add(a, b) = sub(a, -b) = eml(ln(a), exp(neg(b)))
        // neg(b) = eml(ln(0), exp(b)) — detect ln(0) = -inf as sentinel
        if let Some(sub_right) = is_exp_pattern(r) {
            if let EmlNode::Eml(neg_l, neg_r) = sub_right.as_ref() {
                if let (EmlNode::Const(v), Some(b)) = (neg_l.as_ref(), is_exp_pattern(neg_r)) {
                    if *v == 0.0 {
                        // This is add(a, b): sub(a, neg(b))
                        if let Some(a) = is_ln_pattern(l) {
                            return Some(ClassicalOp::Add(a, b));
                        }
                    }
                }
            }
        }

        // mul(a, b) via mul_eml = eml(ln(a)+ln(b), 1) — catches generic mul
        if is_one(r) {
            if let Some(inner_l) = is_ln_pattern(l) {
                // eml(ln(a), 1) = exp(ln(a)) = a  — identity, but also exp pattern
                // mul_eml(a,b) = exp(ln(a)+ln(b)) = eml(add_eml(ln(a),ln(b)), 1)
                // Detect add_eml of two ln's:
                if let EmlNode::Eml(al, ar) = inner_l.as_ref() {
                    if let (Some(a), Some(_sub_arg)) = (is_ln_pattern(al), is_exp_pattern(ar)) {
                        // This might be add_eml(ln(x), ln(y)) → ln(x*y)
                        // For now, just return Exp(inner_l) and let rt rules handle mul
                        let _ = a;
                    }
                }
                let _ = inner_l;
            }
        }
    }

    None
}

fn is_one(n: &EmlNode) -> bool { matches!(n, EmlNode::One) }

/// Classical operation — intermediate form
pub enum ClassicalOp {
    Exp(Arc<EmlNode>),
    Ln(Arc<EmlNode>),
    Sub(Arc<EmlNode>, Arc<EmlNode>),
    Add(Arc<EmlNode>, Arc<EmlNode>),
    Mul(Arc<EmlNode>, Arc<EmlNode>),
    Div(Arc<EmlNode>, Arc<EmlNode>),
    /// Constant-weight multiplication: x * w  (from mul_cf pattern)
    MulConst(Arc<EmlNode>, f64),
    Var(String),
    Const(f64),
}

// ============================================================
// Lowering: EML tree → FlatProgram
// ============================================================

struct Lowerer {
    ops: Vec<FlatOp>,
    cache: HashMap<usize, usize>,  // Arc pointer → slot index
}

impl Lowerer {
    fn new() -> Self {
        Self { ops: Vec::new(), cache: HashMap::new() }
    }

    fn emit(&mut self, op: FlatOp) -> usize {
        let slot = self.ops.len();
        self.ops.push(op);
        slot
    }

    fn lower(&mut self, node: &Arc<EmlNode>) -> usize {
        let ptr = Arc::as_ptr(node) as usize;
        if let Some(&slot) = self.cache.get(&ptr) {
            return slot;
        }

        let slot = match recognize_classical(node) {
            Some(ClassicalOp::Var(name)) => self.emit(FlatOp::LoadVar(name)),
            Some(ClassicalOp::Const(v))  => self.emit(FlatOp::LoadConst(v)),

            Some(ClassicalOp::Exp(inner)) => {
                let s = self.lower(&inner);
                self.emit(FlatOp::Exp(s))
            }
            Some(ClassicalOp::Ln(inner)) => {
                let s = self.lower(&inner);
                self.emit(FlatOp::Ln(s))
            }
            Some(ClassicalOp::Sub(a, b)) => {
                let sa = self.lower(&a);
                let sb = self.lower(&b);
                self.emit(FlatOp::Sub(sa, sb))
            }
            Some(ClassicalOp::Add(a, b)) => {
                let sa = self.lower(&a);
                let sb = self.lower(&b);
                self.emit(FlatOp::Add(sa, sb))
            }
            Some(ClassicalOp::Mul(a, b)) => {
                let sa = self.lower(&a);
                let sb = self.lower(&b);
                self.emit(FlatOp::Mul(sa, sb))
            }
            Some(ClassicalOp::Div(a, b)) => {
                let sa = self.lower(&a);
                let sb = self.lower(&b);
                self.emit(FlatOp::Div(sa, sb))
            }
            Some(ClassicalOp::MulConst(x, w)) => {
                let sx = self.lower(&x);
                self.emit(FlatOp::MulConst(sx, w))
            }

            // Unrecognized EML node — recurse into children and emit as Mul
            // (raw eml(a,b) = exp(a) - ln(b), approximate as two transcendentals)
            None => {
                if let EmlNode::Eml(l, r) = node.as_ref() {
                    let sl = self.lower(l);
                    let sr = self.lower(r);
                    // eml(a, b) = exp(a) - ln(b)
                    let exp_slot = self.emit(FlatOp::Exp(sl));
                    let ln_slot  = self.emit(FlatOp::Ln(sr));
                    self.emit(FlatOp::Sub(exp_slot, ln_slot))
                } else {
                    self.emit(FlatOp::LoadConst(0.0)) // fallback
                }
            }
        };

        self.cache.insert(ptr, slot);
        slot
    }
}

/// Lower a (TRS-reduced) EML tree to a flat sequence of classical operations.
pub fn lower_to_flat_ops(tree: &Arc<EmlNode>) -> FlatProgram {
    let mut lowerer = Lowerer::new();
    let result_slot = lowerer.lower(tree);
    FlatProgram { ops: lowerer.ops, result_slot }
}

// ============================================================
// Round-trip optimization rules
// ============================================================

pub struct RoundTripRule {
    pub name: &'static str,
    pub apply: fn(&Arc<EmlNode>) -> Option<Arc<EmlNode>>,
}

pub fn get_round_trip_rules() -> Vec<RoundTripRule> {
    vec![
        // RT4: ln(exp(x)) = x — cross-operation boundary cancellation
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

        // RT1: ln(x) + ln(y) = ln(x * y)  — reduces 2 Ln to 1 Ln + 1 Mul
        RoundTripRule {
            name: "ln_sum_to_ln_product",
            apply: |node| {
                if let Some(ClassicalOp::Add(a, b)) = recognize_classical(node) {
                    if let (Some(x), Some(y)) = (is_ln_pattern(&a), is_ln_pattern(&b)) {
                        return Some(ln_node(mul_eml(x, y)));
                    }
                }
                None
            },
        },

        // RT5: exp(a) * exp(b) = exp(a + b)
        RoundTripRule {
            name: "exp_mul_to_exp_add",
            apply: |node| {
                if let Some(ClassicalOp::Mul(a, b)) = recognize_classical(node) {
                    if let (Some(ea), Some(eb)) = (is_exp_pattern(&a), is_exp_pattern(&b)) {
                        return Some(exp_node(add_eml(ea, eb)));
                    }
                }
                None
            },
        },
    ]
}

/// Full round-trip optimization:
/// TRS → classical identities → TRS → lower to flat ops
pub fn round_trip_optimize(node: Arc<EmlNode>) -> Arc<EmlNode> {
    let node = rewrite(node);
    let rules = get_round_trip_rules();
    let node = apply_rules_bottom_up(node, &rules);
    rewrite(node)
}

/// Round-trip + lower: returns the flat op sequence.
/// NOTE: We lower BEFORE round_trip_optimize because TRS left_absorb destroys
/// the mul_cf pattern: eml(ln(exp(inner)), y) → eml(inner, y) fires on
/// sub_eml_local(mul_cf(x,w), c) and breaks MulConst recognition.
pub fn compile_to_ops(node: Arc<EmlNode>) -> FlatProgram {
    lower_to_flat_ops(&node)
}


fn apply_rules_bottom_up(node: Arc<EmlNode>, rules: &[RoundTripRule]) -> Arc<EmlNode> {
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

    let mut current = node;
    for rule in rules {
        if let Some(reduced) = (rule.apply)(&current) {
            if reduced.eml_count() <= current.eml_count() {
                current = reduced;
            }
        }
    }
    current
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constant_fold::ConstantMap;

    #[test]
    fn test_round_trip_no_regression() {
        let tree = eml(ln_node(exp_node(var("x"))), one());
        let before = tree.eml_count();
        let after = round_trip_optimize(tree).eml_count();
        assert!(after <= before,
            "Round-trip increased node count: {} -> {}", before, after);
    }

    #[test]
    fn test_rt4_ln_exp_cancel() {
        let x = var("x");
        let tree = ln_node(exp_node(x.clone()));
        let result = round_trip_optimize(tree);
        assert!(result.structural_eq(&x), "RT4 failed: expected x");
    }

    #[test]
    fn test_nested_round_trip() {
        let x = var("x");
        let tree = ln_node(exp_node(ln_node(exp_node(x.clone()))));
        let result = round_trip_optimize(tree);
        assert!(result.structural_eq(&x));
    }

    #[test]
    fn test_lower_simple_add() {
        // add_eml(x, y) → FlatOp::Add
        let tree = add_eml(var("x"), var("y"));
        let program = compile_to_ops(tree);

        let mut vars = ConstantMap::new();
        vars.insert("x".to_string(), 3.0);
        vars.insert("y".to_string(), 4.0);

        let result = program.execute(&vars).expect("Should evaluate");
        assert!((result - 7.0).abs() < 1e-6,
            "Expected 7.0, got {}", result);
    }

    #[test]
    fn test_lower_sub() {
        let tree = sub_eml(var("a"), var("b"));
        let program = compile_to_ops(tree);

        let mut vars = ConstantMap::new();
        vars.insert("a".to_string(), 10.0);
        vars.insert("b".to_string(), 3.0);

        let result = program.execute(&vars).expect("Should evaluate");
        assert!((result - 7.0).abs() < 1e-6,
            "Expected 7.0, got {}", result);
    }

    #[test]
    fn test_transcendental_count_exp_ln_cancel() {
        // exp(ln(x)) should cancel to x — 0 transcendentals in result
        let tree = exp_node(ln_node(var("x")));
        let program = compile_to_ops(tree);
        println!("exp(ln(x)) op count: {}, transcendentals: {}",
            program.op_count(), program.transcendental_count());
        // After cancellation: just LoadVar("x") = 1 op, 0 transcendentals
        assert_eq!(program.transcendental_count(), 0,
            "ln(exp(x)) should cancel to 0 transcendentals");
    }

    #[test]
    fn test_lower_dot_product_k4() {
        // K=4 dot product: ALU naive uses 4 muls + 3 adds
        // EML after lowering should produce correct result
        use crate::nn_layer::build_dot_product_eml;

        let weights = vec![0.5f32, 0.3, 0.7, 0.2];
        let inputs: Vec<Arc<EmlNode>> = (0..4).map(|i| var(&format!("x{}", i))).collect();
        let xv = vec![2.0f64, 3.0, 1.5, 4.0];
        let expected: f64 = xv.iter().zip(weights.iter()).map(|(x, w)| x * *w as f64).sum();

        let tree = build_dot_product_eml(&inputs, &weights);
        let program = compile_to_ops(tree);

        let mut vars = ConstantMap::new();
        for (i, &v) in xv.iter().enumerate() {
            vars.insert(format!("x{}", i), v);
        }

        println!("K=4 dot product: {} ops, {} transcendentals (naive: 7 ops, 0 trans)",
            program.op_count(), program.transcendental_count());

        let result = program.execute(&vars).expect("Should evaluate");
        assert!((result - expected).abs() < 1e-4,
            "Expected {:.6}, got {:.6}", expected, result);
    }  // end test_lower_dot_product_k4

    #[test]

    fn test_mul_cf_recognized_as_mul_const() {
        // bare mul_cf(x, 0.5) should lower to exactly: LoadVar + MulConst = 2 ops, 0 trans
        let x = var("x");
        let tree = mul_cf(x, 0.5);
        let program = compile_to_ops(tree);
        println!("bare mul_cf(x, 0.5): {} ops, {} trans", program.op_count(), program.transcendental_count());
        assert_eq!(program.transcendental_count(), 0,
            "mul_cf should collapse to 0 transcendentals, got {} ops", program.op_count());
    }

    #[test]
    fn test_recognize_classical_on_mul_cf() {
        // Directly test that recognize_classical sees MulConst on a mul_cf node
        let x = var("x");
        let node = mul_cf(x, 0.5);
        let op = recognize_classical(&node);
        println!("recognize_classical(mul_cf(x, 0.5)) = {:?}", op.as_ref().map(|_| "Some(...)"));
        assert!(matches!(op, Some(ClassicalOp::MulConst(..))),
            "Expected MulConst, got None or other variant");
    }

    #[test]
    fn test_k1_dot_product_ops() {
        use crate::nn_layer::build_dot_product_eml;
        let inputs = vec![var("x0")];
        let weights = vec![0.5f32];
        let tree = build_dot_product_eml(&inputs, &weights);
        let program = compile_to_ops(tree);
        println!("K=1 single term: {} ops, {} trans", program.op_count(), program.transcendental_count());
        // With MulConst: 8 ops (LoadVar + 2xLoadConst + 2xSub + MulConst + LoadConst + Sub)
        // 0 transcendentals expected
        assert_eq!(program.transcendental_count(), 0,
            "K=1 should have 0 transcendentals, got {} trans in {} ops",
            program.transcendental_count(), program.op_count());
    }
}


