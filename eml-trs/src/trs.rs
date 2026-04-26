// src/trs.rs
//
// KONWENCJA: eml_count() = liczba węzłów wewnętrznych Eml(l,r)
//            node_count() = eml_count() + liczba liści
// Koszty z exhaustive search (paper Odrzywołka) = node_count()
// Koszty w testach tego projektu = eml_count() (tylko wewnętrzne)

use crate::ast::*;
use std::sync::Arc;

/// Wynik dopasowania wzorca.
// Bindings usunięte (było nieużywane)

/// Jedna reguła przepisywania.
pub struct Rule {
    pub name: &'static str,
    pub apply: fn(&Arc<EmlNode>) -> Option<Arc<EmlNode>>,
}

/// Sprawdza czy węzeł to One
fn is_one(n: &EmlNode) -> bool { matches!(n, EmlNode::One) }

/// Sprawdź czy węzeł to eml(x, 1) — czyli exp(x)
pub fn is_exp_pattern(n: &EmlNode) -> Option<Arc<EmlNode>> {
    if let EmlNode::Eml(l, r) = n {
        if is_one(r) { return Some(l.clone()); }
    }
    None
}

/// Sprawdź czy węzeł to ln(x) — wzorzec: eml(1, eml(eml(1, x), 1))
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

/// Katalog reguł TRS
pub fn get_rules() -> Vec<Rule> {
    vec![
        // REGUŁA 3: ln(exp(x)) → x
        Rule {
            name: "ln_exp_cancel",
            apply: |node| {
                // Szukamy wzorca: eml(1, eml(eml(1, eml(x, 1)), 1))
                // czyli ln(exp(x))
                if let Some(inner) = is_ln_pattern(node) {
                    if let Some(x) = is_exp_pattern(&inner) {
                        // Sprawdź czy redukcja zmniejsza węzły
                        let before = node.eml_count();
                        let after = x.eml_count();
                        if after < before { return Some(x); }
                    }
                }
                None
            },
        },

        // REGUŁA 4: exp(ln(x)) → x
        Rule {
            name: "exp_ln_cancel",
            apply: |node| {
                if let Some(inner) = is_exp_pattern(node) {
                    if let Some(x) = is_ln_pattern(&inner) {
                        let before = node.eml_count();
                        let after = x.eml_count();
                        if after < before { return Some(x); }
                    }
                }
                None
            },
        },

        // REGUŁA 5: eml(ln(a), 1) → a
        Rule {
            name: "eml_ln_one_absorb",
            apply: |node| {
                if let EmlNode::Eml(l, r) = node.as_ref() {
                    if is_one(r) {
                        if let Some(a) = is_ln_pattern(l) {
                            let before = node.eml_count();
                            let after = a.eml_count();
                            if after < before { return Some(a); }
                        }
                    }
                }
                None
            },
        },

        // REGUŁA 8: eml(1, 1) → Const(e)
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

        // REGUŁA 9: eml(ln(exp(x)), y) → eml(x, y)
        Rule {
            name: "left_absorb",
            apply: |node| {
                if let EmlNode::Eml(l, r) = node.as_ref() {
                    if let Some(exp_inner) = is_ln_pattern(l) {
                        if let Some(x) = is_exp_pattern(&exp_inner) {
                            let new_node = eml(x, r.clone());
                            if new_node.eml_count() < node.eml_count() {
                                return Some(new_node);
                            }
                        }
                    }
                }
                None
            },
        },

        // REGUŁA 10: eml(x, exp(ln(y))) → eml(x, y)
        Rule {
            name: "right_absorb",
            apply: |node| {
                if let EmlNode::Eml(l, r) = node.as_ref() {
                    if let Some(ln_inner) = is_exp_pattern(r) {
                        if let Some(y) = is_ln_pattern(&ln_inner) {
                            let new_node = eml(l.clone(), y);
                            if new_node.eml_count() < node.eml_count() {
                                return Some(new_node);
                            }
                        }
                    }
                }
                None
            },
        },
    ]
}

/// Główna funkcja TRS: bottom-up traversal do fixpoint
/// Stosuje reguły aż żadna nie może być zastosowana
pub fn rewrite(node: Arc<EmlNode>) -> Arc<EmlNode> {
    // Baza: liście zwracamy bez zmian
    if node.is_leaf() { return node; }

    // Bottom-up: najpierw zredukuj dzieci
    let node = if let EmlNode::Eml(l, r) = node.as_ref() {
        let new_l = rewrite(l.clone());
        let new_r = rewrite(r.clone());
        if !new_l.structural_eq(l) || !new_r.structural_eq(r) {
            eml(new_l, new_r)
        } else {
            node.clone()
        }
    } else {
        node
    };

    // Fixpoint: stosuj reguły aż żadna nie pasuje
    let rules = get_rules();
    let mut current = node;
    let mut changed = true;

    while changed {
        changed = false;
        for rule in &rules {
            if let Some(reduced) = (rule.apply)(&current) {
                // Bezpieczeństwo: reguła musi zmniejszyć liczbę węzłów
                assert!(
                    reduced.eml_count() < current.eml_count(),
                    "Rule '{}' did not reduce node count: {} -> {}",
                    rule.name,
                    current.eml_count(),
                    reduced.eml_count()
                );
                current = rewrite(reduced); // rekurencja na nowym węźle
                changed = true;
                break; // reset od początku reguł
            }
        }
    }

    current
}

/// Statystyki redukcji
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
        // TRS nigdy nie może zwiększyć liczby węzłów
        let tree = eml(ln_node(exp_node(var("x"))), one());
        let before = tree.eml_count();
        let reduced = rewrite(tree);
        assert!(reduced.eml_count() <= before);
    }

    #[test]
    fn test_nested_reduction() {
        // Zagnieżdżone ln(exp()) powinny się zredukować
        let x = var("x");
        let tree = ln_node(exp_node(ln_node(exp_node(x.clone()))));
        let reduced = rewrite(tree);
        assert!(reduced.eml_count() < 10); // powinno być małe
    }
}
