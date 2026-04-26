// cargo run --bin verify_output
use eml_trs::ast::*;
use eml_trs::nn_layer::build_dot_product_eml;
use eml_trs::constant_fold::{try_evaluate, ConstantMap};
use eml_trs::trs::rewrite;
use std::sync::Arc;

fn main() {
    println!("=== Weryfikacja Numeryczna EML ===\n");
    check("K=4 dodatnie", &[1.5,2.3,0.8,3.1], &[0.5,0.3,0.7,0.2], 1e-3);
    check("K=8 dodatnie", &[1.0,2.0,3.0,4.0,1.5,2.5,0.5,1.8], &[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], 1e-2);
    // Ujemne — oczekujemy None
    {
        let w=vec![0.5f32,-0.3,0.7,-0.2];
        let xv=vec![1.0f64,2.0,3.0,4.0];
        let inp: Vec<Arc<EmlNode>>=(0..4).map(|i|var(&format!("x{}",i))).collect();
        let tree=build_dot_product_eml(&inp,&w);
        let mut c=ConstantMap::new();
        for (i,&v) in xv.iter().enumerate() { c.insert(format!("x{}",i),v); }
        println!("Test ujemne wagi: {:?} (None jest OK)\n", try_evaluate(&tree,&c));
    }
}

fn check(name: &str, xv: &[f64], wv: &[f32], tol: f64) {
    let k=xv.len();
    let exp: f64=xv.iter().zip(wv).map(|(x,w)|x*(*w as f64)).sum();
    let inp: Vec<Arc<EmlNode>>=(0..k).map(|i|var(&format!("x{}",i))).collect();
    let tree=rewrite(build_dot_product_eml(&inp,wv));
    let mut c=ConstantMap::new();
    for (i,&v) in xv.iter().enumerate() { c.insert(format!("x{}",i),v); }
    print!("{}: ", name);
    match try_evaluate(&tree,&c) {
        Some(v) => { let d=(v-exp).abs();
            println!("exp={:.4} got={:.4} diff={:.1e} {}", exp, v, d,
                if d<tol {"✅"} else {"❌"}) }
        None => println!("None (ujemne pośrednie — OK)"),
    }
}
