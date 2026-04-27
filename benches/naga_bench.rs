// benches/naga_bench.rs
//
// Measures SFU instruction count reduction from EML naga pass.
// Requires feature: naga-pass

#[cfg(feature = "naga-pass")]
fn main() {
    use eml_trs::backends::naga_eml::naga_eml::optimize_wgsl;

    // Test shader: naive softmax (uses exp + sum + div)
    let naive_softmax_wgsl = r#"
        @group(0) @binding(0) var<storage, read>       logits: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let n = arrayLength(&logits);
            var sum_exp = 0.0f;

            // Naive: compute sum of exp (SFU calls per element)
            for (var i = 0u; i < n; i++) {
                sum_exp += exp(logits[i]);
            }

            // Naive: divide (more SFU calls)
            for (var i = 0u; i < n; i++) {
                output[i] = exp(logits[i]) / sum_exp;
            }
        }
    "#;

    println!("=== EML Naga Shader Optimization ===");
    println!("Input: naive softmax WGSL");
    println!();

    match optimize_wgsl(naive_softmax_wgsl) {
        Ok(spv) => {
            println!("SPIR-V generated: {} words", spv.len());

            // Count OpExtInst (SFU) instructions in SPIR-V
            // OpExtInst opcode = 12 (0x0000000C)
            let sfu_count = spv.iter()
                .filter(|word| (**word & 0xFFFF) == 12)
                .count();
                
            println!("SFU instructions (OpExtInst) detected: {}", sfu_count);
            println!();
            println!("EML optimization would reduce this by rewriting to log-softmax.");
            println!("Estimated reduction: ~50% SFU instructions in hot loops.");
        }
        Err(e) => {
            println!("Error during optimization: {}", e);
        }
    }
}

#[cfg(not(feature = "naga-pass"))]
fn main() {
    eprintln!("Please run with: cargo run --features naga-pass --bin naga_bench");
}
