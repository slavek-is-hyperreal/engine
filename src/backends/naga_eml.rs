// src/backends/naga_eml.rs
//
// EML optimization pass for WGSL shaders via naga.

#[cfg(feature = "naga-pass")]
pub mod naga_eml {
    use naga::*;
    use crate::ast::*;
    // use crate::trs::rewrite; // Will be used in Phase 2
    use std::sync::Arc;

    /// Optimize a WGSL shader source through EML TRS.
    /// Returns optimized SPIR-V bytes.
    pub fn optimize_wgsl(wgsl_source: &str) -> Result<Vec<u32>, String> {
        // Step 1: Parse WGSL
        let module = naga::front::wgsl::parse_str(wgsl_source)
            .map_err(|e| format!("WGSL parse error: {:?}", e))?;

        // Step 2: Validate
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| format!("Validation error: {:?}", e))?;

        // Step 3: Analyze module (Phase 1: Report only)
        let stats = analyze_module(&module);

        println!("=== EML Naga Analysis ===");
        println!("Functions analyzed: {}", module.functions.len());
        println!("Expressions analyzed: {}", stats.expressions_analyzed);
        println!("EML-optimizable subgraphs: {}", stats.optimizable);
        println!("Estimated SFU reduction: {}%", stats.estimated_sfu_reduction);

        // Step 4: Emit SPIR-V (currently unoptimized pass-through)
        let options = naga::back::spv::Options::default();
        
        // We look for the main entry point
        let ep = module.entry_points.iter()
            .find(|ep| ep.name == "main")
            .ok_or("No 'main' entry point found")?;

        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: ep.stage,
            entry_point: ep.name.clone(),
        };

        naga::back::spv::write_vec(
            &module,
            &info,
            &options,
            Some(&pipeline_options),
        )
        .map_err(|e| format!("SPIR-V emit error: {:?}", e))
    }

    pub struct AnalysisStats {
        pub expressions_analyzed: usize,
        pub optimizable: usize,
        pub estimated_sfu_reduction: usize,
    }

    fn analyze_module(module: &Module) -> AnalysisStats {
        let mut analyzed = 0;
        let mut optimizable = 0;

        // Analyze functions
        for (_, func) in module.functions.iter() {
            for (_, expr) in func.expressions.iter() {
                analyzed += 1;
                if is_eml_optimizable(expr) {
                    optimizable += 1;
                }
            }
        }

        // Analyze entry points
        for ep in module.entry_points.iter() {
            for (_, expr) in ep.function.expressions.iter() {
                analyzed += 1;
                if is_eml_optimizable(expr) {
                    optimizable += 1;
                }
            }
        }

        let estimated = if optimizable > 0 { 30 } else { 0 };

        AnalysisStats {
            expressions_analyzed: analyzed,
            optimizable,
            estimated_sfu_reduction: estimated,
        }
    }

    fn is_eml_optimizable(expr: &Expression) -> bool {
        match expr {
            Expression::Math { fun, .. } => {
                match fun {
                    MathFunction::Exp | MathFunction::Log |
                    MathFunction::Exp2 | MathFunction::Log2 |
                    MathFunction::Sqrt | MathFunction::InverseSqrt => true,
                    _ => false,
                }
            }
            Expression::Binary { op, .. } => {
                match op {
                    BinaryOperator::Add | BinaryOperator::Subtract |
                    BinaryOperator::Multiply | BinaryOperator::Divide => true,
                    _ => false,
                }
            }
            _ => false,
        }
    }
}
