# OxiEML Engine Code Index

## [lib.rs](file:///my_data/engine/src/lib.rs)
- **Role**: Root crate module. Defines the module hierarchy and public API.
- **Submodules**:
  - `ast`: Core EML node types and constructors.
  - `cost_model`: Analysis of computational cost (naive vs EML).
  - `trs`: Term Rewriting System for algebraic optimization.
  - `constant_fold`: Constant folding for EML nodes.
  - `asis`: Structural identity synthesis (Legacy/Alternative).
  - `dag`: In-memory DAG representation and deduplication.
  - `backends`: Execution backends (ALU, WGSL, Naga).
  - `fusions`: Higher-level operation fusions (Softmax, RMSNorm).
  - `round_trip`: Lowering EML back to classical operations.
  - `polar`: Polar coordinates/Rotation experiments.
  - `nn_layer`: Neural network layer builders (Linear, SwiGLU).
  - `tslp`: Tree Straight-Line Programs (Grammar compression).
  - `compress`: Serialization and model compression tools.
  - `dag_mmap`: Out-of-core SSD-mapped DAG engine.
  - `loader`: Weight loaders (GGUF).

## [ast.rs](file:///my_data/engine/src/ast.rs)
- **Role**: Defines the core EML Expression Tree.
- **Key Types**:
  - `EmlNode` (Enum): `One`, `Var(String)`, `Const(f64)`, `Eml(Arc, Arc)`.
- **Key Methods**:
  - `node_count()`: DAG-aware total node count.
  - `eml_count()`: DAG-aware internal node count.
  - `depth()`: Memoized depth calculation.
  - `structural_eq()`: Recursive structural equality with pointer-eq shortcut.
  - `evaluate()`: Memoized numerical evaluation using `exp(x) - ln(y)`.
- **Constructors & Macros**:
  - `one()`, `var()`, `konst()`, `eml()`: Basic building blocks.
  - `exp_node(x)`: `eml(x, 1)`.
  - `ln_node(x)`: 7-node pattern for natural log.
  - `mul_eml(x, y)`: Multiplication via log-space.
  - `add_eml()`, `sub_eml()`: Addition and subtraction.
  - `neg_node(x)`: Negation using `ln(0)` trick.
  - `mul_cf(x, w)`: Optimized Constant-Weight multiplication (5 nodes).

## [trs.rs](file:///my_data/engine/src/trs.rs)
- **Role**: Term Rewriting System for EML algebraic simplification.
- **Key Types**:
  - `Rule`: Named rewriting rule with an `apply` function.
- **Key Functions**:
  - `rewrite(node)`: Main entry point. Applies rules bottom-up to fixpoint.
  - `rewrite_internal(node, cache)`: DAG-aware recursive rewriter with memoization.
  - `is_exp_pattern(node)`, `is_ln_pattern(node)`: High-level pattern recognizers.
  - `get_rules()`: Catalog of implemented EML rules:
    - `ln_exp_cancel`: `ln(exp(x)) -> x`
    - `exp_ln_cancel`: `exp(ln(x)) -> x`
    - `eml_ln_one_absorb`: `eml(ln(a), 1) -> a`
    - `constant_e`: `eml(1, 1) -> Const(e)`
    - `left_absorb`, `right_absorb`: Identities for simplifying `eml(ln(exp(x)), y)` etc.
  - `rewrite_with_stats(node)`: Rewrites and returns node count reduction metrics.

## [round_trip.rs](file:///my_data/engine/src/round_trip.rs)
- **Role**: Lowering EML trees back to classical linear algebra operations.
- **Key Types**:
  - `FlatOp` (Enum): `LoadVar`, `LoadConst`, `Mul`, `Add`, `Sub`, `Div`, `Exp`, `Ln`, `MulConst`.
  - `FlatProgram`: Sequence of `FlatOp` with a `result_slot`.
  - `ClassicalOp` (Enum): Intermediate representation for recognized patterns.
- **Key Functions**:
  - `recognize_classical(node)`: Patterns EML nodes into `ClassicalOp` (e.g. recognizing `sub_eml` as `Sub`).
  - `lower_to_flat_ops(tree)`: Recursively converts an EML tree to a `FlatProgram`.
  - `round_trip_optimize(node)`: Applies classical identities (e.g. `ln(exp(x)) -> x`) across EML boundaries.
  - `compile_to_ops(node)`: High-level entry point. Tries raw lowering, fallbacks to RT optimization.
- **Execution**:
  - `FlatProgram::execute(vars)`: Interprets the flat ops on a `HashMap` of variable values.

## [dag_mmap.rs](file:///my_data/engine/src/dag_mmap.rs)
- **Role**: SSD-backed out-of-core DAG engine for massive model audits.
- **Key Types**:
  - `CompactNode` (Struct, 24 bytes): `tag`, `left`, `right`, `hash`, `ref_count`.
  - `MmapDag` (Struct): Manages `MmapMut`, capacity, and a RAM-based `hash_index`.
- **Key Functions**:
  - `MmapDag::create(path, capacity)`: Initializes or opens a memory-mapped file on disk (e.g. ZFS).
  - `MmapDag::add_eml_node(l, r, hash)`: Global deduplication using the `hash_index`.
  - `MmapDag::sharing_savings()`: Calculates global node reduction across all integrated trees.
  - `add_tree_to_mmap_dag(dag, node, local_cache)`: Adapter for integrating `Arc<EmlNode>` trees into the persistent DAG.
- **Note**: Pre-allocates space to avoid remapping overhead. Uses 24-byte aligned structs for high-performance IO.

## [nn_layer.rs](file:///my_data/engine/src/nn_layer.rs)
- **Role**: High-level builders for Neural Network components in EML.
- **Key Functions**:
  - `build_dot_product_eml(input, weights)`: Balanced-tree dot product implementation.
    - **Stability**: Uses the **BIAS Trick** `(x + BIAS) * |w| - (BIAS * |w|)` to handle signed weights and avoid `ln(x)` domains issues.
  - `preprocess_wq_offline(w_q, gamma, d_k, hidden)`: Absorbs layer normalization and scaling into weights at zero runtime cost.
  - `build_and_optimize_sample(weights, sample_k)`: Full pipeline test (Build -> Optimize -> Statistics).
- **Architecture**:
  - Employs a balanced binary tree sum to keep depth O(log K), critical for numerical stability and parallel execution.
  - Signed arithmetic is handled by partitioning terms into `pos_sum` and `neg_sum` and subtracting.

## [fusions.rs](file:///my_data/engine/src/fusions.rs)
- **Role**: Specialized operation fusions for Transformer layers.
- **Key Functions**:
  - `swiglu_fused(gate, up)`: Fuses SiLU(gate) * up into a single EML structure.
    - **Note**: Requires `gate * up > 1.0` due to `ln(ln())` domain constraints.
  - `residual_fused(ln_x, neg_out)`: Fuses addition of a residual connection when the operand is already in `ln(x)` form.
  - `rmsnorm_gamma_fold(gamma, w_q)`: Offline folding of RMSNorm scaling into weights.
  - `scale_weight_fold(w_q, d_k)`: Offline folding of attention scaling `1/sqrt(dk)` into weights.
- **Objective**: Minimize runtime node count and avoid redundant `exp/ln` pairs across operation boundaries.

## [tslp/](file:///my_data/engine/src/tslp/) (Tree Straight-Line Programs)
- **Role**: Theoretical framework for parallel evaluation and grammar compression.
- **Components**:
  - `grammar.rs`: Converts EML DAGs into formal TSLP grammars.
    - `TslpGrammar`: Collection of productions (`id -> eml(l,r)` or `id -> leaf`).
    - `extract_grammar(root)`: Uses pointer identity to find shared non-terminals.
  - `depth.rs`: Assigns parallel evaluation levels to DAG nodes.
  - `scheduler.rs`: Builds "waves" of nodes for parallel GPU/Multi-core execution.
  - `rake_compress.rs`: Implements the "Rake" transformation to reduce tree depth while preserving algebraic structure.
  - `ganardi.rs`: Implementation of grammar-balancing algorithms (O(g) time).

## [backends/](file:///my_data/engine/src/backends/)
- **Role**: Execution environments for EML programs.
- **Components**:
  - `alu.rs`: Reference CPU implementation. Defines `BackendChoice` (EML vs ALU) and cycle costs.
  - `wgsl.rs`: Shader generator for WebGPU/Vulkan.
    - `generate_eml_kernel()`: Implements `fast_exp` (N=2) and `fast_ln` (N=3) using FMA.
    - `generate_log_softmax_kernel()`: Native EML Log-Softmax implementation (`eml(ln(x_i), S)`).
  - `naga_eml.rs`: Integration with the Naga shader compiler for validation.
  - `vulkan_eml.rs`: Low-level Vulkan compute pipeline for EML execution.

## [compress/](file:///my_data/engine/src/compress/)
- **Role**: Model persistence and binary serialization.
- **Key Functions**:
  - `serialize_grammar(grammar, writer)`: Saves TSLP grammars to binary `EMLT` format.
  - `deserialize_grammar(reader)`: Restores `TslpGrammar` from disk.
  - `rebuild_tree(grammar)`: Reconstructs an `Arc<EmlNode>` DAG from a TSLP grammar.
- **Format**:
  - `EMLT` (v1): Header + Metadata (N productions, start index) + Compact productions (u8 tag + data).

## [constant_fold.rs](file:///my_data/engine/src/constant_fold.rs)
- **Role**: Constant folding and evaluation logic.
- **Key Functions**:
  - `try_evaluate(node, consts)`: Full numerical evaluation of constant subtrees.
  - `fold_constants(node, consts)`: Recursive transformation to collapse constant nodes into `EmlNode::Const`.
  - `mul_with_const_weight(x, w)`: Specifically optimizes constant multiplication at the structural level.
  - `asis_preprocess_weights(weights)`: Utility for ASIS-compatible weight normalization.

## [asis.rs](file:///my_data/engine/src/asis.rs)
- **Role**: Algebraic Structural Identity Synthesis (ASIS) implementation.
- **Key Functions**:
  - `build_asis_dot_product(inputs, weights)`: Implements dot product via accumulation by subtraction (`A1B1 - (-A2)B2 - ...`).
  - `verify_asis_correctness(inputs, weights)`: Numerical validation against naive sum.
- **Theory**: ASIS exploits structural symmetry to reduce internal EML node count by pre-negating weights offline.

## [bin/](file:///my_data/engine/src/bin/) (Applications & Utilities)
- **Audits**:
  - `full_layer_unified_v3.rs`: Current production audit binary for TinyLlama. Uses `MmapDag` on ZFS.
  - `full_layer_unified.rs`: V1 audit (deprecated, RAM-intensive).
- **Verification**:
  - `emlm_parity.rs`: Numerical parity test between EML execution and ALU matmul.
  - `full_layer_parity.rs`: Parity test for entire Transformer layers.
  - `math_verification.rs`: Basic EML algebraic identity tests.
- **Benchmarks**:
  - `eml_benchmark.rs`: Microbenchmarks for single EML operations.
  - `layer_benchmark.rs`: Latency measurements for full layers.
- **Utilities**:
  - `eml_compress.rs`: Tool for generating `.emlm` compressed model files.
  - `test_mmap_dag.rs`: Unit tests for the out-of-core DAG engine.

## [loader/gguf.rs](file:///my_data/engine/src/loader/gguf.rs)
- **Role**: Minimal GGUF format loader.
- **Key Types**:
  - `GgufLoader`: File handle and tensor metadata management.
  - `TensorInfo`: Shape, offset, and dtype (`F32`, `F16`) information.
- **Key Functions**:
  - `load_f32(name)`: Reads tensor and converts/unpacks to `f32`.
  - `load_layer(idx)`: Helper for bulk-loading all weights for a Transformer layer.

## [polar.rs](file:///my_data/engine/src/polar.rs)
- **Role**: Research into Polar Coordinate Representation for RoPE.
- **Theory**: Polar representation `(ln(r), φ)` makes RoPE a 0-node addition.
- **Key Types**:
  - `PolarVector`: Holds `ln_r` and `phi`.
- **Key Functions**:
  - `apply_rope(angle)`: Additive rotation.
  - `dot(other)`: Polar-space dot product.

## [cost_model.rs](file:///my_data/engine/src/cost_model.rs)
- **Role**: Theoretical performance and complexity analyzer.
- **Constants**:
  - `exp=3`, `ln=7`, `sub=11`, `mul=17`, `add=19`.
- **Key Functions**:
  - `tinyllama_layer_reduction()`: Calculates ~61% node reduction for TinyLlama.
  - `log_softmax_dag(n)`: Shows Log-Softmax is native to EML.

## [dag.rs](file:///my_data/engine/src/dag.rs)
- **Role**: In-memory DAG representation.
- **Key Types**:
  - `EmlDag`: Structural hash index for RAM-based deduplication.
- **Key Functions**:
  - `structural_hash(node)`: Cached structural hashing.
  - `add_node(node)`: Deduplicates nodes using hash.

## [tslp/depth.rs](file:///my_data/engine/src/tslp/depth.rs)
- **Key Functions**:
  - `assign_depths(root)`: Assigns parallel evaluation levels to nodes.

## [tslp/scheduler.rs](file:///my_data/engine/src/tslp/scheduler.rs)
- **Key Types**:
  - `TslpSchedule`: Groups nodes by depth for parallel waves.

## [tslp/executor.rs](file:///my_data/engine/src/tslp/executor.rs)
- **Key Functions**:
  - `simulate_execution(schedule)`: Dry-run for parallel schedule performance.

## [tslp/rake_compress.rs](file:///my_data/engine/src/tslp/rake_compress.rs)
- **Key Functions**:
  - `rake_compress(node)`: Heuristic depth reduction via "Rake" and "Compress" phases.

## [tslp/parallel_prefix.rs](file:///my_data/engine/src/tslp/parallel_prefix.rs)
- **Key Functions**:
  - `parallel_prefix_sum(terms)`: Balanced binary tree summation.
  - `build_balanced_dot_product(inputs, weights)`: O(log K) depth dot product.

## [tslp/ganardi.rs](file:///my_data/engine/src/tslp/ganardi.rs)
- **Role**: TSLP grammar balancing (O(g log g)).
- **Key Functions**:
  - `balance_grammar(grammar)`: Guaranteed O(log N) depth balancing.

## [compress/lift.rs](file:///my_data/engine/src/compress/lift.rs)
- **Key Functions**:
  - `lift_bytes(data)`: Converts raw byte data into EML structures for grammar lifting.

## [compress/json_loader.rs](file:///my_data/engine/src/compress/json_loader.rs)
- **Role**: Helper for loading weights from JSON format during audit/compression.

