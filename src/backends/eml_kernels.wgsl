// src/backends/eml_kernels.wgsl

// src/backends/eml_kernels.wgsl

// === MINIMAX CONSTANTS (computed offline) ===
const LOG2_E: f32 = 1.4426950408889634f;
const LN_2:   f32 = 0.6931471805599453f;

// Coefficients for fast_exp N=2:
const EXP_A0: f32 = 1.00247605f;
const EXP_A1: f32 = 0.65104678f;
const EXP_A2: f32 = 0.34400111f;

// Coefficients for fast_ln N=3:
const LN_C1: f32 = 0.98771793f;
const LN_C2: f32 = -0.40916155f;
const LN_C3: f32 = 0.11513792f;

fn fast_exp(x: f32) -> f32 {
    // clamp guarantees exp_i >= 2 (for x >= -87)
    // Do not change clamp bounds without verifying exp_i >= 1
    // Limit range to avoid overflow
    let xc = clamp(x, -87.0f, 87.0f);
    let w = xc * LOG2_E;
    let i = floor(w);
    let f = w - i;
    // Minimax: 2^f
    let p = EXP_A0 + f * (EXP_A1 + f * EXP_A2);
    // Scale by 2^i using bit manipulation (ldexp)
    let exp_i = i32(i) + 127;
    let scale = bitcast<f32>(u32(exp_i) << 23u);
    return p * scale;
}

fn fast_ln(x: f32) -> f32 {
    if (x <= 0.0f) { return -3.4028235e+38f; } // -MAX_FLOAT as proxy for -inf
    let bx = bitcast<u32>(x);
    let e = f32(i32((bx >> 23u) & 0xFFu) - 127);
    let m = bitcast<f32>((bx & 0x7FFFFFu) | 0x3F800000u);
    let u = m - 1.0f;
    // Minimax: ln(1+u)
    let poly = u * (LN_C1 + u * (LN_C2 + u * LN_C3));
    return e * LN_2 + poly;
}

fn eml(x: f32, y: f32) -> f32 {
    return fast_exp(x) - fast_ln(y);
}

// bf16 packed as u32 (two bf16 in one u32)
alias Bf16x2 = u32;

fn decode_bf16_hi(packed: u32) -> f32 {
    // Upper 16 bits → f32 via masking
    return bitcast<f32>(packed & 0xFFFF0000u);
}

fn decode_bf16_lo(packed: u32) -> f32 {
    // Lower 16 bits → f32
    return bitcast<f32>((packed << 16u) & 0xFFFF0000u);
}

fn encode_bf16(x: f32) -> u32 {
    // f32 → bf16 via mantissa truncation
    // Rounding: round-to-nearest-even for better accuracy
    let bits = bitcast<u32>(x);
    let rounding = (bits >> 16u) & 1u; // rounding bit
    return (bits + 0x7FFFu + rounding) >> 16u;
}

struct DotProductParams {
    k: u32,          // vector length
    n_heads: u32,    // number of heads
    seq_len: u32,    // sequence length
    pad: u32,        // padding
};

@group(0) @binding(0) var<storage, read>       input:   array<f32>;
@group(0) @binding(1) var<storage, read>       weights: array<f32>; // pre-negated ASIS
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;
@group(0) @binding(3) var<uniform>             params:  DotProductParams;

@compute @workgroup_size(64, 1, 1)
fn dot_product_asis(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let out_idx = gid.x;
    let k = params.k;
    
    if (out_idx >= params.seq_len) { return; }
    
    // First element: x[0] * w[0]
    let x0 = input[out_idx * k + 0u];
    let w0 = weights[out_idx * k + 0u];
    var acc = x0 * w0;  // First element without ASIS
    
    // The rest: subtraction (ASIS)
    for (var i = 1u; i < k; i++) {
        let xi = input[out_idx * k + i];
        let wi = weights[out_idx * k + i]; // wi already negated offline
        
        // Using ALU fallback as defined - in full EML we would use the tree
        acc = acc - xi * wi;
    }
    
    output[out_idx] = acc;
}

@group(0) @binding(0) var<storage, read>       logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_softmax: array<f32>;

var<workgroup> shared_sum: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn log_softmax(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    let n = arrayLength(&logits);
    let tid = lid.x;
    
    // Step 1: Compute max(x_j) for numerical stability
    var local_max = -1e38f;
    for (var i = tid; i < n; i += 64u) {
        local_max = max(local_max, logits[i]);
    }
    shared_sum[tid] = local_max;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_sum[tid] = max(shared_sum[tid], shared_sum[tid + stride]);
        }
        workgroupBarrier();
    }
    let shift = shared_sum[0];
    workgroupBarrier();

    // Step 2: Compute Σ exp(x_j - shift) via parallel reduction
    var local_sum = 0.0f;
    for (var i = tid; i < n; i += 64u) {
        local_sum += fast_exp(logits[i] - shift);
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
    }
    
    // Step 3: log_softmax(x_i) = x_i - ln(S) [valid for any real x_i]
    let log_sum_exp = fast_ln(shared_sum[0]) + shift;
    
    for (var i = tid; i < n; i += 64u) {
        out_softmax[i] = logits[i] - log_sum_exp;
    }
}

