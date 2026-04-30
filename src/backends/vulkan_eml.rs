// src/backends/vulkan_eml.rs
//
// Vulkan/wgpu backend dla EML — iloczyn skalarny ASIS + Log-Softmax.
//
// Architektura (PAPER.md §9.2, §10.1):
//   - Dense dot product: zwykłe ALU (najszybsze dla matmul)
//   - Nieliniowości (log-softmax, SiLU): fast_exp + fast_ln Minimax N=2/3
//
// Minimax coefficients: SET A z eml_kernels.wgsl (verify_minimax.py potwierdza):
//   fast_exp: E_max = 0.0024  (<< BF16 eps 0.0078)
//   fast_ln:  E_max = 0.0005  (<< BF16 eps 0.0078)

use wgpu::{Device, Queue, ComputePipeline};
use bytemuck::{Pod, Zeroable};

// ============================================================
// CPU-side Minimax (dokładna kopia logiki z eml_kernels.wgsl)
// Używana wyłącznie w testach parytetu.
// ============================================================

const LOG2_E: f32 = 1.4426950408889634;
const LN_2: f32   = 0.6931471805599453;

// SET A — eml_kernels.wgsl (wybrany przez scripts/verify_minimax.py)
const EXP_A0: f32 = 1.00247605;
const EXP_A1: f32 = 0.65104678;
const EXP_A2: f32 = 0.34400111;
const LN_C1: f32  = 0.98771793;
const LN_C2: f32  = -0.40916155;
const LN_C3: f32  = 0.11513792;

/// fast_exp(x) — N=2 Minimax, E_max ≈ 0.0024 < BF16 eps 0.0078
/// Emuluje dokładnie logikę z eml_kernels.wgsl (bit-level).
pub fn cpu_fast_exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 87.0);
    let w = x * LOG2_E;
    let i = w.floor();
    let f = w - i;
    let p = EXP_A0 + f * (EXP_A1 + f * EXP_A2);
    // 2^i przez bit manipulation (identyczne z WGSL bitcast)
    let exp_i = (i as i32) + 127;
    let scale = f32::from_bits((exp_i as u32) << 23);
    p * scale
}

/// fast_ln(x) — N=3 Minimax, E_max ≈ 0.0005 < BF16 eps 0.0078
/// Emuluje dokładnie logikę z eml_kernels.wgsl (bit-level).
pub fn cpu_fast_ln(x: f32) -> f32 {
    if x <= 0.0 { return f32::NEG_INFINITY; }
    let bx = x.to_bits();
    let e = (((bx >> 23) & 0xFF) as i32) - 127;
    let m = f32::from_bits((bx & 0x7FFFFF) | 0x3F800000);
    let u = m - 1.0;
    let poly = u * (LN_C1 + u * (LN_C2 + u * LN_C3));
    (e as f32) * LN_2 + poly
}

/// eml(x, y) = exp(x) - ln(y) — CPU reference
pub fn cpu_eml(x: f32, y: f32) -> f32 {
    cpu_fast_exp(x) - cpu_fast_ln(y)
}

/// log_softmax(logits) — CPU reference używana w testach parytetu
pub fn cpu_log_softmax(logits: &[f32]) -> Vec<f32> {
    let sum_exp: f32 = logits.iter().map(|&x| cpu_fast_exp(x)).sum();
    let log_sum = cpu_fast_ln(sum_exp);
    logits.iter().map(|&x| x - log_sum).collect()
}

// ============================================================
// GPU Structs
// ============================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DotProductParams {
    pub k: u32,
    pub n_heads: u32,
    pub seq_len: u32,
    pub pad: u32,
}

pub struct EmlKernel {
    device: Device,
    queue: Queue,
    log_softmax_pipeline: ComputePipeline,
    dot_product_pipeline: ComputePipeline,
}

// ============================================================
// GPU Implementation
// ============================================================

impl EmlKernel {
    /// Inicjalizuje GPU backend. Wymaga Vulkan/Metal/DX12 adaptera.
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("No GPU adapter found")?;

        let info = adapter.get_info();
        eprintln!("[EmlKernel] GPU: {} ({:?})", info.name, info.backend);

        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .await
            .map_err(|e| format!("Device init failed: {}", e))?;

        // Kompilujemy jeden shader zawierający oba kernele
        let shader_src = include_str!("eml_kernels.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("EML Kernels"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let log_softmax_pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Log-Softmax EML"),
                layout: None,
                module: &shader,
                entry_point: "log_softmax",
                compilation_options: Default::default(),
            }
        );

        let dot_product_pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Dot Product ASIS"),
                layout: None,
                module: &shader,
                entry_point: "dot_product_asis",
                compilation_options: Default::default(),
            }
        );

        Ok(Self { device, queue, log_softmax_pipeline, dot_product_pipeline })
    }

    // --------------------------------------------------------
    // Helper: Odczyt bufora GPU → Vec<f32>
    // --------------------------------------------------------
    async fn read_buffer(&self, src: &wgpu::Buffer, encoder: wgpu::CommandEncoder, size: u64) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = encoder;
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    // --------------------------------------------------------
    // Log-Softmax GPU
    // --------------------------------------------------------

    /// Liczy log-softmax na GPU, zwraca Vec<f32>.
    pub async fn run_log_softmax(&self, logits: &[f32]) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let n = logits.len();
        let byte_size = (n * 4) as u64;

        let input_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("logits"),
            contents: bytemuck::cast_slice(logits),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.log_softmax_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("log_softmax") }
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }
            );
            pass.set_pipeline(&self.log_softmax_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n as u32 + 63) / 64, 1, 1);
        }

        self.read_buffer(&output_buf, encoder, byte_size).await
    }

    // --------------------------------------------------------
    // Dot Product ASIS GPU
    // --------------------------------------------------------

    /// Liczy iloczyn skalarny ASIS dla `seq_len` wektorów po `k` elementów.
    /// `weights` powinny być pre-negowane (ASIS offline).
    pub async fn run_dot_product_asis(
        &self,
        input: &[f32],
        weights: &[f32],
        k: u32,
        seq_len: u32,
    ) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let out_byte_size = (seq_len * 4) as u64;

        let input_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let weights_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("weights"),
            contents: bytemuck::cast_slice(weights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: out_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = DotProductParams { k, n_heads: 1, seq_len, pad: 0 };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group_layout = self.dot_product_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("dot_product") }
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }
            );
            pass.set_pipeline(&self.dot_product_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((seq_len + 63) / 64, 1, 1);
        }

        self.read_buffer(&output_buf, encoder, out_byte_size).await
    }
}

// ============================================================
// Testy parytetu GPU ↔ CPU
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerancja: 2 * E_max(fast_exp) + E_max(fast_ln) = 2*0.0024 + 0.0005 ≈ 0.006
    // Bezpiecznie zostawiamy 0.01 (> BF16 eps 0.0078, ale GPU float32 jest precyzyjniejszy)
    const TOL: f32 = 0.01;

    // ---- Testy CPU (deterministyczne, zawsze działają) ----

    #[test]
    fn test_cpu_fast_exp_accuracy() {
        // Weryfikacja E_max < BF16 eps dla zakresu [-5, 5]
        let mut max_err: f32 = 0.0;
        let n = 10_000;
        for i in 0..n {
            let x = -5.0 + 10.0 * (i as f32) / (n as f32);
            let true_val = x.exp();
            let approx = cpu_fast_exp(x);
            let rel_err = (true_val - approx).abs() / true_val.abs().max(1e-10);
            max_err = max_err.max(rel_err);
        }
        assert!(
            max_err < 0.0078,
            "cpu_fast_exp E_max = {:.6} >= BF16 eps 0.0078",
            max_err
        );
    }

    #[test]
    fn test_cpu_fast_ln_accuracy() {
        // Weryfikacja E_max < BF16 eps dla zakresu [0.01, 20.0]
        let mut max_err: f32 = 0.0;
        let n = 10_000;
        for i in 1..=n {
            let x = 0.01 + 19.99 * (i as f32) / (n as f32);
            let true_val = x.ln();
            let approx = cpu_fast_ln(x);
            let abs_err = (true_val - approx).abs();
            max_err = max_err.max(abs_err);
        }
        assert!(
            max_err < 0.0078,
            "cpu_fast_ln E_max = {:.6} >= BF16 eps 0.0078",
            max_err
        );
    }

    #[test]
    fn test_cpu_log_softmax_sums_to_zero() {
        // log-softmax poprawny ↔ log(Σ exp(log_softmax_i)) = 0
        let logits = vec![1.0f32, 2.0, 0.5, -1.0, 3.0];
        let ls = cpu_log_softmax(&logits);
        let sum_exp: f32 = ls.iter().map(|&x| cpu_fast_exp(x)).sum();
        assert!(
            (sum_exp - 1.0).abs() < TOL,
            "Σ exp(log_softmax) = {} (expected 1.0)",
            sum_exp
        );
    }

    #[test]
    fn test_cpu_log_softmax_parity_with_naive() {
        // Porówanie z referencją (stdlib exp/ln)
        let logits = vec![0.5f32, 1.5, -0.5, 2.0, 0.0];
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits.iter().map(|&x| (x - max_l).exp()).sum();
        let log_sum = sum_exp.ln() + max_l;
        let naive: Vec<f32> = logits.iter().map(|&x| x - log_sum).collect();

        let fast = cpu_log_softmax(&logits);
        for (a, b) in naive.iter().zip(fast.iter()) {
            assert!(
                (a - b).abs() < TOL,
                "log_softmax parity: naive={:.5} fast={:.5} diff={:.5}",
                a, b, (a-b).abs()
            );
        }
    }

    // ---- Testy GPU (wymagają karty graficznej) ----
    // Oznaczone jako #[ignore] — uruchom przez: cargo test -- --ignored

    #[tokio::test]
    #[ignore = "requires GPU / Vulkan adapter"]
    async fn test_gpu_log_softmax_parity() {
        let kernel = EmlKernel::new().await
            .expect("GPU unavailable — run with WGPU_BACKEND=vulkan or skip this test");

        let logits = vec![1.0f32, 2.0, 0.5, -1.0, 3.0, 0.0, -0.5, 1.5];
        let cpu_result = cpu_log_softmax(&logits);
        let gpu_result = kernel.run_log_softmax(&logits).await;

        assert_eq!(cpu_result.len(), gpu_result.len());
        for (i, (c, g)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            assert!(
                (c - g).abs() < TOL,
                "log_softmax[{}]: cpu={:.5} gpu={:.5} diff={:.5}",
                i, c, g, (c - g).abs()
            );
        }
        println!("GPU log_softmax parity OK (tol={TOL})");
    }

    #[tokio::test]
    #[ignore = "requires GPU / Vulkan adapter"]
    async fn test_gpu_dot_product_parity() {
        let kernel = EmlKernel::new().await
            .expect("GPU unavailable");

        // K=8, seq_len=4
        let k: u32 = 8;
        let seq_len: u32 = 4;

        // Wagi dla JEDNEGO wektora wyjściowego
        let weights_row: Vec<f32> = vec![0.5, -0.3, 0.7, -0.2, 1.0, -0.1, 0.4, -0.6];

        // Inputs: seq_len wektorów po K elementów (płaski bufor row-major)
        let inputs: Vec<f32> = (0..seq_len * k)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        // Shader indeksuje wagi jako weights[out_idx * k + i] —
        // czyli oczekuje seq_len*K wag (jedna kopia na każdy wiersz wyjściowy).
        // Powielamy te same wagi dla uproszczenia (wszystkie wiersze używają tych samych W).
        let weights_flat: Vec<f32> = weights_row.iter()
            .cloned()
            .cycle()
            .take((seq_len * k) as usize)
            .collect();

        // ASIS pre-negation offline:
        // Shader liczy: acc = x[0]*w[0] - Σ(i>0) x[i]*w[i]
        // Żeby uzyskać prawdziwy dot product x·w, pre-negujemy w[1..]:
        //   acc = x[0]*w[0] - Σ x[i]*(-w[i]) = x[0]*w[0] + Σ x[i]*w[i] = x·w
        let mut asis_weights = weights_flat.clone();
        for chunk in asis_weights.chunks_mut(k as usize) {
            for w in chunk.iter_mut().skip(1) {
                *w = -*w;
            }
        }

        // CPU reference: zwykły dot product (= co shader powinien obliczyć)
        let cpu_results: Vec<f32> = (0..seq_len as usize)
            .map(|s| {
                inputs[s * k as usize..(s + 1) * k as usize]
                    .iter()
                    .zip(weights_row.iter())
                    .map(|(x, w)| x * w)
                    .sum()
            })
            .collect();

        let gpu_results = kernel.run_dot_product_asis(&inputs, &asis_weights, k, seq_len).await;

        assert_eq!(cpu_results.len(), gpu_results.len(), "output length mismatch");
        for (i, (c, g)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            assert!(
                (c - g).abs() < TOL,
                "dot_product[{}]: cpu={:.5} gpu={:.5} diff={:.5}",
                i, c, g, (c - g).abs()
            );
        }
        println!("GPU dot_product parity OK (tol={TOL})");
    }
}

