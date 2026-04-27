// src/backends/vulkan_eml.rs

use wgpu::{Device, Queue, ComputePipeline};
use bytemuck::{Pod, Zeroable};

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

impl EmlKernel {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("Nie znaleziono adaptera Vulkan (AMD R7 260X?)");
        
        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .await
            .expect("Nie można zainicjować urządzenia");
        
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
                label: Some("Dot Product ASIS EML"),
                layout: None,
                module: &shader,
                entry_point: "dot_product_asis",
                compilation_options: Default::default(),
            }
        );
        
        Self { device, queue, log_softmax_pipeline, dot_product_pipeline }
    }
    
    pub async fn run_log_softmax(&self, logits: &[f32]) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let n = logits.len();
        
        let input_buf = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("logits"),
                contents: bytemuck::cast_slice(logits),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }
        );
        
        let output_buf = self.device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("output"),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        
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
            &wgpu::CommandEncoderDescriptor { label: None }
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }
            );
            pass.set_pipeline(&self.log_softmax_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n as u32 + 63) / 64, 1, 1);
        }
        
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, (n * 4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));
        
        let slice = staging_buf.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();
        result
    }

    pub async fn run_dot_product_asis(&self, input: &[f32], weights: &[f32], k: u32, seq_len: u32) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        
        let input_buf = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("input"),
                contents: bytemuck::cast_slice(input),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }
        );

        let weights_buf = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("weights"),
                contents: bytemuck::cast_slice(weights),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }
        );
        
        let output_buf = self.device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("output"),
                size: (seq_len * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        );

        let params = DotProductParams {
            k,
            n_heads: 1,
            seq_len,
            pad: 0,
        };

        let params_buf = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
        
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
            &wgpu::CommandEncoderDescriptor { label: None }
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }
            );
            pass.set_pipeline(&self.dot_product_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((seq_len + 63) / 64, 1, 1);
        }
        
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (seq_len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, (seq_len * 4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));
        
        let slice = staging_buf.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();
        result
    }
}
