use std::sync::{OnceLock, Mutex};
use wgpu::{Device, Queue};

pub struct CachedBuffer {
    pub size: wgpu::BufferAddress,
    pub usage: wgpu::BufferUsages,
    pub buffer: wgpu::Buffer,
}

pub struct WgpuBackend {
    pub device: Device,
    pub queue: Queue,
    pub buffer_cache: Mutex<Vec<CachedBuffer>>,
}

pub static BACKEND: OnceLock<WgpuBackend> = OnceLock::new();

pub fn init_backend() {
    BACKEND.get_or_init(|| {
        // We use pollster to block synchronously since PyO3 loads us synchronously
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("Failed to find an appropriate wgpu adapter (e.g., Vulkan)");

        let info = adapter.get_info();
        println!("[vulkannn_rusted] Found WGPU Adapter: {} ({:?})", info.name, info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("VulkanNN Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create wgpu device");

        WgpuBackend {
            device,
            queue,
            buffer_cache: Mutex::new(Vec::new()),
        }
    });
}

// ---------------------------------------------------------
// BUFFER POOLING (Ping-Pong Buffers)
// ---------------------------------------------------------
pub fn get_buffer(device: &Device, size: wgpu::BufferAddress, usage: wgpu::BufferUsages, label: Option<&str>) -> wgpu::Buffer {
    let backend = BACKEND.get().unwrap();
    if let Ok(mut cache) = backend.buffer_cache.lock() {
        if let Some(idx) = cache.iter().position(|b| b.size >= size && b.usage == usage) {
            let cached = cache.swap_remove(idx);
            return cached.buffer;
        }
    }
    
    device.create_buffer(&wgpu::BufferDescriptor {
        label,
        size,
        usage,
        mapped_at_creation: false,
    })
}

pub fn recycle_buffer(buffer: wgpu::Buffer, size: wgpu::BufferAddress, usage: wgpu::BufferUsages) {
    if let Ok(mut cache) = BACKEND.get().unwrap().buffer_cache.lock() {
        cache.push(CachedBuffer { size, usage, buffer });
    }
}

pub fn execute_add(a_data: &[f32], b_data: &[f32], is_hybrid: bool) -> Vec<f32> {
    let backend = BACKEND.get().expect("Backend not initialized. Did you import the python module properly?");
    let device = &backend.device;
    let queue = &backend.queue;

    let total_elements = a_data.len();
    let mut result = vec![0.0; total_elements];

    // Determine the split ratio
    // If not hybrid, 100% goes to GPU. If hybrid, let's say 70% GPU, 30% CPU.
    let gpu_ratio: f64 = if is_hybrid { 0.7 } else { 1.0 };
    let gpu_elements = (total_elements as f64 * gpu_ratio) as usize;
    let cpu_elements = total_elements - gpu_elements;

    // Use a scoped thread to run Rayon and WGPU simultaneously
    std::thread::scope(|s| {
        let (a_gpu, a_cpu) = a_data.split_at(gpu_elements);
        let (b_gpu, b_cpu) = b_data.split_at(gpu_elements);
        let (res_gpu, res_cpu) = result.split_at_mut(gpu_elements);

        // CPU Thread
        if cpu_elements > 0 {
            s.spawn(move || {
                use rayon::prelude::*;
                res_cpu.par_iter_mut()
                       .zip(a_cpu.par_iter())
                       .zip(b_cpu.par_iter())
                       .for_each(|((c, &a), &b)| *c = a + b);
            });
        }

        // GPU Thread (Main thread)
        if gpu_elements > 0 {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Add Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/add.wgsl").into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Add Pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            let chunk_elements = 32 * 1024 * 1024; // 128 MB elements
            for chunk_start in (0..gpu_elements).step_by(chunk_elements) {
                let current_elements = std::cmp::min(chunk_elements, gpu_elements - chunk_start);
                let size = (current_elements * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

                let a_chunk = &a_gpu[chunk_start..chunk_start + current_elements];
                let b_chunk = &b_gpu[chunk_start..chunk_start + current_elements];
                let res_chunk = &mut res_gpu[chunk_start..chunk_start + current_elements];

                let usage_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
                let buf_a = get_buffer(device, size, usage_storage, Some("Buf A"));
                let buf_b = get_buffer(device, size, usage_storage, Some("Buf B"));
                let buf_c = get_buffer(device, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, Some("Buf C"));
                let staging_buf = get_buffer(device, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, Some("Staging Buffer"));

                queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(a_chunk));
                queue.write_buffer(&buf_b, 0, bytemuck::cast_slice(b_chunk));

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Add Bindings"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
                    ],
                });
                
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    cpass.set_pipeline(&pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    let total_workgroups = ((current_elements as u32) + 63) / 64;
                    let wx = total_workgroups.min(65535);
                    let wy = (total_workgroups + wx - 1) / wx;
                    cpass.dispatch_workgroups(wx, wy, 1);
                }
                
                encoder.copy_buffer_to_buffer(&buf_c, 0, &staging_buf, 0, size);
                queue.submit(Some(encoder.finish()));
                
                let slice = staging_buf.slice(..size);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
                device.poll(wgpu::Maintain::Wait);
                rx.recv().unwrap().unwrap();
                
                let data = slice.get_mapped_range();
                let gpu_result: &[f32] = bytemuck::cast_slice(&data);
                res_chunk.copy_from_slice(gpu_result);
                drop(data);
                staging_buf.unmap();
                
                recycle_buffer(buf_a, size, usage_storage);
                recycle_buffer(buf_b, size, usage_storage);
                recycle_buffer(buf_c, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
                recycle_buffer(staging_buf, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            }
        }
    });

    result
}

// ---------------------------------------------------------
// MATRIX MULTIPLICATION
// ---------------------------------------------------------
pub fn execute_matmul(a_data: &[f32], b_data: &[f32], m: u32, k: u32, n: u32, is_hybrid: bool) -> Vec<f32> {
    let backend = BACKEND.get().expect("Backend not init");
    let device = &backend.device;
    let queue = &backend.queue;

    let total_elements = (m * n) as usize;
    let mut result = vec![0.0; total_elements];

    let gpu_ratio: f64 = if is_hybrid { 0.7 } else { 1.0 };
    // For matrix multiplication, we split the rows of the output matrix (m).
    let m_gpu = (m as f64 * gpu_ratio) as u32;
    let m_cpu = m - m_gpu;

    let a_elements_gpu = (m_gpu * k) as usize;
    let res_elements_gpu = (m_gpu * n) as usize;

    std::thread::scope(|s| {
        let (a_gpu, a_cpu) = a_data.split_at(a_elements_gpu);
        let (res_gpu, res_cpu) = result.split_at_mut(res_elements_gpu);

        // CPU Thread
        if m_cpu > 0 {
            s.spawn(move || {
                unsafe {
                    matrixmultiply::sgemm(
                        m_cpu as usize, k as usize, n as usize,
                        1.0,
                        a_cpu.as_ptr(), k as isize, 1,
                        b_data.as_ptr(), n as isize, 1,
                        0.0,
                        res_cpu.as_mut_ptr(), n as isize, 1,
                    );
                }
            });
        }

        // GPU Thread
        if m_gpu > 0 {
            let size_a = (m_gpu * k * 4) as wgpu::BufferAddress;
            let size_b = (k * n * 4) as wgpu::BufferAddress;
            let size_c = (m_gpu * n * 4) as wgpu::BufferAddress;

            let usage_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            let buf_a = get_buffer(device, size_a, usage_storage, Some("MatMul A"));
            let buf_b = get_buffer(device, size_b, usage_storage, Some("MatMul B"));
            let buf_c = get_buffer(device, size_c, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, Some("MatMul C"));
            let staging_buf = get_buffer(device, size_c, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, Some("Staging Buffer C"));

            queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(a_gpu));
            queue.write_buffer(&buf_b, 0, bytemuck::cast_slice(b_data));

            let dims: [u32; 3] = [m_gpu, k, n];
            let usage_uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
            let buf_dims = get_buffer(device, std::mem::size_of::<[u32; 3]>() as wgpu::BufferAddress, usage_uniform, Some("MatMul Dims"));
            queue.write_buffer(&buf_dims, 0, bytemuck::cast_slice(&dims));

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MatMul Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
            });
            
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MatMul Pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MatMul Bindings"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_dims.as_entire_binding() },
                ],
            });
            
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                let wg_x = (n + 15) / 16;
                let wg_y = (m_gpu + 15) / 16;
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            
            encoder.copy_buffer_to_buffer(&buf_c, 0, &staging_buf, 0, size_c);
            queue.submit(Some(encoder.finish()));
            
            let slice = staging_buf.slice(..size_c);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            
            let data = slice.get_mapped_range();
            let gpu_result: &[f32] = bytemuck::cast_slice(&data);
            res_gpu.copy_from_slice(gpu_result);
            drop(data);
            staging_buf.unmap();
            
            recycle_buffer(buf_a, size_a, usage_storage);
            recycle_buffer(buf_b, size_b, usage_storage);
            recycle_buffer(buf_c, size_c, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
            recycle_buffer(staging_buf, size_c, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            recycle_buffer(buf_dims, std::mem::size_of::<[u32; 3]>() as wgpu::BufferAddress, usage_uniform);
        }
    });

    result
}

// ---------------------------------------------------------
// NON-LINEAR ACTIVATIONS
// ---------------------------------------------------------
pub fn execute_activation(data: &[f32], activation_type: &str, is_hybrid: bool) -> Vec<f32> {
    let backend = BACKEND.get().expect("Backend not init");
    let device = &backend.device;
    let queue = &backend.queue;

    let total_elements = data.len();
    let mut result = vec![0.0; total_elements];

    let gpu_ratio: f64 = if is_hybrid { 0.7 } else { 1.0 };
    let gpu_elements = (total_elements as f64 * gpu_ratio) as usize;
    let cpu_elements = total_elements - gpu_elements;

    std::thread::scope(|s| {
        let (data_gpu, data_cpu) = data.split_at(gpu_elements);
        let (res_gpu, res_cpu) = result.split_at_mut(gpu_elements);

        if cpu_elements > 0 {
            // Because of the scoping and multi-threading we need to capture activation_type by value or copy it
            let act_type = activation_type.to_string();
            s.spawn(move || {
                use rayon::prelude::*;
                match act_type.as_str() {
                    "relu" => {
                        res_cpu.par_iter_mut().zip(data_cpu.par_iter()).for_each(|(o, &i)| *o = if i > 0.0 { i } else { 0.0 });
                    },
                    "sigmoid" => {
                        res_cpu.par_iter_mut().zip(data_cpu.par_iter()).for_each(|(o, &i)| *o = 1.0 / (1.0 + (-i).exp()));
                    },
                    "silu" => {
                        res_cpu.par_iter_mut().zip(data_cpu.par_iter()).for_each(|(o, &i)| *o = i * (1.0 / (1.0 + (-i).exp())));
                    },
                    _ => panic!("Unsupported CPU activation")
                }
            });
        }

        if gpu_elements > 0 {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Activation Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/activation.wgsl").into()),
            });
            
            // Note: need to capture the exact act_type correctly as str slice.
            let act_type_str = activation_type;
            let entry_point = match act_type_str {
                "relu" => "relu_main",
                "sigmoid" => "sigmoid_main",
                "silu" => "silu_main",
                _ => panic!("Unsupported activation specificied"),
            };

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Activation Pipeline"),
                layout: None,
                module: &shader,
                entry_point,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            let chunk_elements = 32 * 1024 * 1024; // 128 MB elements
            for chunk_start in (0..gpu_elements).step_by(chunk_elements) {
                let current_elements = std::cmp::min(chunk_elements, gpu_elements - chunk_start);
                let size = (current_elements * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

                let data_chunk = &data_gpu[chunk_start..chunk_start + current_elements];
                let res_chunk = &mut res_gpu[chunk_start..chunk_start + current_elements];

                let usage_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
                let buf_in = get_buffer(device, size, usage_storage, Some("Activation In"));
                let buf_out = get_buffer(device, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, Some("Activation Out"));
                let staging_buf = get_buffer(device, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, Some("Staging Buffer Activation"));

                queue.write_buffer(&buf_in, 0, bytemuck::cast_slice(data_chunk));
                
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Activation Binding"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_in.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_out.as_entire_binding() },
                    ],
                });
                
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    cpass.set_pipeline(&pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    let total_workgroups = ((current_elements as u32) + 63) / 64;
                    let wx = total_workgroups.min(65535);
                    let wy = (total_workgroups + wx - 1) / wx;
                    cpass.dispatch_workgroups(wx, wy, 1);
                }
                
                encoder.copy_buffer_to_buffer(&buf_out, 0, &staging_buf, 0, size);
                queue.submit(Some(encoder.finish()));
                
                let slice = staging_buf.slice(..size);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
                device.poll(wgpu::Maintain::Wait);
                rx.recv().unwrap().unwrap();
                
                let res_data = slice.get_mapped_range();
                let gpu_result: &[f32] = bytemuck::cast_slice(&res_data);
                res_chunk.copy_from_slice(gpu_result);
                drop(res_data);
                staging_buf.unmap();
                
                recycle_buffer(buf_in, size, usage_storage);
                recycle_buffer(buf_out, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
                recycle_buffer(staging_buf, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            }
        }
    });

    result
}
