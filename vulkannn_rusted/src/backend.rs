use std::sync::{OnceLock, Mutex};
use rayon::prelude::*;
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
    
    // Pipeline Cache
    pub add_pipeline: wgpu::ComputePipeline,
    pub matmul_pipeline: wgpu::ComputePipeline,
    pub relu_pipeline: wgpu::ComputePipeline,
    pub sigmoid_pipeline: wgpu::ComputePipeline,
    pub silu_pipeline: wgpu::ComputePipeline,
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

        // Compile Shaders and Pipelines once
        let add_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Add Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/add.wgsl").into()),
        });
        let add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Add Pipeline"),
            layout: None,
            module: &add_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
        });
        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"),
            layout: None,
            module: &matmul_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let activation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Activation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/activation.wgsl").into()),
        });
        // Note: Activation has multiple entry points. We'll use a specialized approach or just one for now.
        // For simplicity, we can create multiple pipelines or just change the entry point during execute?
        // Actually, entry_point is fixed in the pipeline. 
        // We need 3 pipelines for activation.
        
        let relu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ReLU Pipeline"),
            layout: None,
            module: &activation_shader,
            entry_point: "relu_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let sigmoid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sigmoid Pipeline"),
            layout: None,
            module: &activation_shader,
            entry_point: "sigmoid_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let silu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SiLU Pipeline"),
            layout: None,
            module: &activation_shader,
            entry_point: "silu_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        WgpuBackend {
            device,
            queue,
            buffer_cache: Mutex::new(Vec::new()),
            add_pipeline,
            matmul_pipeline,
            relu_pipeline,
            sigmoid_pipeline,
            silu_pipeline,
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
    let mut result = vec![0.0; a_data.len()];
    execute_add_into(a_data, b_data, &mut result, is_hybrid);
    result
}

pub fn execute_add_into(a_data: &[f32], b_data: &[f32], result: &mut [f32], is_hybrid: bool) {
    let backend = BACKEND.get().expect("Backend not initialized");
    let device = &backend.device;
    let queue = &backend.queue;

    let total_elements = a_data.len();

    let gpu_ratio: f64 = if is_hybrid { 0.7 } else { 1.0 };
    let gpu_elements = (total_elements as f64 * gpu_ratio) as usize;
    let cpu_elements = total_elements - gpu_elements;

    std::thread::scope(|s| {
        let (a_gpu, a_cpu) = a_data.split_at(gpu_elements);
        let (b_gpu, b_cpu) = b_data.split_at(gpu_elements);
        let (res_gpu, res_cpu) = result.split_at_mut(gpu_elements);

        if cpu_elements > 0 {
            s.spawn(move || {
                res_cpu.par_iter_mut()
                       .zip(a_cpu.par_iter())
                       .zip(b_cpu.par_iter())
                       .for_each(|((c, &a), &b)| *c = a + b);
            });
        }

        if gpu_elements > 0 {
            let pipeline = &backend.add_pipeline;
            let chunk_elements = 16 * 1024 * 1024; // 64MB chunks
            let num_chunks = (gpu_elements + chunk_elements - 1) / chunk_elements;
            
            // Triple Buffering: 3 sets of buffers
            const NUM_STAGES: usize = 3;
            let mut buf_a = Vec::with_capacity(NUM_STAGES);
            let mut buf_b = Vec::with_capacity(NUM_STAGES);
            let mut buf_c = Vec::with_capacity(NUM_STAGES);
            let mut staging_bufs = Vec::with_capacity(NUM_STAGES);
            
            let size = (chunk_elements * 4) as wgpu::BufferAddress;
            let usage_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            let usage_result = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            let usage_staging = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

            for i in 0..NUM_STAGES {
                buf_a.push(get_buffer(device, size, usage_storage, Some(&format!("Add A {}", i))));
                buf_b.push(get_buffer(device, size, usage_storage, Some(&format!("Add B {}", i))));
                buf_c.push(get_buffer(device, size, usage_result, Some(&format!("Add C {}", i))));
                staging_bufs.push(get_buffer(device, size, usage_staging, Some(&format!("Staging {}", i))));
            }

            // Pipelined Execution
            let mut inflight_receivers: std::collections::VecDeque<(std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>, usize, usize, usize)> = std::collections::VecDeque::new();
            
            for chunk_idx in 0..num_chunks {
                let stage = chunk_idx % NUM_STAGES;
                let chunk_start = chunk_idx * chunk_elements;
                let current_elements = std::cmp::min(chunk_elements, gpu_elements - chunk_start);
                let current_size = (current_elements * 4) as wgpu::BufferAddress;

                // 1. If buffer is full, wait for the oldest chunk and read it back
                if inflight_receivers.len() == NUM_STAGES {
                    let (rx, s_idx, elements, start_offset) = inflight_receivers.pop_front().unwrap();
                    rx.recv().unwrap().unwrap();
                    {
                        let data = staging_bufs[s_idx].slice(.. (elements * 4) as u64).get_mapped_range();
                        res_gpu[start_offset .. start_offset + elements].copy_from_slice(bytemuck::cast_slice(&data));
                    }
                    staging_bufs[s_idx].unmap();
                }

                // 2. Upload and Dispatch new chunk
                queue.write_buffer(&buf_a[stage], 0, bytemuck::cast_slice(&a_gpu[chunk_start..chunk_start + current_elements]));
                queue.write_buffer(&buf_b[stage], 0, bytemuck::cast_slice(&b_gpu[chunk_start..chunk_start + current_elements]));

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_a[stage].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_b[stage].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_c[stage].as_entire_binding() },
                    ],
                });

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    cpass.set_pipeline(&pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    let workgroups = (current_elements as u32 + 63) / 64;
                    cpass.dispatch_workgroups(workgroups.min(65535), (workgroups + 65534) / 65535, 1);
                }
                encoder.copy_buffer_to_buffer(&buf_c[stage], 0, &staging_bufs[stage], 0, current_size);
                queue.submit(Some(encoder.finish()));

                // 3. Map for read (non-blocking)
                let (tx, rx) = std::sync::mpsc::channel();
                staging_bufs[stage].slice(..current_size).map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
                inflight_receivers.push_back((rx, stage, current_elements, chunk_start));
                
                // Active polling to keep GPU/OS moving
                device.poll(wgpu::Maintain::Poll);
            }

            // Final Drain
            while let Some((rx, s_idx, elements, start_offset)) = inflight_receivers.pop_front() {
                device.poll(wgpu::Maintain::Wait);
                rx.recv().unwrap().unwrap();
                {
                    let data = staging_bufs[s_idx].slice(.. (elements * 4) as u64).get_mapped_range();
                    res_gpu[start_offset .. start_offset + elements].copy_from_slice(bytemuck::cast_slice(&data));
                }
                staging_bufs[s_idx].unmap();
            }

            // Cleanup
            for i in 0..NUM_STAGES {
                recycle_buffer(buf_a.pop().unwrap(), size, usage_storage);
                recycle_buffer(buf_b.pop().unwrap(), size, usage_storage);
                recycle_buffer(buf_c.pop().unwrap(), size, usage_result);
                recycle_buffer(staging_bufs.pop().unwrap(), size, usage_staging);
            }
        }
    });
}

// ---------------------------------------------------------
// MATRIX MULTIPLICATION (Dynamic Blockwise Tiled / CROOK Work Stealing)
// ---------------------------------------------------------
pub fn execute_matmul(a_data: &[f32], b_data: &[f32], m: u32, k: u32, n: u32, is_hybrid: bool) -> Vec<f32> {
    let backend = BACKEND.get().expect("Backend not init");
    let device = &backend.device;
    let queue = &backend.queue;

    let total_elements = (m * n) as usize;
    let mut result = vec![0.0; total_elements];

    // Constant tile sizes for RAM/VRAM safety
    let block_m = 512;
    let block_k = 2048;
    let block_n = 2048;

    let total_m_blocks = (m + block_m - 1) / block_m;
    
    // Dynamic Work Stealing Queue (Atomic Counter)
    let next_m_start = std::sync::atomic::AtomicU32::new(0);
    let completed_blocks = std::sync::atomic::AtomicU32::new(0);

    let res_ptr = result.as_mut_ptr() as usize;
    let a_ptr = a_data.as_ptr() as usize;
    let b_ptr = b_data.as_ptr() as usize;

    std::thread::scope(|s| {
        // CPU Worker Pool (Hybrid)
        if is_hybrid {
            let num_cpu_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            
            for _ in 0..num_cpu_threads {
                s.spawn(|| {
                    let res_ptr = res_ptr as *mut f32;
                    
                    loop {
                        let m_start = next_m_start.fetch_add(block_m, std::sync::atomic::Ordering::Relaxed);
                        if m_start >= m { break; }
                        let bm = std::cmp::min(block_m, m - m_start);
                        
                        let band_offset = (m_start * n) as usize;
                        let a_offset = (m_start * k) as usize;

                        // Zero-allocation SIMD BLAS! 
                        // We pipe the OS memmap pointers straight into the CPU matrix multiply.
                        // Linux will automatically aggressively cache the unrolled rows up to the 19GB limit!
                        unsafe {
                            matrixmultiply::sgemm(
                                bm as usize, k as usize, n as usize,
                                1.0,
                                (a_ptr as *const f32).add(a_offset), k as isize, 1,
                                b_ptr as *const f32, n as isize, 1,
                                0.0,
                                (res_ptr as *mut f32).add(band_offset), n as isize, 1,
                            );
                        }
                        
                        let done = completed_blocks.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        println!("  -> [CPU Worker] Rusted Compute Progress: {} / {} blocks processed...", done, total_m_blocks);
                        use std::io::Write;
                        let _ = std::io::stdout().flush();
                    }
                });
            }
        }

        // GPU Worker (Main Thread)
        s.spawn(|| {
            let pipeline = &backend.matmul_pipeline;
            let res_ptr = res_ptr as *mut f32;

            loop {
                let m_start = next_m_start.fetch_add(block_m, std::sync::atomic::Ordering::Relaxed);
                if m_start >= m { break; }
                let bm = std::cmp::min(block_m, m - m_start);
                
                let band_offset = (m_start * n) as usize;
                let band_len = (bm * n) as usize;
                let res_band = unsafe { std::slice::from_raw_parts_mut(res_ptr.add(band_offset), band_len) };
                
                for n_start in (0..n).step_by(block_n as usize) {
                    let bn = std::cmp::min(block_n, n - n_start);
                    
                    let mut c_block = vec![0.0; (bm * bn) as usize];
                    
                    let usage_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
                    // Buffers for Ping-Pong
                    let bufs_a = [
                        get_buffer(device, (bm * block_k * 4) as u64, usage_storage, Some("MatMul A0")),
                        get_buffer(device, (bm * block_k * 4) as u64, usage_storage, Some("MatMul A1")),
                    ];
                    let bufs_b = [
                        get_buffer(device, (block_k * bn * 4) as u64, usage_storage, Some("MatMul B0")),
                        get_buffer(device, (block_k * bn * 4) as u64, usage_storage, Some("MatMul B1")),
                    ];
                    let buf_c = get_buffer(device, (bm * bn * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, Some("MatMul C"));
                    
                    let dims_size = std::mem::size_of::<[u32; 3]>() as u64;
                    let buf_dims = get_buffer(device, dims_size, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, Some("MatMul Dims"));

                    // Initial Fill for first C block
                    queue.write_buffer(&buf_c, 0, bytemuck::cast_slice(&c_block));

                    let k_tiles: Vec<u32> = (0..k).step_by(block_k as usize).collect();
                    
                    for (t_idx, &k_start) in k_tiles.iter().enumerate() {
                        let bk = std::cmp::min(block_k, k - k_start);
                        let p_idx = t_idx % 2; // Ping-pong index

                        // 1. Prepare data for CURRENT tile in RAM (L2)
                        let mut a_block = vec![0.0; (bm * bk) as usize];
                        let mut b_block = vec![0.0; (bk * bn) as usize];

                        a_block.par_chunks_mut(bk as usize).enumerate().for_each(|(i, chunk): (usize, &mut [f32])| {
                            let global_row_a = m_start + i as u32;
                            let a_row_start = (global_row_a * k + k_start) as usize;
                            chunk.copy_from_slice(&a_data[a_row_start .. a_row_start + (bk as usize)]);
                        });

                        b_block.par_chunks_mut(bn as usize).enumerate().for_each(|(i, chunk): (usize, &mut [f32])| {
                            let global_row_b = k_start + i as u32;
                            let b_row_start = (global_row_b * n + n_start) as usize;
                            chunk.copy_from_slice(&b_data[b_row_start .. b_row_start + (bn as usize)]);
                        });

                        // 2. Upload to VRAM (L1) - Overlaps with previous dispatch if queue depth allows
                        queue.write_buffer(&bufs_a[p_idx], 0, bytemuck::cast_slice(&a_block));
                        queue.write_buffer(&bufs_b[p_idx], 0, bytemuck::cast_slice(&b_block));
                        
                        let dims: [u32; 3] = [bm, bk, bn];
                        queue.write_buffer(&buf_dims, 0, bytemuck::cast_slice(&dims));

                        // 3. Dispatch Compute
                        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("MatMul Bindings"),
                            layout: &pipeline.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: bufs_a[p_idx].as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 1, resource: bufs_b[p_idx].as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 3, resource: buf_dims.as_entire_binding() },
                            ],
                        });

                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                        {
                            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                            cpass.set_pipeline(&pipeline);
                            cpass.set_bind_group(0, &bind_group, &[]);
                            cpass.dispatch_workgroups((bn + 15) / 16, (bm + 15) / 16, 1);
                        }
                        
                        // If last tile, we need to read back
                        if t_idx == k_tiles.len() - 1 {
                            let size_c_bytes = (bm * bn * 4) as u64;
                            let staging_buf = get_buffer(device, size_c_bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, Some("Staging C"));
                            encoder.copy_buffer_to_buffer(&buf_c, 0, &staging_buf, 0, size_c_bytes);
                            queue.submit(Some(encoder.finish()));

                            let slice = staging_buf.slice(..size_c_bytes);
                            let (tx, rx) = std::sync::mpsc::channel();
                            slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
                            device.poll(wgpu::Maintain::Wait);
                            rx.recv().unwrap().unwrap();

                            let data = slice.get_mapped_range();
                            c_block.copy_from_slice(bytemuck::cast_slice(&data));
                            drop(data);
                            staging_buf.unmap();
                            recycle_buffer(staging_buf, (bm * bn * 4) as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
                        } else {
                            queue.submit(Some(encoder.finish()));
                        }
                    }

                    // Recycle all resources
                    // We move buffers out of the array to avoid cloning
                    let [a0, a1] = bufs_a;
                    let [b0, b1] = bufs_b;
                    recycle_buffer(a0, (bm * block_k * 4) as u64, usage_storage);
                    recycle_buffer(a1, (bm * block_k * 4) as u64, usage_storage);
                    recycle_buffer(b0, (block_k * bn * 4) as u64, usage_storage);
                    recycle_buffer(b1, (block_k * bn * 4) as u64, usage_storage);
                    recycle_buffer(buf_c, (bm * bn * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
                    recycle_buffer(buf_dims, dims_size, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                    
                    // Final write-back of the accumulated block to the global result band
                    res_band.par_chunks_mut(n as usize).skip((0) as usize).enumerate().take(bm as usize).for_each(|(i, row)| {
                        let src_start = i * bn as usize;
                        let dst_start = n_start as usize;
                        row[dst_start .. dst_start + bn as usize].copy_from_slice(&c_block[src_start .. src_start + bn as usize]);
                    });
                }
                
                let done = completed_blocks.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                println!("  -> [GPU Worker] Rusted Compute Progress: {} / {} blocks processed...", done, total_m_blocks);
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        });
    });

    result
}

// ---------------------------------------------------------
// NON-LINEAR ACTIVATIONS
// ---------------------------------------------------------
pub fn execute_activation(input_data: &[f32], op: &str, is_hybrid: bool) -> Vec<f32> {
    let mut result = vec![0.0; input_data.len()];
    execute_activation_into(input_data, op, &mut result, is_hybrid);
    result
}

pub fn execute_activation_into(input_data: &[f32], op: &str, result: &mut [f32], is_hybrid: bool) {
    let backend = BACKEND.get().expect("Backend not initialized");
    let device = &backend.device;
    let queue = &backend.queue;

    let total_elements = input_data.len();

    let gpu_ratio: f64 = if is_hybrid { 0.7 } else { 1.0 };
    let gpu_elements = (total_elements as f64 * gpu_ratio) as usize;
    let cpu_elements = total_elements - gpu_elements;

    std::thread::scope(|s| {
        let (i_gpu, i_cpu) = input_data.split_at(gpu_elements);
        let (res_gpu, res_cpu) = result.split_at_mut(gpu_elements);

        if cpu_elements > 0 {
            // The `is_hybrid` parameter is used, so no need for `_is_hybrid` prefix.
            // The `op` parameter is captured by value in the closure.
            let _op = op.to_string(); // Capture op by value for the spawned thread
            s.spawn(move || {
                use rayon::prelude::*;
                match _op.as_str() {
                    "relu" => res_cpu.par_iter_mut().zip(i_cpu.par_iter()).for_each(|(o, &i)| *o = if i > 0.0 { i } else { 0.0 }),
                    "sigmoid" => res_cpu.par_iter_mut().zip(i_cpu.par_iter()).for_each(|(o, &i)| *o = 1.0 / (1.0 + (-i).exp())),
                    "silu" => res_cpu.par_iter_mut().zip(i_cpu.par_iter()).for_each(|(o, &i)| *o = i * (1.0 / (1.0 + (-i).exp()))),
                    _ => {} // Should not happen if `op` is validated earlier
                }
            });
        }

        if gpu_elements > 0 {
            let pipeline = match op {
                "relu" => &backend.relu_pipeline,
                "sigmoid" => &backend.sigmoid_pipeline,
                "silu" => &backend.silu_pipeline,
                _ => panic!("Unknown activation: {}", op),
            };

            let chunk_elements = 16 * 1024 * 1024; // 64MB chunks (16M floats)
            let num_chunks = (gpu_elements + chunk_elements - 1) / chunk_elements;
            
            const NUM_STAGES: usize = 3;
            let mut buf_i = Vec::with_capacity(NUM_STAGES);
            let mut buf_o = Vec::with_capacity(NUM_STAGES);
            let mut staging_bufs = Vec::with_capacity(NUM_STAGES);
            
            let size = (chunk_elements * 4) as wgpu::BufferAddress;
            let usage_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            let usage_result = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            let usage_staging = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

            for _i in 0..NUM_STAGES {
                buf_i.push(get_buffer(device, size, usage_storage, Some(&format!("Act I {}", _i))));
                buf_o.push(get_buffer(device, size, usage_result, Some(&format!("Act O {}", _i))));
                staging_bufs.push(get_buffer(device, size, usage_staging, Some(&format!("Staging {}", _i))));
            }

            let mut inflight_receivers: std::collections::VecDeque<(std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>, usize, usize, usize)> = std::collections::VecDeque::new();
            
            for chunk_idx in 0..num_chunks {
                let stage = chunk_idx % NUM_STAGES;
                let chunk_start = chunk_idx * chunk_elements;
                let current_elements = std::cmp::min(chunk_elements, gpu_elements - chunk_start);
                let current_size = (current_elements * 4) as wgpu::BufferAddress;

                if inflight_receivers.len() == NUM_STAGES {
                    let (rx, s_idx, elements, start_offset) = inflight_receivers.pop_front().unwrap();
                    rx.recv().unwrap().unwrap();
                    {
                        let data = staging_bufs[s_idx].slice(.. (elements * 4) as u64).get_mapped_range();
                        res_gpu[start_offset .. start_offset + elements].copy_from_slice(bytemuck::cast_slice(&data));
                    }
                    staging_bufs[s_idx].unmap();
                }

                queue.write_buffer(&buf_i[stage], 0, bytemuck::cast_slice(&i_gpu[chunk_start..chunk_start + current_elements]));

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_i[stage].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_o[stage].as_entire_binding() },
                    ],
                });

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    cpass.set_pipeline(&pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    let workgroups = (current_elements as u32 + 63) / 64;
                    cpass.dispatch_workgroups(workgroups.min(65535), (workgroups + 65534) / 65535, 1);
                }
                encoder.copy_buffer_to_buffer(&buf_o[stage], 0, &staging_bufs[stage], 0, current_size);
                queue.submit(Some(encoder.finish()));

                let (tx, rx) = std::sync::mpsc::channel();
                staging_bufs[stage].slice(..current_size).map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
                inflight_receivers.push_back((rx, stage, current_elements, chunk_start));
                
                device.poll(wgpu::Maintain::Poll);
            }

            while let Some((rx, s_idx, elements, start_offset)) = inflight_receivers.pop_front() {
                device.poll(wgpu::Maintain::Wait);
                rx.recv().unwrap().unwrap();
                {
                    let data = staging_bufs[s_idx].slice(.. (elements * 4) as u64).get_mapped_range();
                    res_gpu[start_offset .. start_offset + elements].copy_from_slice(bytemuck::cast_slice(&data));
                }
                staging_bufs[s_idx].unmap();
            }

            for _i in 0..NUM_STAGES {
                recycle_buffer(buf_i.pop().unwrap(), size, usage_storage);
                recycle_buffer(buf_o.pop().unwrap(), size, usage_result);
                recycle_buffer(staging_bufs.pop().unwrap(), size, usage_staging);
            }
        }
    });
}
