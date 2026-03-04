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
    pub vec_cache: Mutex<Vec<Vec<f32>>>,
    
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

        let add_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Add Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/add.wgsl").into()),
        });
        let add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Add Pipeline"), layout: None, module: &add_shader, entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
        });
        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"), layout: None, module: &matmul_shader, entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let activation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Activation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/activation.wgsl").into()),
        });
        
        let relu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ReLU Pipeline"), layout: None, module: &activation_shader, entry_point: "relu_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });
        let sigmoid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sigmoid Pipeline"), layout: None, module: &activation_shader, entry_point: "sigmoid_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });
        let silu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SiLU Pipeline"), layout: None, module: &activation_shader, entry_point: "silu_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        WgpuBackend {
            device, queue, buffer_cache: Mutex::new(Vec::new()),
            vec_cache: Mutex::new(Vec::new()),
            add_pipeline, matmul_pipeline, relu_pipeline, sigmoid_pipeline, silu_pipeline,
        }
    });
}

pub fn get_buffer(device: &Device, size: wgpu::BufferAddress, usage: wgpu::BufferUsages, label: Option<&str>) -> wgpu::Buffer {
    let backend = BACKEND.get().unwrap();
    if let Ok(mut cache) = backend.buffer_cache.lock() {
        if let Some(idx) = cache.iter().position(|b| b.size >= size && b.usage == usage) {
            let cached = cache.swap_remove(idx);
            return cached.buffer;
        }
    }
    device.create_buffer(&wgpu::BufferDescriptor { label, size, usage, mapped_at_creation: false })
}

pub fn recycle_buffer(buffer: wgpu::Buffer, size: wgpu::BufferAddress, usage: wgpu::BufferUsages) {
    if let Ok(mut cache) = BACKEND.get().unwrap().buffer_cache.lock() {
        cache.push(CachedBuffer { size, usage, buffer });
    }
}

pub fn get_vec(size: usize, zeroed: bool) -> Vec<f32> {
    if let Ok(mut cache) = BACKEND.get().unwrap().vec_cache.lock() {
        if let Some(idx) = cache.iter().position(|v| v.capacity() >= size) {
            let mut v = cache.swap_remove(idx);
            v.resize(size, 0.0);
            if zeroed { v.fill(0.0); }
            return v;
        }
    }
    vec![0.0; size]
}

pub fn recycle_vec(v: Vec<f32>) {
    if let Ok(mut cache) = BACKEND.get().unwrap().vec_cache.lock() {
        cache.push(v);
    }
}

#[allow(dead_code)]
pub fn execute_add(a_data: &[f32], b_data: &[f32], is_hybrid: bool) -> Vec<f32> {
    let mut result = vec![0.0; a_data.len()];
    execute_add_into(a_data, b_data, &mut result, is_hybrid, false);
    result
}

pub fn execute_add_into(a_data: &[f32], b_data: &[f32], result: &mut [f32], is_hybrid: bool, use_staging: bool) {
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
                res_cpu.par_iter_mut().zip(a_cpu.par_iter()).zip(b_cpu.par_iter()).for_each(|((c, &a), &b)| *c = a + b);
            });
        }

        if gpu_elements > 0 {
            let pipeline = &backend.add_pipeline;
            let chunk_elements = 8 * 1024 * 1024; // 32MB
            let num_chunks = (gpu_elements + chunk_elements - 1) / chunk_elements;
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

            let mut inflight_receivers: std::collections::VecDeque<(std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>, usize, usize, usize)> = std::collections::VecDeque::new();
            
            for chunk_idx in 0..num_chunks {
                let stage = chunk_idx % NUM_STAGES;
                let chunk_start = chunk_idx * chunk_elements;
                let current_elements = std::cmp::min(chunk_elements, gpu_elements - chunk_start);
                let current_size = (current_elements * 4) as wgpu::BufferAddress;

                if inflight_receivers.len() >= NUM_STAGES {
                    let (rx, s_idx, elements, start_offset) = inflight_receivers.pop_front().unwrap();
                    // Polling in a loop is often faster than a single Maintain::Wait for high-throughput
                    while rx.try_recv().is_err() { device.poll(wgpu::Maintain::Poll); }
                    {
                        let data = staging_bufs[s_idx].slice(.. (elements * 4) as u64).get_mapped_range();
                        res_gpu[start_offset .. start_offset + elements].copy_from_slice(bytemuck::cast_slice(&data));
                    }
                    staging_bufs[s_idx].unmap();
                }

                if use_staging {
                    let mut tmp_a = vec![0.0; current_elements];
                    let mut tmp_b = vec![0.0; current_elements];
                    tmp_a.copy_from_slice(&a_gpu[chunk_start..chunk_start + current_elements]);
                    tmp_b.copy_from_slice(&b_gpu[chunk_start..chunk_start + current_elements]);
                    queue.write_buffer(&buf_a[stage], 0, bytemuck::cast_slice(&tmp_a));
                    queue.write_buffer(&buf_b[stage], 0, bytemuck::cast_slice(&tmp_b));
                } else {
                    queue.write_buffer(&buf_a[stage], 0, bytemuck::cast_slice(&a_gpu[chunk_start..chunk_start + current_elements]));
                    queue.write_buffer(&buf_b[stage], 0, bytemuck::cast_slice(&b_gpu[chunk_start..chunk_start + current_elements]));
                }

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None, layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_a[stage].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_b[stage].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_c[stage].as_entire_binding() },
                    ],
                });

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    cpass.set_pipeline(pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch_workgroups((current_elements as u32 + 255) / 256, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&buf_c[stage], 0, &staging_bufs[stage], 0, current_size);
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
                recycle_buffer(buf_a.pop().unwrap(), size, usage_storage);
                recycle_buffer(buf_b.pop().unwrap(), size, usage_storage);
                recycle_buffer(buf_c.pop().unwrap(), size, usage_result);
                recycle_buffer(staging_bufs.pop().unwrap(), size, usage_staging);
            }
        }
    });
}

pub fn execute_matmul(a_data: &[f32], b_data: &[f32], m: u32, k: u32, n: u32, is_hybrid: bool) -> Vec<f32> {
    let backend = BACKEND.get().expect("Backend not init");
    let device = &backend.device;
    let queue = &backend.queue;
    let total_elements = (m * n) as usize;
    let result = vec![0.0; total_elements];
    
    // Adaptive tiling
    let (block_m, block_k, block_n) = if m <= 16 {
        (16, 16384, 2048) // v2.8.17: "The Union". forced N-tiling for GEMV to enable Double Buffering.
    } else {
        (512, 16384, 1024) // 512x16384 A (32MB), 16384x1024 B (64MB). Safe for 2GB VRAM.
    };
    
    let total_m_blocks = (m + block_m - 1) / block_m;
    let next_m_start = std::sync::atomic::AtomicU32::new(0);
    let completed_blocks = std::sync::atomic::AtomicU32::new(0);
    #[allow(unused_mut)]
    let mut result_data = result;
    let res_ptr = result_data.as_mut_ptr() as usize;

    std::thread::scope(|s| {
        if is_hybrid {
            s.spawn(|| {
                let res_ptr = res_ptr as *mut f32;
                loop {
                    let m_start = next_m_start.fetch_add(block_m, std::sync::atomic::Ordering::Relaxed);
                    if m_start >= m { break; }
                    let bm = std::cmp::min(block_m, m - m_start);
                    let band_offset = (m_start * n) as usize;
                    let a_offset = (m_start * k) as usize;
                    unsafe {
                        matrixmultiply::sgemm(
                            bm as usize, k as usize, n as usize, 1.0,
                            a_data.as_ptr().add(a_offset), k as isize, 1,
                            b_data.as_ptr(), n as isize, 1, 0.0,
                            res_ptr.add(band_offset), n as isize, 1,
                        );
                    }
                    let done = completed_blocks.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    println!("  -> [CPU Worker] Rusted Compute Progress: {} / {} blocks processed...", done, total_m_blocks);
                }
            });
        }

        s.spawn(|| {
            let pipeline = &backend.matmul_pipeline;
            let res_ptr = res_ptr as *mut f32;
            let mut a_block = get_vec((block_m * block_k) as usize, false);
            let mut b_block = get_vec((block_k * block_n) as usize, false);
            let mut c_block = get_vec((block_m * block_n) as usize, true);

            let k_tiles: Vec<u32> = (0..k).step_by(block_k as usize).collect();
            loop {
                let m_start = next_m_start.fetch_add(block_m, std::sync::atomic::Ordering::Relaxed);
                if m_start >= m { break; }
                let bm = std::cmp::min(block_m, m - m_start);

                for &k_start in &k_tiles {
                    let bk = std::cmp::min(block_k, k - k_start);
                    let size_a = (bm * bk * 4) as u64;
                    let buf_a = get_buffer(device, size_a, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, Some("MatMul A"));
                    
                    // Upload A-tile ONCE for all N-blocks
                    let cur_a_len = (bm * bk) as usize;
                    if bm == 1 {
                        unsafe { std::ptr::copy_nonoverlapping(a_data.as_ptr().add((m_start * k + k_start) as usize), a_block.as_mut_ptr(), cur_a_len); }
                    } else {
                        a_block[..cur_a_len].par_chunks_mut(bk as usize).enumerate().for_each(|(i, chunk)| {
                            let a_row_start = ( (m_start + i as u32) * k + k_start) as usize;
                            chunk.copy_from_slice(&a_data[a_row_start .. a_row_start + (bk as usize)]);
                        });
                    }
                    queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(&a_block[..cur_a_len]));

                    let bufs_b = [
                        get_buffer(device, (bk * block_n * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, Some("MatMul B0")),
                        get_buffer(device, (bk * block_n * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, Some("MatMul B1")),
                    ];

                    let n_steps: Vec<u32> = (0..n).step_by(block_n as usize).collect();
                    let mut staging_requests = Vec::new();

                    let mut gemv_command_buffers = Vec::new();
                    for (ni, &n_start) in n_steps.iter().enumerate() {
                        let bn = std::cmp::min(block_n, n - n_start);
                        let p_idx = ni % 2;
                        let next_p_idx = (ni + 1) % 2;
                        let _size_b = (bk * bn * 4) as u64;
                        let size_c = (bm * bn * 4) as u64;
                        let band_offset = (m_start * n) as usize;

                        // 1. Upload CURRENT B-tile if not pre-fetched
                        if ni == 0 {
                            if bn == n && n_start == 0 {
                                queue.write_buffer(&bufs_b[p_idx], 0, bytemuck::cast_slice(&b_data[(k_start * n) as usize .. ((k_start + bk) * n) as usize]));
                            } else {
                                let cur_b_len = (bk * bn) as usize;
                                b_block[..cur_b_len].par_chunks_mut(bn as usize).enumerate().for_each(|(i, chunk)| {
                                    let b_row_start = ( (k_start + i as u32) * n + n_start) as usize;
                                    chunk.copy_from_slice(&b_data[b_row_start .. b_row_start + (bn as usize)]);
                                });
                                queue.write_buffer(&bufs_b[p_idx], 0, bytemuck::cast_slice(&b_block[..cur_b_len]));
                            }
                        }

                        // 2. PRE-FETCH NEXT B-tile
                        if let Some(&next_n_start) = n_steps.get(ni + 1) {
                            let nbn = std::cmp::min(block_n, n - next_n_start);
                            if nbn == n && next_n_start == 0 {
                                queue.write_buffer(&bufs_b[next_p_idx], 0, bytemuck::cast_slice(&b_data[(k_start * n) as usize .. ((k_start + bk) * n) as usize]));
                            } else {
                                let next_b_len = (bk * nbn) as usize;
                                b_block[..next_b_len].par_chunks_mut(nbn as usize).enumerate().for_each(|(i, chunk)| {
                                    let b_row_start = ( (k_start + i as u32) * n + next_n_start) as usize;
                                    chunk.copy_from_slice(&b_data[b_row_start .. b_row_start + (nbn as usize)]);
                                });
                                queue.write_buffer(&bufs_b[next_p_idx], 0, bytemuck::cast_slice(&b_block[..next_b_len]));
                            }
                        }

                        let buf_c = get_buffer(device, size_c, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, Some("MatMul C"));
                        let buf_dims = get_buffer(device, 12, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, Some("MatMul Dims"));

                        if k_start == 0 {
                            c_block[.. (bm * bn) as usize].fill(0.0);
                            queue.write_buffer(&buf_c, 0, bytemuck::cast_slice(&c_block[.. (bm * bn) as usize]));
                        }

                        let dims: [u32; 3] = [bm, bk, bn];
                        queue.write_buffer(&buf_dims, 0, bytemuck::cast_slice(&dims));

                        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None, layout: &pipeline.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 1, resource: bufs_b[p_idx].as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 3, resource: buf_dims.as_entire_binding() },
                            ],
                        });

                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                        {
                            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                            cpass.set_pipeline(pipeline);
                            cpass.set_bind_group(0, &bind_group, &[]);
                            cpass.dispatch_workgroups((bn + 15) / 16, (bm + 15) / 16, 1);
                        }

                        if k_start + bk >= k {
                            let staging_buf = get_buffer(device, size_c, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, Some("Staging C"));
                            encoder.copy_buffer_to_buffer(&buf_c, 0, &staging_buf, 0, size_c);
                            queue.submit(Some(encoder.finish()));
                            
                            let (tx, rx) = std::sync::mpsc::channel();
                            staging_buf.slice(..size_c).map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
                            staging_requests.push((rx, staging_buf, n_start, bn, size_c, band_offset));
                        } else {
                            if bm == 1 {
                                gemv_command_buffers.push(encoder.finish());
                            } else {
                                queue.submit(Some(encoder.finish()));
                            }
                        }

                        recycle_buffer(buf_c, size_c, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
                        recycle_buffer(buf_dims, 12, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                    }
                    if !gemv_command_buffers.is_empty() {
                        queue.submit(gemv_command_buffers);
                    }

                    // Ghost Speed: Batch Wait & Copy
                    if !staging_requests.is_empty() {
                        device.poll(wgpu::Maintain::Wait);
                        for (rx, staging_buf, ns, b_n, s_c, b_o) in staging_requests {
                            rx.recv().unwrap().unwrap();
                            {
                                let data = staging_buf.slice(..s_c).get_mapped_range();
                                let data_f32: &[f32] = bytemuck::cast_slice(&data);
                                unsafe {
                                    if bm == 1 {
                                        // GEMV FAST PATH: single copy
                                        std::ptr::copy_nonoverlapping(data_f32.as_ptr(), res_ptr.add(b_o + ns as usize), b_n as usize);
                                    } else {
                                        for i in 0..bm as usize {
                                            let src_row = &data_f32[i * b_n as usize .. (i + 1) * b_n as usize];
                                            let dst_row_ptr = res_ptr.add(b_o + ns as usize + i * n as usize);
                                            std::ptr::copy_nonoverlapping(src_row.as_ptr(), dst_row_ptr, b_n as usize);
                                        }
                                    }
                                }
                            }
                            staging_buf.unmap();
                            recycle_buffer(staging_buf, s_c, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
                        }
                    }

                    let [b0, b1] = bufs_b;
                    recycle_buffer(b0, (bk * block_n * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
                    recycle_buffer(b1, (bk * block_n * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
                    recycle_buffer(buf_a, size_a, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
                }
                let done = completed_blocks.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                println!("  -> [GPU Worker] Rusted Compute Progress: {} / {} blocks processed...", done, total_m_blocks);
            }
            recycle_vec(a_block);
            recycle_vec(b_block);
            recycle_vec(c_block);
        });
    });

    result_data
}

#[allow(dead_code)]
pub fn execute_activation(input_data: &[f32], op: &str, is_hybrid: bool) -> Vec<f32> {
    let mut result = vec![0.0; input_data.len()];
    execute_activation_into(input_data, op, &mut result, is_hybrid, false);
    result
}

pub fn execute_activation_into(input_data: &[f32], op: &str, result: &mut [f32], is_hybrid: bool, _use_staging: bool) {
    let backend = BACKEND.get().expect("Backend not initialized");
    let device = &backend.device;
    let queue = &backend.queue;
    let total_elements = input_data.len();
    
    // Performance Threshold: Avoid GPU overhead for small tensors
    if is_hybrid && total_elements < 2_000_000 {
        match op {
            "relu" => result.par_iter_mut().zip(input_data.par_iter()).for_each(|(o, &i)| *o = if i > 0.0 { i } else { 0.0 }),
            "sigmoid" => result.par_iter_mut().zip(input_data.par_iter()).for_each(|(o, &i)| *o = 1.0 / (1.0 + (-i).exp())),
            "silu" => result.par_iter_mut().zip(input_data.par_iter()).for_each(|(o, &i)| *o = i * (1.0 / (1.0 + (-i).exp()))),
            _ => {}
        }
        return;
    }

    let gpu_ratio: f64 = if is_hybrid { 0.7 } else { 1.0 };
    let gpu_elements = (total_elements as f64 * gpu_ratio) as usize;
    let cpu_elements = total_elements - gpu_elements;
    if is_hybrid {
        std::thread::scope(|s| {
            let (i_gpu, i_cpu) = input_data.split_at(gpu_elements);
            let (res_gpu, res_cpu) = result.split_at_mut(gpu_elements);

            if cpu_elements > 0 {
                let _op = op.to_string();
                let compute_cpu = move || {
                    if cpu_elements < 1_000_000 {
                        match _op.as_str() {
                            "relu" => for (o, &i) in res_cpu.iter_mut().zip(i_cpu.iter()) { *o = if i > 0.0 { i } else { 0.0 } },
                            "sigmoid" => for (o, &i) in res_cpu.iter_mut().zip(i_cpu.iter()) { *o = 1.0 / (1.0 + (-i).exp()) },
                            "silu" => for (o, &i) in res_cpu.iter_mut().zip(i_cpu.iter()) { *o = i * (1.0 / (1.0 + (-i).exp())) },
                            _ => {}
                        }
                    } else {
                        use rayon::prelude::*;
                        match _op.as_str() {
                            "relu" => res_cpu.par_iter_mut().zip(i_cpu.par_iter()).for_each(|(o, &i)| *o = if i > 0.0 { i } else { 0.0 }),
                            "sigmoid" => res_cpu.par_iter_mut().zip(i_cpu.par_iter()).for_each(|(o, &i)| *o = 1.0 / (1.0 + (-i).exp())),
                            "silu" => res_cpu.par_iter_mut().zip(i_cpu.par_iter()).for_each(|(o, &i)| *o = i * (1.0 / (1.0 + (-i).exp()))),
                            _ => {}
                        }
                    }
                };
                s.spawn(compute_cpu);
            }

            if gpu_elements > 0 {
                let pipeline = match op {
                    "relu" => &backend.relu_pipeline,
                    "sigmoid" => &backend.sigmoid_pipeline,
                    "silu" => &backend.silu_pipeline,
                    _ => panic!("Unknown activation: {}", op),
                };

                // FAST PATH for small tensors
                if gpu_elements <= 1024 * 1024 {
                    let size = (gpu_elements * 4) as wgpu::BufferAddress;
                    let buf_i = get_buffer(device, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, Some("ReLU In"));
                    let buf_o = get_buffer(device, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, Some("ReLU Out"));
                    
                    queue.write_buffer(&buf_i, 0, bytemuck::cast_slice(i_gpu));
                    
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None, layout: &pipeline.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: buf_i.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: buf_o.as_entire_binding() },
                        ],
                    });

                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                        cpass.set_pipeline(pipeline);
                        cpass.set_bind_group(0, &bind_group, &[]);
                        cpass.dispatch_workgroups((gpu_elements as u32 + 63) / 64, 1, 1);
                    }
                    
                    let staging_buf = get_buffer(device, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, Some("Staging ReLU"));
                    encoder.copy_buffer_to_buffer(&buf_o, 0, &staging_buf, 0, size);
                    queue.submit(Some(encoder.finish()));

                    let (tx, rx) = std::sync::mpsc::channel();
                    staging_buf.slice(..size).map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
                    device.poll(wgpu::Maintain::Wait);
                    rx.recv().unwrap().unwrap();
                    {
                        let data = staging_buf.slice(..size).get_mapped_range();
                        res_gpu.copy_from_slice(bytemuck::cast_slice(&data));
                    }
                    staging_buf.unmap();
                    recycle_buffer(buf_i, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
                    recycle_buffer(buf_o, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
                    recycle_buffer(staging_buf, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
                } else {
                    // Multi-stage path (reduced for brevity, keeping same logic)
                    let num_stages = (gpu_elements + 1024 * 1024 - 1) / (1024 * 1024);
                    let _size = (1024 * 1024 * 4) as wgpu::BufferAddress;
                    let tmp_i = get_vec(1024 * 1024, false);
                    for stage in 0..num_stages {
                        let elements = std::cmp::min(1024 * 1024, gpu_elements - stage * 1024 * 1024);
                        let cur_size = (elements * 4) as wgpu::BufferAddress;
                        let bi = get_buffer(device, cur_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, None);
                        let bo = get_buffer(device, cur_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, None);
                        
                        queue.write_buffer(&bi, 0, bytemuck::cast_slice(&i_gpu[stage * 1024 * 1024 .. stage * 1024 * 1024 + elements]));
                        
                        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None, layout: &pipeline.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: bi.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 1, resource: bo.as_entire_binding() },
                            ],
                        });
                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                        {
                            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                            cpass.set_pipeline(pipeline);
                            cpass.set_bind_group(0, &bg, &[]);
                            cpass.dispatch_workgroups((elements as u32 + 63) / 64, 1, 1);
                        }
                        let sb = get_buffer(device, cur_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, None);
                        encoder.copy_buffer_to_buffer(&bo, 0, &sb, 0, cur_size);
                        queue.submit(Some(encoder.finish()));
                        
                        let (tx, rx) = std::sync::mpsc::channel();
                        sb.slice(..cur_size).map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
                        device.poll(wgpu::Maintain::Wait);
                        rx.recv().unwrap().unwrap();
                        {
                            let data = sb.slice(..cur_size).get_mapped_range();
                            res_gpu[stage * 1024 * 1024 .. stage * 1024 * 1024 + elements].copy_from_slice(bytemuck::cast_slice(&data));
                        }
                        sb.unmap();
                        recycle_buffer(bi, cur_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
                        recycle_buffer(bo, cur_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
                        recycle_buffer(sb, cur_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
                    }
                    recycle_vec(tmp_i);
                }
            }
        });
    } else {
        // CPU ONLY - NO THREADING overhead for small activations
        let _op = op.to_string();
        if input_data.len() < 1_000_000 {
            match _op.as_str() {
                "relu" => for (o, &i) in result.iter_mut().zip(input_data.iter()) { *o = if i > 0.0 { i } else { 0.0 } },
                "sigmoid" => for (o, &i) in result.iter_mut().zip(input_data.iter()) { *o = 1.0 / (1.0 + (-i).exp()) },
                "silu" => for (o, &i) in result.iter_mut().zip(input_data.iter()) { *o = i * (1.0 / (1.0 + (-i).exp())) },
                _ => {}
            }
        } else {
            use rayon::prelude::*;
            match _op.as_str() {
                "relu" => result.par_iter_mut().zip(input_data.par_iter()).for_each(|(o, &i)| *o = if i > 0.0 { i } else { 0.0 }),
                "sigmoid" => result.par_iter_mut().zip(input_data.par_iter()).for_each(|(o, &i)| *o = 1.0 / (1.0 + (-i).exp())),
                "silu" => result.par_iter_mut().zip(input_data.par_iter()).for_each(|(o, &i)| *o = i * (1.0 / (1.0 + (-i).exp()))),
                _ => {}
            }
        }
    }
}
