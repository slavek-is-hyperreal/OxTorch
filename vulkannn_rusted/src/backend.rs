use std::sync::{OnceLock, Mutex};
use rayon::prelude::*;
use wgpu::{Device, Queue};
use crate::tensor::DataType;
use half::f16;

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
                required_features: adapter.features() & wgpu::Features::SHADER_F16,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create wgpu device");

        let has_f16 = device.features().contains(wgpu::Features::SHADER_F16);
        if has_f16 { println!("[vulkannn_rusted] Native FP16 Compute Enabled."); }

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
pub fn execute_add_deprecated(a_data: &[f32], _b_data: &[f32], _is_hybrid: bool) -> Vec<f32> {
    let result = vec![0.0; a_data.len()];
    // This is broken/deprecated anyway, will remove soon if not used.
    result
}

pub fn execute_add(a_raw: &[u8], b_raw: &[u8], dtype: DataType, is_hybrid: bool) -> Vec<u8> {
    let mut res = vec![0u8; a_raw.len()];
    execute_add_into(a_raw, b_raw, &mut res, dtype, is_hybrid, false);
    res
}

pub fn execute_add_into(a_raw: &[u8], b_raw: &[u8], res_raw: &mut [u8], dtype: DataType, is_hybrid: bool, _use_staging: bool) {
    let backend = BACKEND.get().expect("Backend not initialized");
    let device = &backend.device;
    let queue = &backend.queue;
    
    let (a_f32_backup, b_f32_backup);
    let (_a_f32, _b_f32) = if dtype == DataType::F16 && !is_hybrid {
        // GPU F16 Fallback: Convert to F32 for compute
        a_f32_backup = bytemuck::cast_slice::<u8, half::f16>(a_raw).iter().map(|x| x.to_f32()).collect::<Vec<_>>();
        b_f32_backup = bytemuck::cast_slice::<u8, half::f16>(b_raw).iter().map(|x| x.to_f32()).collect::<Vec<_>>();
        (a_f32_backup.as_slice(), b_f32_backup.as_slice())
    } else if dtype == DataType::F32 {
        (bytemuck::cast_slice(a_raw), bytemuck::cast_slice(b_raw))
    } else {
        // Hybrid F16: Will handle CPU/GPU parts specifically
        (&[][..], &[][..])
    };

    let bytes_per_element = if dtype == DataType::F32 { 4 } else { 2 };
    let total_elements = if dtype == DataType::F32 { a_raw.len() / 4 } else { a_raw.len() / 2 };
    let gpu_ratio: f64 = if is_hybrid { 0.7 } else { 1.0 };
    let gpu_elements = (total_elements as f64 * gpu_ratio) as usize;
    let cpu_elements = total_elements - gpu_elements;
    let gpu_bytes = gpu_elements * bytes_per_element;

    let (a_gpu, a_cpu) = a_raw.split_at(gpu_bytes);
    let (b_gpu, b_cpu) = b_raw.split_at(gpu_bytes);
    let (res_gpu, res_cpu) = res_raw.split_at_mut(gpu_bytes);

    std::thread::scope(|s| {
        if is_hybrid && cpu_elements > 0 {
            let a_cpu_bytes = a_cpu; // Bind to local to satisfy closure
            let b_cpu_bytes = b_cpu;
            let res_cpu_bytes = &mut *res_cpu;
            s.spawn(move || {
                if dtype == DataType::F32 {
                    let a_cpu_f32: &[f32] = bytemuck::cast_slice(a_cpu_bytes);
                    let b_cpu_f32: &[f32] = bytemuck::cast_slice(b_cpu_bytes);
                    let res_cpu_f32: &mut [f32] = bytemuck::cast_slice_mut(res_cpu_bytes);
                    res_cpu_f32.par_iter_mut().zip(a_cpu_f32.par_iter()).zip(b_cpu_f32.par_iter()).for_each(|((c, &a), &b)| *c = a + b);
                } else {
                    let a_cpu_f16: &[half::f16] = bytemuck::cast_slice(a_cpu_bytes);
                    let b_cpu_f16: &[half::f16] = bytemuck::cast_slice(b_cpu_bytes);
                    let res_cpu_f16: &mut [half::f16] = bytemuck::cast_slice_mut(res_cpu_bytes);
                    res_cpu_f16.par_iter_mut().zip(a_cpu_f16.par_iter()).zip(b_cpu_f16.par_iter()).for_each(|((c, &a), &b)| *c = half::f16::from_f32(a.to_f32() + b.to_f32()));
                }
            });
        }

        if gpu_elements > 0 {
            // GPU Path ALWAYS uses F32 compute for now (R7 200 compat)
            let (a_gpu_f32, b_gpu_f32);
            let (ag, bg) = if dtype == DataType::F16 {
                a_gpu_f32 = bytemuck::cast_slice::<u8, half::f16>(a_gpu).iter().map(|x| x.to_f32()).collect::<Vec<_>>();
                b_gpu_f32 = bytemuck::cast_slice::<u8, half::f16>(b_gpu).iter().map(|x| x.to_f32()).collect::<Vec<_>>();
                (a_gpu_f32.as_slice(), b_gpu_f32.as_slice())
            } else {
                (bytemuck::cast_slice(a_gpu), bytemuck::cast_slice(b_gpu))
            };
            
            let mut res_gpu_f32 = vec![0.0f32; gpu_elements];
            
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
                    while rx.try_recv().is_err() { device.poll(wgpu::Maintain::Poll); }
                    {
                        let data = staging_bufs[s_idx].slice(.. (elements * 4) as u64).get_mapped_range();
                        res_gpu_f32[start_offset .. start_offset + elements].copy_from_slice(bytemuck::cast_slice(&data));
                    }
                    staging_bufs[s_idx].unmap();
                }

                queue.write_buffer(&buf_a[stage], 0, bytemuck::cast_slice(&ag[chunk_start..chunk_start + current_elements]));
                queue.write_buffer(&buf_b[stage], 0, bytemuck::cast_slice(&bg[chunk_start..chunk_start + current_elements]));

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
                    res_gpu_f32[start_offset .. start_offset + elements].copy_from_slice(bytemuck::cast_slice(&data));
                }
                staging_bufs[s_idx].unmap();
            }
            
            // Final Cast back to F16 if needed
            if dtype == DataType::F16 {
                let res_f16 = bytemuck::cast_slice_mut::<u8, f16>(res_gpu);
                for i in 0..gpu_elements {
                    res_f16[i] = half::f16::from_f32(res_gpu_f32[i]);
                }
            } else {
                let res_f32 = bytemuck::cast_slice_mut::<u8, f32>(res_gpu);
                res_f32[..gpu_elements].copy_from_slice(&res_gpu_f32);
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

pub fn execute_matmul(a_raw: &[u8], b_raw: &[u8], m: u32, k: u32, n: u32, dtype: DataType, is_hybrid: bool) -> Vec<u8> {
    let backend = BACKEND.get().expect("Backend not init");
    let device = &backend.device;
    let queue = &backend.queue;
    let total_elements = (m * n) as usize;
    
    // Hybrid/CPU-only F16 handled by gemm crate in tensor.rs? 
    // Wait, tensor.rs calls this FOR GPU/Hybrid.
    
    // Convert to F32 for GPU compute (fallback)
    let a_f32_gpu: Vec<f32>;
    let b_f32_gpu: Vec<f32>;
    let (ag, bg) = if dtype == DataType::F16 {
        a_f32_gpu = bytemuck::cast_slice::<u8, half::f16>(a_raw).iter().map(|x| x.to_f32()).collect();
        b_f32_gpu = bytemuck::cast_slice::<u8, half::f16>(b_raw).iter().map(|x| x.to_f32()).collect();
        (a_f32_gpu.as_slice(), b_f32_gpu.as_slice())
    } else {
        (bytemuck::cast_slice(a_raw), bytemuck::cast_slice(b_raw))
    };

    let mut result_f32 = vec![0.0f32; total_elements];
    let res_ptr = result_f32.as_mut_ptr() as usize;

    // Adaptive tiling
    let (block_m, block_k, block_n) = if m <= 16 {
        (16, 16384, 2048) // v2.8.17: "The Union". forced N-tiling for GEMV to enable Double Buffering.
    } else {
        (512, 16384, 1024) // 512x16384 A (32MB), 16384x1024 B (64MB). Safe for 2GB VRAM.
    };
    
    let total_m_blocks = (m + block_m - 1) / block_m;
    let next_m_start = std::sync::atomic::AtomicU32::new(0);
    let completed_blocks = std::sync::atomic::AtomicU32::new(0);
    let n_usize = n as usize;
    let k_usize = k as usize;

    std::thread::scope(|s| {
        if is_hybrid {
            s.spawn(|| {
                let res_ptr = res_ptr as *mut f32;
                loop {
                    let m_start = next_m_start.fetch_add(block_m, std::sync::atomic::Ordering::Relaxed);
                    if m_start >= m { break; }
                    let bm = std::cmp::min(block_m, m - m_start);
                    let band_offset = (m_start * n) as usize;
                    
                    if dtype == DataType::F32 {
                        let a_f32: &[f32] = bytemuck::cast_slice(a_raw);
                        let b_f32: &[f32] = bytemuck::cast_slice(b_raw);
                        let a_offset = (m_start * k) as usize;
                        unsafe {
                            matrixmultiply::sgemm(
                                bm as usize, k_usize, n_usize, 1.0,
                                a_f32.as_ptr().add(a_offset), k as isize, 1,
                                b_f32.as_ptr(), n as isize, 1, 0.0,
                                res_ptr.add(band_offset), n as isize, 1,
                            );
                        }
                    } else {
                        // Hybrid F16 CPU path using gemm crate (already tested to be fast)
                        let _a_f16: &[f16] = bytemuck::cast_slice(a_raw);
                        let _b_f16: &[f16] = bytemuck::cast_slice(b_raw);
                        let a_offset = (m_start * k) as usize;
                        unsafe {
                        // Use a temporary F16 buffer for gemm then cast to F32
                        let mut temp_res_f16 = vec![gemm::f16::ZERO; bm as usize * n_usize];
                        let a_ptr = (a_raw.as_ptr() as *const gemm::f16).add(a_offset);
                        gemm::gemm(
                            bm as usize, n_usize, k_usize,
                            temp_res_f16.as_mut_ptr(), n as isize, 1,
                            false,
                            a_ptr, k as isize, 1,
                            b_raw.as_ptr() as *const gemm::f16, n as isize, 1,
                            gemm::f16::ONE, gemm::f16::ZERO,
                            false, false, false,
                            gemm::Parallelism::Rayon(rayon::current_num_threads()),
                        );
                        let res_f32_ptr = res_ptr.add(band_offset) as *mut f32;
                        let temp_res_h16: &[half::f16] = bytemuck::cast_slice(&temp_res_f16);
                        for i in 0..temp_res_h16.len() {
                            *res_f32_ptr.add(i) = f32::from(temp_res_h16[i]);
                        }
                        }
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
                    
                    let cur_a_len = (bm * bk) as usize;
                    if bm == 1 {
                        unsafe { std::ptr::copy_nonoverlapping(ag.as_ptr().add((m_start * k + k_start) as usize), a_block.as_mut_ptr(), cur_a_len); }
                    } else {
                        a_block[..cur_a_len].par_chunks_mut(bk as usize).enumerate().for_each(|(i, chunk)| {
                            let a_row_start = ( (m_start + i as u32) * k + k_start) as usize;
                            chunk.copy_from_slice(&ag[a_row_start .. a_row_start + (bk as usize)]);
                        });
                    }
                    queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(&a_block[..cur_a_len]));

                    let bufs_b = [
                        get_buffer(device, (bk * block_n * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, Some("MatMul B0")),
                        get_buffer(device, (bk * block_n * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, Some("MatMul B1")),
                    ];

                    let n_steps: Vec<u32> = (0..n).step_by(block_n as usize).collect();
                    let mut staging_requests = Vec::new();

                    for (ni, &n_start) in n_steps.iter().enumerate() {
                        let bn = std::cmp::min(block_n, n - n_start);
                        let p_idx = ni % 2;
                        let next_p_idx = (ni + 1) % 2;
                        let size_c = (bm * bn * 4) as u64;
                        let band_offset = (m_start * n) as usize;

                        if ni == 0 {
                            let cur_b_len = (bk * bn) as usize;
                            b_block[..cur_b_len].par_chunks_mut(bn as usize).enumerate().for_each(|(i, chunk)| {
                                let b_row_start = ( (k_start + i as u32) * n + n_start) as usize;
                                chunk.copy_from_slice(&bg[b_row_start .. b_row_start + (bn as usize)]);
                            });
                            queue.write_buffer(&bufs_b[p_idx], 0, bytemuck::cast_slice(&b_block[..cur_b_len]));
                        }

                        if let Some(&next_n_start) = n_steps.get(ni + 1) {
                            let nbn = std::cmp::min(block_n, n - next_n_start);
                            let next_b_len = (bk * nbn) as usize;
                            b_block[..next_b_len].par_chunks_mut(nbn as usize).enumerate().for_each(|(i, chunk)| {
                                let b_row_start = ( (k_start + i as u32) * n + next_n_start) as usize;
                                chunk.copy_from_slice(&bg[b_row_start .. b_row_start + (nbn as usize)]);
                            });
                            queue.write_buffer(&bufs_b[next_p_idx], 0, bytemuck::cast_slice(&b_block[..next_b_len]));
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
                            queue.submit(Some(encoder.finish()));
                        }

                        recycle_buffer(buf_c, size_c, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
                        recycle_buffer(buf_dims, 12, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                    }

                    if !staging_requests.is_empty() {
                        device.poll(wgpu::Maintain::Wait);
                        for (rx, staging_buf, ns, b_n, s_c, b_o) in staging_requests {
                            rx.recv().unwrap().unwrap();
                            {
                                let data = staging_buf.slice(..s_c).get_mapped_range();
                                let data_f32: &[f32] = bytemuck::cast_slice(&data);
                                unsafe {
                                    for i in 0..bm as usize {
                                        let src_row = &data_f32[i * b_n as usize .. (i + 1) * b_n as usize];
                                        let dst_row_ptr = res_ptr.add(b_o + ns as usize + i * n_usize);
                                        std::ptr::copy_nonoverlapping(src_row.as_ptr(), dst_row_ptr, b_n as usize);
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

    if dtype == DataType::F16 {
        let f16_vec: Vec<f16> = result_f32.iter().map(|&x| f16::from_f32(x)).collect();
        bytemuck::cast_slice(&f16_vec).to_vec()
    } else {
        bytemuck::cast_slice(&result_f32).to_vec()
    }
}

#[allow(dead_code)]
pub fn execute_activation(input_raw: &[u8], op: &str, dtype: DataType, is_hybrid: bool) -> Vec<u8> {
    let _total_elements = if dtype == DataType::F32 { input_raw.len() / 4 } else { input_raw.len() / 2 };
    let mut result = vec![0u8; input_raw.len()];
    execute_activation_into(input_raw, op, &mut result, dtype, is_hybrid, false);
    result
}

pub fn execute_activation_into(input_raw: &[u8], op: &str, res_raw: &mut [u8], dtype: DataType, is_hybrid: bool, _use_staging: bool) {
    let backend = BACKEND.get().expect("Backend not initialized");
    let device = &backend.device;
    let queue = &backend.queue;
    let total_elements = if dtype == DataType::F32 { input_raw.len() / 4 } else { input_raw.len() / 2 };
    
    // Performance Threshold: Avoid GPU overhead for small tensors
    if is_hybrid && total_elements < 2_000_000 {
        if dtype == DataType::F32 {
            let i_f32: &[f32] = bytemuck::cast_slice(input_raw);
            let r_f32: &mut [f32] = bytemuck::cast_slice_mut(res_raw);
            match op {
                "relu" => r_f32.par_iter_mut().zip(i_f32.par_iter()).for_each(|(o, &i)| *o = if i > 0.0 { i } else { 0.0 }),
                "sigmoid" => r_f32.par_iter_mut().zip(i_f32.par_iter()).for_each(|(o, &i)| *o = 1.0 / (1.0 + (-i).exp())),
                "silu" => r_f32.par_iter_mut().zip(i_f32.par_iter()).for_each(|(o, &i)| *o = i * (1.0 / (1.0 + (-i).exp()))),
                _ => {}
            }
        } else {
            let i_f16: &[f16] = bytemuck::cast_slice(input_raw);
            let r_f16: &mut [f16] = bytemuck::cast_slice_mut(res_raw);
            match op {
                "relu" => r_f16.par_iter_mut().zip(i_f16.par_iter()).for_each(|(o, &i)| *o = if i.to_f32() > 0.0 { i } else { f16::ZERO }),
                "sigmoid" => r_f16.par_iter_mut().zip(i_f16.par_iter()).for_each(|(o, &i)| *o = f16::from_f32(1.0 / (1.0 + (-i.to_f32()).exp()))),
                "silu" => r_f16.par_iter_mut().zip(i_f16.par_iter()).for_each(|(o, &i)| *o = f16::from_f32(i.to_f32() * (1.0 / (1.0 + (-i.to_f32()).exp())))),
                _ => {}
            }
        }
        return;
    }

    let bytes_per_element = if dtype == DataType::F32 { 4 } else { 2 };
    let gpu_ratio: f64 = if is_hybrid { 0.7 } else { 1.0 };
    let gpu_elements = (total_elements as f64 * gpu_ratio) as usize;
    let cpu_elements = total_elements - gpu_elements;
    let gpu_bytes = gpu_elements * bytes_per_element;

    let (i_gpu, i_cpu) = input_raw.split_at(gpu_bytes);
    let (res_gpu, res_cpu) = res_raw.split_at_mut(gpu_bytes);

    std::thread::scope(|s| {
        if cpu_elements > 0 && is_hybrid {
            let i_cpu_loc = i_cpu;
            let res_cpu_loc = &mut *res_cpu;
            s.spawn(move || {
                if dtype == DataType::F32 {
                    let i_cpu_f32: &[f32] = bytemuck::cast_slice(i_cpu_loc);
                    let r_cpu_f32: &mut [f32] = bytemuck::cast_slice_mut(res_cpu_loc);
                    match op {
                        "relu" => r_cpu_f32.par_iter_mut().zip(i_cpu_f32.par_iter()).for_each(|(o, &i)| *o = if i > 0.0 { i } else { 0.0 }),
                        "sigmoid" => r_cpu_f32.par_iter_mut().zip(i_cpu_f32.par_iter()).for_each(|(o, &i)| *o = 1.0 / (1.0 + (-i).exp())),
                        "silu" => r_cpu_f32.par_iter_mut().zip(i_cpu_f32.par_iter()).for_each(|(o, &i)| *o = i * (1.0 / (1.0 + (-i).exp()))),
                        _ => {}
                    }
                } else {
                    let i_cpu_f16: &[half::f16] = bytemuck::cast_slice(i_cpu_loc);
                    let r_cpu_f16: &mut [half::f16] = bytemuck::cast_slice_mut(res_cpu_loc);
                    match op {
                        "relu" => r_cpu_f16.par_iter_mut().zip(i_cpu_f16.par_iter()).for_each(|(o, &i)| *o = if i.to_f32() > 0.0 { i } else { half::f16::ZERO }),
                        "sigmoid" => r_cpu_f16.par_iter_mut().zip(i_cpu_f16.par_iter()).for_each(|(o, &i)| *o = half::f16::from_f32(1.0 / (1.0 + (-i.to_f32()).exp()))),
                        "silu" => r_cpu_f16.par_iter_mut().zip(i_cpu_f16.par_iter()).for_each(|(o, &i)| *o = half::f16::from_f32(i.to_f32() * (1.0 / (1.0 + (-i.to_f32()).exp())))),
                        _ => {}
                    }
                }
            });
        }

        if gpu_elements > 0 {
            // GPU Fallback to F32
            let i_gpu_f32 = if dtype == DataType::F16 {
                bytemuck::cast_slice::<u8, f16>(i_gpu).iter().map(|&x| x.to_f32()).collect::<Vec<f32>>()
            } else {
                bytemuck::cast_slice::<u8, f32>(i_gpu).to_vec()
            };
            
            let mut r_gpu_f32 = vec![0.0f32; gpu_elements];
            let pipeline = match op {
                "relu" => &backend.relu_pipeline,
                "sigmoid" => &backend.sigmoid_pipeline,
                "silu" => &backend.silu_pipeline,
                _ => panic!("Unknown activation: {}", op),
            };

            // Use multi-stage path for simplicity in fallback
            let chunk_size = 1024 * 1024;
            let num_stages = (gpu_elements + chunk_size - 1) / chunk_size;
            
            for stage in 0..num_stages {
                let start = stage * chunk_size;
                let elements = std::cmp::min(chunk_size, gpu_elements - start);
                let cur_size = (elements * 4) as wgpu::BufferAddress;
                
                let bi = get_buffer(device, cur_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, None);
                let bo = get_buffer(device, cur_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, None);
                
                queue.write_buffer(&bi, 0, bytemuck::cast_slice(&i_gpu_f32[start..start+elements]));
                
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
                    r_gpu_f32[start .. start + elements].copy_from_slice(bytemuck::cast_slice(&data));
                }
                sb.unmap();
                recycle_buffer(bi, cur_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
                recycle_buffer(bo, cur_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
                recycle_buffer(sb, cur_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            }
            
            // Cast back
            if dtype == DataType::F16 {
                let r_f16 = bytemuck::cast_slice_mut::<u8, f16>(res_gpu);
                for i in 0..gpu_elements { r_f16[i] = f16::from_f32(r_gpu_f32[i]); }
            } else {
                let r_f32 = bytemuck::cast_slice_mut::<u8, f32>(res_gpu);
                r_f32[..gpu_elements].copy_from_slice(&r_gpu_f32);
            }
        }
    });
}
