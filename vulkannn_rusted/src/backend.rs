use std::sync::{OnceLock, Mutex};
use wgpu::{Adapter, Device, Queue};
use wgpu::util::DeviceExt;

pub struct CachedBuffer {
    pub size: wgpu::BufferAddress,
    pub usage: wgpu::BufferUsages,
    pub buffer: wgpu::Buffer,
}

pub struct WgpuBackend {
    pub device: Device,
    pub queue: Queue,
    pub adapter: Adapter,
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
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .expect("Failed to create wgpu device");

        WgpuBackend {
            device,
            queue,
            adapter,
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

pub fn execute_add(a_data: &[f32], b_data: &[f32]) -> Vec<f32> {
    let backend = BACKEND.get().expect("Backend not initialized. Did you import the python module properly?");
    let device = &backend.device;
    let queue = &backend.queue;

    let size = (a_data.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let usage_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let buf_a = get_buffer(device, size, usage_storage, Some("Buf A"));
    let buf_b = get_buffer(device, size, usage_storage, Some("Buf B"));
    let buf_c = get_buffer(device, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, Some("Buf C"));
    let staging_buf = get_buffer(device, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, Some("Staging Buffer"));

    queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(a_data));
    queue.write_buffer(&buf_b, 0, bytemuck::cast_slice(b_data));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Add Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/add.wgsl").into()),
    });
    
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Add Pipeline"),
        layout: None,
        module: &shader,
        entry_point: "main",
    });
    
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
        let total_workgroups = ((a_data.len() as u32) + 63) / 64;
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
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buf.unmap();
    
    recycle_buffer(buf_a, size, usage_storage);
    recycle_buffer(buf_b, size, usage_storage);
    recycle_buffer(buf_c, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    recycle_buffer(staging_buf, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);

    result
}

// ---------------------------------------------------------
// MATRIX MULTIPLICATION
// ---------------------------------------------------------
pub fn execute_matmul(a_data: &[f32], b_data: &[f32], m: u32, k: u32, n: u32) -> Vec<f32> {
    let backend = BACKEND.get().expect("Backend not init");
    let device = &backend.device;
    let queue = &backend.queue;

    let size_a = (m * k * 4) as wgpu::BufferAddress;
    let size_b = (k * n * 4) as wgpu::BufferAddress;
    let size_c = (m * n * 4) as wgpu::BufferAddress;

    let usage_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let buf_a = get_buffer(device, size_a, usage_storage, Some("MatMul A"));
    let buf_b = get_buffer(device, size_b, usage_storage, Some("MatMul B"));
    let buf_c = get_buffer(device, size_c, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, Some("MatMul C"));
    let staging_buf = get_buffer(device, size_c, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, Some("Staging Buffer C"));

    queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(a_data));
    queue.write_buffer(&buf_b, 0, bytemuck::cast_slice(b_data));

    let dims: [u32; 3] = [m, k, n];
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
        let wg_y = (m + 15) / 16;
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
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buf.unmap();
    
    recycle_buffer(buf_a, size_a, usage_storage);
    recycle_buffer(buf_b, size_b, usage_storage);
    recycle_buffer(buf_c, size_c, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    recycle_buffer(staging_buf, size_c, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    recycle_buffer(buf_dims, std::mem::size_of::<[u32; 3]>() as wgpu::BufferAddress, usage_uniform);

    result
}

// ---------------------------------------------------------
// NON-LINEAR ACTIVATIONS
// ---------------------------------------------------------
pub fn execute_activation(data: &[f32], activation_type: &str) -> Vec<f32> {
    let backend = BACKEND.get().expect("Backend not init");
    let device = &backend.device;
    let queue = &backend.queue;

    let size = (data.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let usage_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let buf_in = get_buffer(device, size, usage_storage, Some("Activation In"));
    let buf_out = get_buffer(device, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, Some("Activation Out"));
    let staging_buf = get_buffer(device, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, Some("Staging Buffer Activation"));

    queue.write_buffer(&buf_in, 0, bytemuck::cast_slice(data));
    
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Activation Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/activation.wgsl").into()),
    });
    
    let entry_point = match activation_type {
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
    });
    
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
        let total_workgroups = ((data.len() as u32) + 63) / 64;
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
    let result: Vec<f32> = bytemuck::cast_slice(&res_data).to_vec();
    drop(res_data);
    staging_buf.unmap();
    
    recycle_buffer(buf_in, size, usage_storage);
    recycle_buffer(buf_out, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    recycle_buffer(staging_buf, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);

    result
}
