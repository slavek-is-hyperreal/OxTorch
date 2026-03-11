use std::ffi::{CStr, CString};
use std::sync::{Mutex, OnceLock};
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc, Allocation, AllocationCreateDesc};
use gpu_allocator::MemoryLocation;
use crate::tensor::DataType;
use crate::avx_swar::*;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct AsyncOp {
    pub staging_in1: Option<CachedBuffer>,
    pub staging_in2: Option<CachedBuffer>,
    pub staging_out: Option<CachedBuffer>,
    pub device_in1: Option<CachedBuffer>,
    pub device_in2: Option<CachedBuffer>,
    pub device_out: Option<CachedBuffer>,
    pub cmd_buffer: vk::CommandBuffer,
    pub desc_set: vk::DescriptorSet,
    pub wait_id: u64,
}

pub struct AshBackend {
    pub _entry: ash::Entry,
    pub _instance: ash::Instance,
    pub _pdevice: vk::PhysicalDevice,
    pub device: ash::Device,
    pub compute_queue: vk::Queue,
    pub _compute_family: u32,
    #[allow(dead_code)]
    pub transfer_queue: vk::Queue,
    #[allow(dead_code)]
    pub _transfer_family: u32,
    pub allocator: Mutex<Allocator>,
    
    pub desc_pool: Mutex<vk::DescriptorPool>,
    pub dsl_add: vk::DescriptorSetLayout,
    pub dsl_matmul: vk::DescriptorSetLayout,
    pub dsl_act: vk::DescriptorSetLayout,
    
    pub pipe_layout_add: vk::PipelineLayout,
    pub pipe_layout_matmul: vk::PipelineLayout,
    pub pipe_layout_act: vk::PipelineLayout,
    
    pub pipe_add: vk::Pipeline,
    pub pipe_matmul: vk::Pipeline,
    pub pipe_relu: vk::Pipeline,
    pub pipe_sigmoid: vk::Pipeline,
    pub pipe_silu: vk::Pipeline,

    pub compute_cmd_pool: vk::CommandPool,
    #[allow(dead_code)]
    pub transfer_cmd_pool: vk::CommandPool,
    pub buffer_cache: Mutex<Vec<CachedBuffer>>,

    // Bug #3 fix: Pre-allocated permanent descriptor sets.
    // Instead of allocating+freeing a DescriptorSet on EVERY op call (which fragments the pool
    // and causes bimodal latency), we keep 3 permanent sets and just call UpdateDescriptorSets.
    // Protected by Mutex since GPU ops are serialized via queue_wait_idle anyway.
    pub perm_desc_matmul: Mutex<vk::DescriptorSet>,
    pub perm_desc_add: Mutex<vk::DescriptorSet>,
    pub perm_desc_act: Mutex<vk::DescriptorSet>,
    
    pub timeline_semaphore: vk::Semaphore,
    pub timeline_value: AtomicU64,
    pub pending_ops: Mutex<Vec<AsyncOp>>,
}

pub struct CachedBuffer {
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub cpu_visible: bool,
    pub mapped_ptr: Option<*mut u8>,
}

unsafe impl Send for CachedBuffer {}
unsafe impl Sync for CachedBuffer {}

pub static BACKEND: OnceLock<AshBackend> = OnceLock::new();

pub fn init_backend() {
    BACKEND.get_or_init(|| {
        println!("[vulkannn_rusted v{}] Initializing Raw Ash Vulkan Engine...", env!("CARGO_PKG_VERSION"));
        let entry = unsafe { ash::Entry::load().expect("Failed to load Vulkan library") };

        let app_name = CString::new("VulkanNN Rusted").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .api_version(vk::make_api_version(0, 1, 2, 0)); // Vulkan 1.2 required for timeline semaphores

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        let instance = unsafe { entry.create_instance(&instance_create_info, None).expect("Instance creation failed") };

        let pdevices = unsafe { instance.enumerate_physical_devices().expect("No physical devices found") };
        let pdevice = pdevices.into_iter().find(|&pd| {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU || props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
                println!("[vulkannn_rusted] Selected Physical Device: {:?}", name);
                true
            } else { false }
        }).expect("No suitable GPU found");

        let queue_family_properties = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };
        
        let mut compute_family = None;
        let mut transfer_family = None;

        for (i, prop) in queue_family_properties.iter().enumerate() {
            if prop.queue_flags.contains(vk::QueueFlags::COMPUTE) && !prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                compute_family = Some(i as u32);
            } else if prop.queue_flags.contains(vk::QueueFlags::COMPUTE) && compute_family.is_none() {
                compute_family = Some(i as u32);
            }

            if prop.queue_flags.contains(vk::QueueFlags::TRANSFER) && !prop.queue_flags.contains(vk::QueueFlags::COMPUTE) && !prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                transfer_family = Some(i as u32);
            }
        }
        
        // Fallbacks
        let compute_family = compute_family.expect("No compute queue family found");
        let transfer_family = transfer_family.unwrap_or(compute_family);

        let priorities = [1.0f32];
        let mut queue_create_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(compute_family)
                .queue_priorities(&priorities)
        ];
        if transfer_family != compute_family {
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(transfer_family)
                    .queue_priorities(&priorities)
            );
        }

        let ext_timeline = CStr::from_bytes_with_nul(b"VK_KHR_timeline_semaphore\0").unwrap().as_ptr();

        let mut device_extension_names_raw = vec![ext_timeline];
        let mut has_fp16_ext = false;
        
        if let Ok(device_exts) = unsafe { instance.enumerate_device_extension_properties(pdevice) } {
            for ext in device_exts {
                let name_bytes = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                if let Ok(name_str) = name_bytes.to_str() {
                    if name_str == "VK_KHR_shader_float16_int8" {
                        has_fp16_ext = true;
                        break;
                    }
                }
            }
        }

        let mut has_fp16 = false;
        if has_fp16_ext {
            let mut supported_features16 = vk::PhysicalDeviceShaderFloat16Int8FeaturesKHR::default();
            let mut supported_features2 = vk::PhysicalDeviceFeatures2::default();
            supported_features2.p_next = &mut supported_features16 as *mut _ as *mut std::ffi::c_void;
            unsafe { instance.get_physical_device_features2(pdevice, &mut supported_features2) };
            
            if supported_features16.shader_float16 == vk::TRUE {
                has_fp16 = true;
                println!("[vulkannn_rusted] Found Advanced GPU: VK_KHR_shader_float16_int8 and shader_float16 enabled");
            }
        }

        if has_fp16 {
            device_extension_names_raw.push(CStr::from_bytes_with_nul(b"VK_KHR_shader_float16_int8\0").unwrap().as_ptr());
        }

        let features = vk::PhysicalDeviceFeatures::default();
        
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true);
            
        let mut features11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true);

        // Chain features: features2 -> features12 -> features11 -> [features16 (if available)]
        features12.p_next = &mut features11 as *mut _ as *mut std::ffi::c_void;

        let mut features16 = vk::PhysicalDeviceShaderFloat16Int8FeaturesKHR::default()
            .shader_float16(true);

        #[allow(unused_assignments)]
        if has_fp16 {
            features11.p_next = &mut features16 as *mut _ as *mut std::ffi::c_void;
        }

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(features);
        features2.p_next = &mut features12 as *mut _ as *mut std::ffi::c_void;

        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extension_names_raw);
        device_create_info.p_next = &mut features2 as *mut _ as *mut std::ffi::c_void;

        let device = unsafe { instance.create_device(pdevice, &device_create_info, None) }.expect("Device creation failed");

        let compute_queue = unsafe { device.get_device_queue(compute_family, 0) };
        let transfer_queue = unsafe { device.get_device_queue(transfer_family, 0) };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: pdevice,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        }).unwrap();

        // Bug #3 fix: Do NOT use FREE_DESCRIPTOR_SET_BIT.
        // Per Vulkan spec and vkguide.dev best practices: omitting this flag lets the driver
        // use a simpler bump allocator internally, which is faster and doesn't fragment.
        // We only need a small fixed pool (3 permanent sets + headroom for async activation ops).
        let pool_sizes = [
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(64),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(16),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(32)
            .pool_sizes(&pool_sizes);
        let desc_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }.unwrap();

        // DSL Add: 3 Storage
        let bindings_add = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_add = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings_add), None) }.unwrap();

        // DSL Matmul: 3 Storage, 1 Uniform
        let bindings_matmul = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_matmul = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings_matmul), None) }.unwrap();

        // DSL Act: 2 Storage
        let bindings_act = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_act = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings_act), None) }.unwrap();

        let pipe_layout_add = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_add]), None) }.unwrap();
        let pipe_layout_matmul = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_matmul]), None) }.unwrap();
        let pipe_layout_act = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_act]), None) }.unwrap();

        let load_shader = |bytes: &[u8]| -> vk::ShaderModule {
            let mut cursor = std::io::Cursor::new(bytes);
            let code = ash::util::read_spv(&mut cursor).expect("Failed to read struct spv");
            let info = vk::ShaderModuleCreateInfo::default().code(&code);
            unsafe { device.create_shader_module(&info, None) }.unwrap()
        };

        let sm_add = load_shader(include_bytes!("shaders/add.wgsl.spv"));
        let sm_matmul = load_shader(include_bytes!("shaders/matmul.wgsl.spv"));
        let sm_act = load_shader(include_bytes!("shaders/activation.wgsl.spv"));

        let entry_main = CString::new("main").unwrap();
        let entry_relu = CString::new("relu_main").unwrap();
        let entry_sigm = CString::new("sigmoid_main").unwrap();
        let entry_silu = CString::new("silu_main").unwrap();

        let create_pipe = |sm: vk::ShaderModule, entry: &CStr, layout: vk::PipelineLayout| -> vk::Pipeline {
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(sm)
                .name(entry);
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) }.unwrap()[0]
        };

        let pipe_add = create_pipe(sm_add, &entry_main, pipe_layout_add);
        let pipe_matmul = create_pipe(sm_matmul, &entry_main, pipe_layout_matmul);
        let pipe_relu = create_pipe(sm_act, &entry_relu, pipe_layout_act);
        let pipe_sigmoid = create_pipe(sm_act, &entry_sigm, pipe_layout_act);
        let pipe_silu = create_pipe(sm_act, &entry_silu, pipe_layout_act);

        unsafe {
            device.destroy_shader_module(sm_add, None);
            device.destroy_shader_module(sm_matmul, None);
            device.destroy_shader_module(sm_act, None);
        }

        let compute_cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(compute_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let compute_cmd_pool = unsafe { device.create_command_pool(&compute_cmd_pool_info, None) }.unwrap();

        let transfer_cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(transfer_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let transfer_cmd_pool = unsafe { device.create_command_pool(&transfer_cmd_pool_info, None) }.unwrap();

        let mut sem_type = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut sem_type);
        let timeline_semaphore = unsafe { device.create_semaphore(&sem_info, None) }.expect("Failed to create Timeline Semaphore. Ensure device supports Vulkan 1.2!");

        // Bug #3 fix: Allocate 3 permanent descriptor sets once at startup.
        // From this point on, ops call UpdateDescriptorSets to re-point them at new buffers.
        let perm_sets = unsafe {
            let layouts = [dsl_matmul, dsl_add, dsl_act];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(&layouts);
            device.allocate_descriptor_sets(&alloc_info).expect("Failed to pre-alloc permanent descriptor sets")
        };
        let perm_desc_matmul = perm_sets[0];
        let perm_desc_add    = perm_sets[1];
        let perm_desc_act    = perm_sets[2];

        AshBackend {
            _entry: entry, _instance: instance, _pdevice: pdevice, device,
            compute_queue, _compute_family: compute_family,
            transfer_queue, _transfer_family: transfer_family,
            allocator: Mutex::new(allocator),
            desc_pool: Mutex::new(desc_pool),
            dsl_add, dsl_matmul, dsl_act,
            pipe_layout_add, pipe_layout_matmul, pipe_layout_act,
            pipe_add, pipe_matmul, pipe_relu, pipe_sigmoid, pipe_silu,
            compute_cmd_pool, transfer_cmd_pool,
            buffer_cache: Mutex::new(Vec::new()),
            perm_desc_matmul: Mutex::new(perm_desc_matmul),
            perm_desc_add:    Mutex::new(perm_desc_add),
            perm_desc_act:    Mutex::new(perm_desc_act),
            timeline_semaphore,
            timeline_value: AtomicU64::new(0),
            pending_ops: Mutex::new(Vec::new()),
        }
    });
}

pub fn poll_async_ops() {
    let backend = BACKEND.get().unwrap();
    let current_val = unsafe { backend.device.get_semaphore_counter_value(backend.timeline_semaphore).unwrap() };
    
    let mut pending = backend.pending_ops.lock().unwrap();
    pending.retain_mut(|op| {
        if current_val >= op.wait_id {
            // Operation finished, cleanup and recycle buffers
            if let Some(buf) = op.staging_in1.take() { recycle_buffer(buf); }
            if let Some(buf) = op.staging_in2.take() { recycle_buffer(buf); }
            if let Some(buf) = op.staging_out.take() { recycle_buffer(buf); }
            if let Some(buf) = op.device_in1.take() { recycle_buffer(buf); }
            if let Some(buf) = op.device_in2.take() { recycle_buffer(buf); }
            if let Some(buf) = op.device_out.take() { recycle_buffer(buf); }
            
            unsafe {
                backend.device.free_command_buffers(backend.compute_cmd_pool, &[op.cmd_buffer]);
                // Bug #3 fix: desc_set is now a permanent set — do NOT free it.
                // We store vk::DescriptorSet::null() in AsyncOp to mark this.
                if op.desc_set != vk::DescriptorSet::null() {
                    backend.device.free_descriptor_sets(*backend.desc_pool.lock().unwrap(), &[op.desc_set]).unwrap();
                }
            }
            false // remove from pending
        } else {
            true // keep in pending
        }
    });
}

pub fn poll_async_ops_until(target_val: u64) {
    let backend = BACKEND.get().unwrap();
    let current = unsafe { backend.device.get_semaphore_counter_value(backend.timeline_semaphore).unwrap() };
    if current < target_val {
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&backend.timeline_semaphore))
            .values(std::slice::from_ref(&target_val));
        unsafe { backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap(); }
    }
    poll_async_ops();
}

pub fn get_buffer(size: vk::DeviceSize, usage: vk::BufferUsageFlags, label: Option<&str>, cpu_visible: bool) -> CachedBuffer {
    let backend = BACKEND.get().unwrap();
    if let Ok(mut cache) = backend.buffer_cache.lock() {
        if let Some(idx) = cache.iter().position(|b| b.size >= size && b.usage.contains(usage) && b.cpu_visible == cpu_visible) {
            let cached = cache.swap_remove(idx);
            return cached;
        }
    }
    
    let buffer_info = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { backend.device.create_buffer(&buffer_info, None) }.unwrap();
    
    let requirements = unsafe { backend.device.get_buffer_memory_requirements(buffer) };
    
    // Safety: Only use CpuToGpu for explicitly requested host-visible buffers (staging).
    // Internal buffers stay GpuOnly to avoid coherency/BAR issues on legacy AMD cards.
    let location = if cpu_visible { 
        MemoryLocation::CpuToGpu 
    } else { 
        MemoryLocation::GpuOnly 
    };
    
    let mut retry_count = 0;
    let allocation = loop {
        let result = backend.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name: label.unwrap_or("Buffer"),
            requirements,
            location,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        });

        match result {
            Ok(alloc) => break alloc,
            Err(e) => {
                if retry_count == 0 {
                    println!("[VNN] VRAM Allocation failed: {:?}. Clearing cache and retrying...", e);
                    clear_all_caches();
                    retry_count += 1;
                    continue;
                } else {
                    panic!("[VNN] CRITICAL: VRAM Allocation failed even after clearing cache: {:?}", e);
                }
            }
        }
    };
    
    let cpu_visible_actual = allocation.mapped_ptr().is_some();
    let mapped_ptr = allocation.mapped_ptr().map(|p| p.as_ptr() as *mut u8);
    
    unsafe { backend.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }.unwrap();
    
    CachedBuffer { size, usage, buffer, allocation: Some(allocation), cpu_visible: cpu_visible_actual, mapped_ptr }
}

pub fn recycle_buffer(cached: CachedBuffer) {
    let backend = BACKEND.get().unwrap();
    if let Ok(mut cache) = backend.buffer_cache.lock() {
        // Simple safety: if cache grows too large (> 512MB), start clearing it
        let current_total: u64 = cache.iter().map(|b| b.size).sum();
        if current_total > 512 * 1024 * 1024 {
            // Prune OLD buffers when recycling new ones into a full cache
            prune_buffer_cache(2); // Remove 2 oldest
        }
        cache.push(cached);
    }
}

pub fn prune_buffer_cache(count: usize) {
    let backend = BACKEND.get().unwrap();
    if let Ok(mut cache) = backend.buffer_cache.lock() {
        let to_remove = count.min(cache.len());
        // Bug #1 fix: was cache.remove(0) = O(n) shift of entire Vec.
        // pop() = O(1). LIFO order is fine for cache eviction — we evict
        // the most-recently-recycled buffer (hottest in terms of allocation).
        for _ in 0..to_remove {
            if let Some(buf) = cache.pop() {
                destroy_cached_buffer(buf);
            }
        }
    }
}

fn destroy_cached_buffer(mut buf: CachedBuffer) {
    let backend = BACKEND.get().unwrap();
    unsafe {
        backend.device.destroy_buffer(buf.buffer, None);
        if let Some(alloc) = buf.allocation.take() {
            backend.allocator.lock().unwrap().free(alloc).unwrap();
        }
    }
}

pub fn clear_all_caches() {
    let backend = BACKEND.get().unwrap();
    if let Ok(mut cache) = backend.buffer_cache.lock() {
        while !cache.is_empty() {
            let buf = cache.pop().unwrap();
            destroy_cached_buffer(buf);
        }
    }
}
pub fn execute_add(a_raw: &[u8], b_raw: &[u8], dtype: DataType, is_hybrid: bool) -> Vec<u8> {
    let mut out = vec![0u8; a_raw.len()];
    execute_add_into(a_raw, b_raw, &mut out, dtype, is_hybrid, false);
    out
}

/// Generic elementwise binary op (mul / sub / div) via Vulkan.
/// For "add", falls through to execute_add_into.
/// For mul/sub/div, uses the same pipeline pattern as add but with
/// a simple CPU-side fallback when those shaders are not yet compiled in.
pub fn execute_elementwise(a_raw: &[u8], b_raw: &[u8], op: &str, dtype: DataType, is_hybrid: bool) -> Vec<u8> {
    // For ops without dedicated Vulkan shaders yet, compute on CPU and return.
    // Sprint 4 will add dedicated shaders and remove this fallback.
    let bytes_per_elem = if dtype == DataType::F32 { 4 } else { 2 };
    let n = a_raw.len() / bytes_per_elem;
    let mut out = vec![0u8; a_raw.len()];

    if dtype == DataType::F32 {
        let a: &[f32] = bytemuck::cast_slice(a_raw);
        let b: &[f32] = bytemuck::cast_slice(b_raw);
        let c: &mut [f32] = bytemuck::cast_slice_mut(&mut out);
        match op {
            "add" => { for i in 0..n { c[i] = a[i] + b[i]; } },
            "mul" => { for i in 0..n { c[i] = a[i] * b[i]; } },
            "sub" => { for i in 0..n { c[i] = a[i] - b[i]; } },
            "div" => { for i in 0..n { c[i] = a[i] / b[i]; } },
            _ => {}
        }
    } else if dtype == DataType::F16 {
        let a: &[half::f16] = bytemuck::cast_slice(a_raw);
        let b: &[half::f16] = bytemuck::cast_slice(b_raw);
        let c: &mut [half::f16] = bytemuck::cast_slice_mut(&mut out);
        match op {
            "add" => { for i in 0..n { c[i] = half::f16::from_f32(a[i].to_f32() + b[i].to_f32()); } },
            "mul" => { for i in 0..n { c[i] = half::f16::from_f32(a[i].to_f32() * b[i].to_f32()); } },
            "sub" => { for i in 0..n { c[i] = half::f16::from_f32(a[i].to_f32() - b[i].to_f32()); } },
            "div" => { for i in 0..n { c[i] = half::f16::from_f32(a[i].to_f32() / b[i].to_f32()); } },
            _ => {}
        }
    } else {
        let a: &[half::bf16] = bytemuck::cast_slice(a_raw);
        let b: &[half::bf16] = bytemuck::cast_slice(b_raw);
        let c: &mut [half::bf16] = bytemuck::cast_slice_mut(&mut out);
        match op {
            "add" => { for i in 0..n { c[i] = half::bf16::from_f32(a[i].to_f32() + b[i].to_f32()); } },
            "mul" => { for i in 0..n { c[i] = half::bf16::from_f32(a[i].to_f32() * b[i].to_f32()); } },
            "sub" => { for i in 0..n { c[i] = half::bf16::from_f32(a[i].to_f32() - b[i].to_f32()); } },
            "div" => { for i in 0..n { c[i] = half::bf16::from_f32(a[i].to_f32() / b[i].to_f32()); } },
            _ => {}
        }
    }
    let _ = is_hybrid; // hybrid routing for Sprint 5
    out
}

fn begin_cmd(device: &ash::Device, pool: vk::CommandPool) -> vk::CommandBuffer {
    let alloc_info = vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap()[0];
    let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin_info) }.unwrap();
    cmd
}



fn upload_to_stage(src_raw: &[u8], stage: &CachedBuffer, dtype: DataType) {
    let ptr = stage.mapped_ptr.unwrap() as *mut f32;
    let num_elements = src_raw.len() / if dtype == DataType::F32 { 4 } else { 2 };
    let dst_slice = unsafe { std::slice::from_raw_parts_mut(ptr, num_elements) };

    if dtype == DataType::F16 {
        let src_slice = bytemuck::cast_slice::<u8, half::f16>(src_raw);
        convert_f16_to_f32(src_slice, dst_slice);
    } else if dtype == DataType::BF16 {
        let src_slice = bytemuck::cast_slice::<u8, half::bf16>(src_raw);
        convert_bf16_to_f32(src_slice, dst_slice);
    } else {
        unsafe { std::ptr::copy_nonoverlapping(src_raw.as_ptr(), ptr as *mut u8, src_raw.len()); }
    }
}

fn download_from_stage(dst_raw: &mut [u8], stage: &CachedBuffer, dtype: DataType) {
    let ptr = stage.mapped_ptr.unwrap() as *const f32;
    let num_elements = dst_raw.len() / if dtype == DataType::F32 { 4 } else { 2 };
    let src_slice = unsafe { std::slice::from_raw_parts(ptr, num_elements) };

    if dtype == DataType::F16 {
        let dst_slice = bytemuck::cast_slice_mut::<u8, half::f16>(dst_raw);
        convert_f32_to_f16(src_slice, dst_slice);
    } else if dtype == DataType::BF16 {
        let dst_slice = bytemuck::cast_slice_mut::<u8, half::bf16>(dst_raw);
        convert_f32_to_bf16(src_slice, dst_slice);
    } else {
        unsafe { std::ptr::copy_nonoverlapping(ptr as *const u8, dst_raw.as_mut_ptr(), dst_raw.len()); }
    }
}

pub fn execute_add_into(a_raw: &[u8], b_raw: &[u8], res_raw: &mut [u8], dtype: DataType, _is_hybrid: bool, _use_staging: bool) {
    let backend = BACKEND.get().unwrap();
    let bytes_per_elem = if dtype == DataType::F32 { 4 } else { 2 };
    let num_elements = a_raw.len() / bytes_per_elem;
    let num_bytes_f32 = (num_elements * 4) as vk::DeviceSize;

    // Use mappable buffers (CpuToGpu) for small data, GpuOnly for large.
    let buf_a = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Add_A"), false);
    let buf_b = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Add_B"), false);
    let buf_c = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Add_C"), false);

    let stage_a = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("Add_Stage_A"), true);
    let stage_b = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("Add_Stage_B"), true);
    let stage_c = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_DST, Some("Add_Stage_C"), true);

    upload_to_stage(a_raw, &stage_a, dtype);
    upload_to_stage(b_raw, &stage_b, dtype);

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_a.buffer, buf_a.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: num_bytes_f32 }]);
        backend.device.cmd_copy_buffer(cmd, stage_b.buffer, buf_b.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: num_bytes_f32 }]);
        
        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_a.buffer).size(vk::WHOLE_SIZE).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_b.buffer).size(vk::WHOLE_SIZE).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        // Bug #3 fix: Use permanent pre-allocated descriptor set — no alloc/free per call.
        let set = *backend.perm_desc_add.lock().unwrap();
        
        let info_a = [vk::DescriptorBufferInfo::default().buffer(buf_a.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let info_b = [vk::DescriptorBufferInfo::default().buffer(buf_b.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let info_c = [vk::DescriptorBufferInfo::default().buffer(buf_c.buffer).offset(0).range(vk::WHOLE_SIZE)];
        
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_a),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_b),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_c),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_add);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_add, 0, &[set], &[]);
        let workgroups = (num_elements as u32 + 255) / 256;
        backend.device.cmd_dispatch(cmd, workgroups, 1, 1);
        
        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_c.buffer).size(vk::WHOLE_SIZE).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);
        
        backend.device.cmd_copy_buffer(cmd, buf_c.buffer, stage_c.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: num_bytes_f32 }]);
        
        backend.device.end_command_buffer(cmd).unwrap();
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmds);
        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();
        backend.device.queue_wait_idle(backend.compute_queue).unwrap();
        backend.device.free_command_buffers(backend.compute_cmd_pool, &cmds);
        // NOTE: No free_descriptor_sets — permanent set is reused.
    }

    download_from_stage(res_raw, &stage_c, dtype);

    recycle_buffer(buf_a); recycle_buffer(buf_b); recycle_buffer(buf_c);
    recycle_buffer(stage_a); recycle_buffer(stage_b); recycle_buffer(stage_c);
}

pub fn execute_matmul(a_raw: &[u8], b_raw: &[u8], m: u32, k: u32, n: u32, dtype: DataType, _is_hybrid: bool) -> Vec<u8> {
    let backend = BACKEND.get().unwrap();
    let bytes_per_elem = if dtype == DataType::F32 { 4 } else { 2 };
    
    let size_a_f32 = (m * k * 4) as vk::DeviceSize;
    let size_b_f32 = (k * n * 4) as vk::DeviceSize;
    let size_c_f32 = (m * n * 4) as vk::DeviceSize;
    
    let size_c_raw = (m * n * bytes_per_elem) as vk::DeviceSize;
    let mut out_vec = vec![0u8; size_c_raw as usize];

    let buf_a = get_buffer(size_a_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_A"), false);
    let buf_b = get_buffer(size_b_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_B"), false);
    let buf_u = get_buffer(16, vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_U"), false);
    let buf_c = get_buffer(size_c_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_C"), false);

    let stage_a = get_buffer(size_a_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("MatMul_Stage_A"), true);
    let stage_b = get_buffer(size_b_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("MatMul_Stage_B"), true);
    let stage_u = get_buffer(16, vk::BufferUsageFlags::TRANSFER_SRC, Some("MatMul_Stage_U"), true);
    let stage_c = get_buffer(size_c_f32, vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_Stage_C"), true);

    let uniform_data: [u32; 4] = [m, k, n, 0];
    upload_to_stage(a_raw, &stage_a, dtype);
    upload_to_stage(b_raw, &stage_b, dtype);
    unsafe {
        let ptr = stage_u.mapped_ptr.expect("stage_u must be mapped") as *mut u32;
        std::ptr::copy_nonoverlapping(uniform_data.as_ptr(), ptr, 4);
    }

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_a.buffer, buf_a.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: size_a_f32 }]);
        backend.device.cmd_copy_buffer(cmd, stage_b.buffer, buf_b.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: size_b_f32 }]);
        backend.device.cmd_copy_buffer(cmd, stage_u.buffer, buf_u.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: 16 }]);

        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_a.buffer).size(vk::WHOLE_SIZE).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_b.buffer).size(vk::WHOLE_SIZE).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_u.buffer).size(vk::WHOLE_SIZE).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::UNIFORM_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        // Bug #3 fix: Use permanent pre-allocated descriptor set — no alloc/free per call.
        let set = *backend.perm_desc_matmul.lock().unwrap();
        
        let info_a = [vk::DescriptorBufferInfo::default().buffer(buf_a.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let info_b = [vk::DescriptorBufferInfo::default().buffer(buf_b.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let info_c = [vk::DescriptorBufferInfo::default().buffer(buf_c.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let info_u = [vk::DescriptorBufferInfo::default().buffer(buf_u.buffer).offset(0).range(vk::WHOLE_SIZE)];
        
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_a),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_b),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_c),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&info_u),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_matmul);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_matmul, 0, &[set], &[]);
        backend.device.cmd_dispatch(cmd, (n + 15) / 16, (m + 15) / 16, 1);
        
        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_c.buffer).size(vk::WHOLE_SIZE).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);
        
        backend.device.cmd_copy_buffer(cmd, buf_c.buffer, stage_c.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: size_c_f32 }]);
        
        backend.device.end_command_buffer(cmd).unwrap();
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmds);
        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();
        backend.device.queue_wait_idle(backend.compute_queue).unwrap();
        backend.device.free_command_buffers(backend.compute_cmd_pool, &cmds);
        // NOTE: No free_descriptor_sets — permanent set is reused.
    }

    download_from_stage(&mut out_vec, &stage_c, dtype);

    recycle_buffer(buf_a); recycle_buffer(buf_b); recycle_buffer(buf_c); recycle_buffer(buf_u);
    recycle_buffer(stage_a); recycle_buffer(stage_b); recycle_buffer(stage_c); recycle_buffer(stage_u);
    out_vec
}

pub fn execute_activation(input_raw: &[u8], op: &str, dtype: DataType, is_hybrid: bool) -> Vec<u8> {
    let mut out_vec = vec![0u8; input_raw.len()];
    let (wait_id, stage_out) = submit_activation_into(input_raw, op, &mut out_vec, dtype, is_hybrid, false);
    poll_async_ops_until(wait_id);
    download_from_stage(&mut out_vec, &stage_out, dtype);
    out_vec
}

/// MSTS Tile-Pulling: Process a chunk of [elem_offset..elem_offset+elem_count] elements on GPU.
/// Input/output are full-tensor byte slices; only the tile window is uploaded and computed.
/// Returns the computed bytes for this tile (always elem_count * bytes_per_elem_f32 = elem_count*4).
pub fn execute_activation_chunked(
    input_raw: &[u8],
    output_raw: &mut [u8],
    elem_offset: usize,
    elem_count: usize,
    op: &str,
    dtype: DataType,
) {
    let bytes_per_elem = if dtype == DataType::F32 { 4usize } else { 2usize };
    let tile_in  = &input_raw [elem_offset * bytes_per_elem .. (elem_offset + elem_count) * bytes_per_elem];
    let tile_out = &mut output_raw[elem_offset * bytes_per_elem .. (elem_offset + elem_count) * bytes_per_elem];

    let num_bytes_f32 = (elem_count * 4) as vk::DeviceSize;
    let backend = BACKEND.get().unwrap();

    let buf_in  = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Act_In"),  false);
    let buf_out = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Act_Out"), false);
    let stage_in  = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("Act_Stage_In"),  true);
    let stage_out = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_DST, Some("Act_Stage_Out"), true);

    upload_to_stage(tile_in, &stage_in, dtype);

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_in.buffer, buf_in.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: num_bytes_f32 }]);

        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(buf_in.buffer).size(vk::WHOLE_SIZE)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[barrier], &[]);

        // Bug #3 fix: Use permanent pre-allocated descriptor set — no alloc/free per call.
        let set = *backend.perm_desc_act.lock().unwrap();

        let info_in  = [vk::DescriptorBufferInfo::default().buffer(buf_in.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let info_out = [vk::DescriptorBufferInfo::default().buffer(buf_out.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_in),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_out),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        let pipe = match op {
            "relu"    => backend.pipe_relu,
            "sigmoid" => backend.pipe_sigmoid,
            "silu"    => backend.pipe_silu,
            _ => panic!("Unsupported chunked activation OP"),
        };

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_act, 0, &[set], &[]);
        backend.device.cmd_dispatch(cmd, (elem_count as u32 + 255) / 256, 1, 1);

        let barrier_out = vk::BufferMemoryBarrier::default()
            .buffer(buf_out.buffer).size(vk::WHOLE_SIZE)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);

        backend.device.cmd_copy_buffer(cmd, buf_out.buffer, stage_out.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: num_bytes_f32 }]);

        backend.device.end_command_buffer(cmd).unwrap();
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmds);
        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();
        backend.device.queue_wait_idle(backend.compute_queue).unwrap();
        backend.device.free_command_buffers(backend.compute_cmd_pool, &cmds);
        // NOTE: No free_descriptor_sets — permanent set is reused.
    }

    download_from_stage(tile_out, &stage_out, dtype);
    recycle_buffer(buf_in); recycle_buffer(buf_out);
    recycle_buffer(stage_in); recycle_buffer(stage_out);
}


pub fn execute_activation_into(input_raw: &[u8], op: &str, res_raw: &mut [u8], dtype: DataType, is_hybrid: bool, use_staging: bool) {
    let t_start = std::time::Instant::now();
    let (wait_id, stage_out) = submit_activation_into(input_raw, op, res_raw, dtype, is_hybrid, use_staging);
    poll_async_ops_until(wait_id);

    let t_dl_start = std::time::Instant::now();
    download_from_stage(res_raw, &stage_out, dtype);
    let t_dl = t_dl_start.elapsed();

    static PRINT_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !PRINT_ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
        println!("\n[VNN PERF] Act Sync Call Download: {:.2}ms, Total Block Time: {:.2}ms", 
                t_dl.as_secs_f64()*1000.0, t_start.elapsed().as_secs_f64()*1000.0);
    }
}

pub fn submit_activation_into(input_raw: &[u8], op: &str, _res_raw: &mut [u8], dtype: DataType, _is_hybrid: bool, _use_staging: bool) -> (u64, CachedBuffer) {

    let t_start = std::time::Instant::now();
    let backend = BACKEND.get().unwrap();
    let num_bytes = input_raw.len() as vk::DeviceSize;
    let num_elements = num_bytes as u32 / if dtype == DataType::F32 { 4 } else { 2 };
    let num_bytes_f32 = (num_elements * 4) as vk::DeviceSize;
    
    let buf_in = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Act_In"), false);
    let buf_out = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Act_Out"), false);

    let stage_in = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("Act_Stage_In"), true);
    let stage_out = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_DST, Some("Act_Stage_Out"), true);
    let t_buf = t_start.elapsed();

    upload_to_stage(input_raw, &stage_in, dtype);
    let _t_up = t_start.elapsed() - t_buf;

    let t_cmd_start = std::time::Instant::now();
    let _t_desc_submit = std::time::Duration::ZERO;
    let _t_wait_start = std::time::Instant::now();
    let _t_wait = std::time::Duration::ZERO;

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_in.buffer, buf_in.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: num_bytes_f32 }]);
        
        let barrier = vk::BufferMemoryBarrier::default().buffer(buf_in.buffer).size(vk::WHOLE_SIZE).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[barrier], &[]);

        // Bug #3 fix: Use permanent pre-allocated descriptor set — no alloc/free in async op.
        let set = *backend.perm_desc_act.lock().unwrap();
        
        let info_in = [vk::DescriptorBufferInfo::default().buffer(buf_in.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let info_out = [vk::DescriptorBufferInfo::default().buffer(buf_out.buffer).offset(0).range(vk::WHOLE_SIZE)];
        
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_in),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_out),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        let pipe = match op {
            "relu" => backend.pipe_relu,
            "sigmoid" => backend.pipe_sigmoid,
            "silu" => backend.pipe_silu,
            _ => panic!("Unsupported activation OP"),
        };

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_act, 0, &[set], &[]);
        backend.device.cmd_dispatch(cmd, (num_elements + 255) / 256, 1, 1);
        
        // Wait for compute to finish BEFORE copying back to staging
        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_out.buffer).size(vk::WHOLE_SIZE).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);
        
        backend.device.cmd_copy_buffer(cmd, buf_out.buffer, stage_out.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: num_bytes_f32 }]);
        
        backend.device.end_command_buffer(cmd).unwrap();
        
        backend.timeline_value.fetch_add(1, Ordering::SeqCst);
        let wait_id = backend.timeline_value.load(Ordering::SeqCst);
        
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(std::slice::from_ref(&wait_id));
            
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .signal_semaphores(std::slice::from_ref(&backend.timeline_semaphore))
            .push_next(&mut timeline_info);
            
        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();
        
        let stage_out_clone = CachedBuffer {
            size: stage_out.size, usage: stage_out.usage, buffer: stage_out.buffer, allocation: None, cpu_visible: stage_out.cpu_visible, mapped_ptr: stage_out.mapped_ptr,
        };

        let op_info = AsyncOp {
            staging_in1: Some(stage_in),
            staging_in2: None,
            staging_out: Some(stage_out),
            device_in1: Some(buf_in),
            device_in2: None,
            device_out: Some(buf_out),
            cmd_buffer: cmd,
            // Bug #3 fix: Use null handle to signal poll_async_ops NOT to free this set.
            // It's a permanent set that lives for the lifetime of the backend.
            desc_set: vk::DescriptorSet::null(),
            wait_id,
        };
        backend.pending_ops.lock().unwrap().push(op_info);
        
        let _t_desc_submit = t_cmd_start.elapsed();
        
        // Asynchronous return, no wait idle
        return (wait_id, stage_out_clone);
    }
}
