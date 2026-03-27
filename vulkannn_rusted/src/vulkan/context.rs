use std::ffi::{CStr, CString};
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use crate::vulkan::memory::{CachedBuffer, PoolBlock};
use crate::vulkan::pipeline::DescriptorSetPool;

pub struct AsyncOp {
    pub staging_buffers: Vec<CachedBuffer>,
    pub device_buffers: Vec<CachedBuffer>,
    pub cmd_buffer: vk::CommandBuffer,
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
    
    #[allow(dead_code)]
    pub desc_pools: Mutex<Vec<vk::DescriptorPool>>,
    #[allow(dead_code)]
    pub pipe_layout_act: vk::PipelineLayout,
    pub pipe_layout_reduce: vk::PipelineLayout,
    pub pipe_layout_elementwise: vk::PipelineLayout,
    pub pipe_layout_matmul: vk::PipelineLayout,
    pub pipe_layout_bit_linear: vk::PipelineLayout,
    pub pipe_layout_layer_norm: vk::PipelineLayout,
    pub pipe_layout_rms_norm: vk::PipelineLayout,
    pub pipe_layout_index_select: vk::PipelineLayout,
    #[allow(dead_code)]
    pub pipe_layout_bit_linear_fast: vk::PipelineLayout,
    #[allow(dead_code)]
    pub pipe_layout_bit_linear_lut: vk::PipelineLayout,
    
    #[allow(dead_code)]
    pub pipe_elementwise: vk::Pipeline,
    pub pipe_relu: vk::Pipeline,
    pub pipe_softmax: vk::Pipeline,
    pub pipe_matmul: vk::Pipeline,
    pub pipe_bit_linear: vk::Pipeline,
    #[allow(dead_code)]
    pub pipe_bit_linear_fast: vk::Pipeline,
    #[allow(dead_code)]
    pub pipe_bit_linear_lut: vk::Pipeline,
    pub pipe_layer_norm: vk::Pipeline,
    pub pipe_rms_norm: vk::Pipeline,
    pub pipe_index_select: vk::Pipeline,
    pub pipe_sigmoid: vk::Pipeline,

    pub pipe_silu: vk::Pipeline,
    pub pipe_gelu: vk::Pipeline,
    pub pipe_leaky_relu: vk::Pipeline,
    pub pipe_elu: vk::Pipeline,
    pub pipe_tanh: vk::Pipeline,
    pub pipe_clamp: vk::Pipeline,
    pub pipe_neg: vk::Pipeline,
    pub pipe_pow: vk::Pipeline,
    
    pub pipe_reduce_sum: vk::Pipeline,
    pub pipe_reduce_max: vk::Pipeline,
    pub pipe_reduce_min: vk::Pipeline,
    pub pipe_reduce_argmax: vk::Pipeline,

    pub compute_cmd_pool: vk::CommandPool,
    #[allow(dead_code)]
    pub transfer_cmd_pool: vk::CommandPool,
    pub buffer_cache: Mutex<Vec<CachedBuffer>>,

    // Rotating descriptor set pools to fix race conditions in async execution.
    pub pool_desc_act: Mutex<DescriptorSetPool>,
    pub pool_desc_reduce: Mutex<DescriptorSetPool>,
    pub pool_desc_elementwise: Mutex<DescriptorSetPool>,
    pub pool_desc_linear: Mutex<DescriptorSetPool>,
    pub pool_desc_matmul: Mutex<DescriptorSetPool>,
    pub pool_desc_bit_linear: Mutex<DescriptorSetPool>,
    pub pool_desc_layer_norm: Mutex<DescriptorSetPool>,
    pub pool_desc_rms_norm: Mutex<DescriptorSetPool>,
    pub pool_desc_index_select: Mutex<DescriptorSetPool>,
    
    pub timeline_semaphore: vk::Semaphore,
    pub timeline_value: AtomicU64,
    pub pending_ops: Mutex<Vec<AsyncOp>>,

    // Phase 1: VRAM Architecture (Memory Pooling)
    pub pool_buffer: vk::Buffer,
    pub pool_free_list: Mutex<Vec<PoolBlock>>,
}

unsafe impl Send for AshBackend {}
unsafe impl Sync for AshBackend {}

impl AshBackend {
    /// Allocates descriptor sets from the managed pools.
    /// If all current pools are exhausted (VK_ERROR_OUT_OF_POOL_MEMORY), 
    /// it automatically allocates a new hardware descriptor pool on the fly.
    #[allow(dead_code)]
    pub fn allocate_descriptor_sets(&self, layouts: &[vk::DescriptorSetLayout]) -> Vec<vk::DescriptorSet> {
        let mut pools = self.desc_pools.lock().unwrap();
        
        // Try to allocate from existing pools (starting from the last/warmest one)
        for pool in pools.iter().rev() {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(*pool)
                .set_layouts(layouts);
            if let Ok(sets) = unsafe { self.device.allocate_descriptor_sets(&alloc_info) } {
                return sets;
            }
        }
        
        // If we reach here, all pools are exhausted or none exist.
        // Lazy-allocate a new pool to support large models/heavy async load.
        println!("[VNN] Descriptor Pool exhausted (limit reached). Creating new dynamic pool...");
        let pool_sizes = [
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(4096),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1024),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(512)
            .pool_sizes(&pool_sizes);
        let new_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None) }.unwrap();
        pools.push(new_pool);
        
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(new_pool)
            .set_layouts(layouts);
        unsafe { self.device.allocate_descriptor_sets(&alloc_info) }.expect("CRITICAL: Failed to allocate descriptor sets even after pool expansion")
    }
}

/// Checks the global Vulkan timeline semaphore and cleans up completed asynchronous
/// operations. Reclaims all associated memory blocks into the caching pool.
pub fn poll_async_ops() {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let current_val = unsafe { backend.device.get_semaphore_counter_value(backend.timeline_semaphore).unwrap() };
    
    let mut pending = backend.pending_ops.lock().unwrap();
    pending.retain_mut(|op| {
        if current_val >= op.wait_id {
            for buf in op.staging_buffers.drain(..) { crate::vulkan::memory::recycle_buffer(buf); }
            for buf in op.device_buffers.drain(..) { crate::vulkan::memory::recycle_buffer(buf); }
            
            unsafe {
                backend.device.free_command_buffers(backend.compute_cmd_pool, &[op.cmd_buffer]);
            }
            false 
        } else {
            true
        }
    });
}

/// Blocks the host CPU until the Vulkan timeline semaphore reaches the target value.
pub fn poll_async_ops_until(target_val: u64) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let current = unsafe { backend.device.get_semaphore_counter_value(backend.timeline_semaphore).unwrap() };
    if current < target_val {
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&backend.timeline_semaphore))
            .values(std::slice::from_ref(&target_val));
        unsafe { backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap(); }
    }
    poll_async_ops();
}

#[allow(dead_code)]
pub fn wait_for_all() {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let current = backend.timeline_value.load(Ordering::SeqCst);
    if current > 0 {
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&backend.timeline_semaphore))
            .values(std::slice::from_ref(&current));
        unsafe { backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap(); }
    }
    poll_async_ops();
}

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
        let ext_coop     = CStr::from_bytes_with_nul(b"VK_KHR_cooperative_matrix\0").unwrap().as_ptr();

        let mut device_extension_names_raw = vec![ext_timeline];
        let mut has_fp16_ext = false;
        let mut has_coop_ext = false;
        
        if let Ok(device_exts) = unsafe { instance.enumerate_device_extension_properties(pdevice) } {
            for ext in device_exts {
                let name_bytes = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                if let Ok(name_str) = name_bytes.to_str() {
                    if name_str == "VK_KHR_shader_float16_int8" {
                        has_fp16_ext = true;
                    }
                    if name_str == "VK_KHR_cooperative_matrix" {
                        has_coop_ext = true;
                    }
                }
            }
        }

        if has_coop_ext {
            device_extension_names_raw.push(ext_coop);
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

        features12.p_next = &mut features11 as *mut _ as *mut std::ffi::c_void;

        let mut features16 = vk::PhysicalDeviceShaderFloat16Int8FeaturesKHR::default()
            .shader_float16(true);

        let mut features_coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(true);

        #[allow(unused_assignments)]
        if has_fp16 {
            features11.p_next = &mut features16 as *mut _ as *mut std::ffi::c_void;
            if has_coop_ext {
                features16.p_next = &mut features_coop as *mut _ as *mut std::ffi::c_void;
            }
        } else if has_coop_ext {
            features11.p_next = &mut features_coop as *mut _ as *mut std::ffi::c_void;
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

        let pool_sizes = [
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(4096),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1024),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(512)
            .pool_sizes(&pool_sizes);
        let desc_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }.unwrap();

        // DSL Elem: 3 Storage
        let bindings_elem = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_elem = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings_elem), None) }.unwrap();

        // DSL Act: 2 Storage
        let bindings_act = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_act = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings_act), None) }.unwrap();

        // DSL Linear: 4 Storage
        let bindings_linear = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_linear = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings_linear), None) }.unwrap();
        
        // DSL MatMul: 4 Storage (A, B, C, Bias)
        let bindings_matmul = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_matmul = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings_matmul), None) }.unwrap();
        
        // DSL BitLinear: 5 Storage
        let bindings_bit_linear = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_bit_linear = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings_bit_linear), None) }.unwrap();

        let bindings_layer_norm = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_layer_norm = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings_layer_norm), None) }.unwrap();

        let bindings_rms_norm = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_rms_norm = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings_rms_norm), None) }.unwrap();

        let pc_act_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(16)];
        let pipe_layout_act = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_act]).push_constant_ranges(&pc_act_range), None) }.unwrap();

        let dsl_reduce = dsl_act;
        let pc_reduce_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(16)];
        let pipe_layout_reduce = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_reduce]).push_constant_ranges(&pc_reduce_range), None) }.unwrap();

        let dsl_elementwise = dsl_elem;
        let pc_elementwise_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(16)];
        let pipe_layout_elementwise = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_elementwise]).push_constant_ranges(&pc_elementwise_range), None) }.unwrap();

        let dsl_index_select = dsl_elem;
        let pc_index_select_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(8)]; 
        let pipe_layout_index_select = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_index_select]).push_constant_ranges(&pc_index_select_range), None) }.unwrap();

        let pc_matmul_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(24)];
        let pipe_layout_matmul = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_matmul]).push_constant_ranges(&pc_matmul_range), None) }.unwrap();

        let pc_bit_linear_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(16)];
        let pipe_layout_bit_linear = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_bit_linear]).push_constant_ranges(&pc_bit_linear_range), None) }.unwrap();

        let pc_bit_linear_fast_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(20)];
        let pipe_layout_bit_linear_fast = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_bit_linear]).push_constant_ranges(&pc_bit_linear_fast_range), None) }.unwrap();

        let pc_norm_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(12)];
        let pipe_layout_layer_norm = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_layer_norm]).push_constant_ranges(&pc_norm_range), None) }.unwrap();
        let pipe_layout_rms_norm = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_rms_norm]).push_constant_ranges(&pc_norm_range), None) }.unwrap();

        let load_shader = |bytes: &[u8]| -> vk::ShaderModule {
            let mut cursor = std::io::Cursor::new(bytes);
            let code = ash::util::read_spv(&mut cursor).expect("Failed to read struct spv");
            let info = vk::ShaderModuleCreateInfo::default().code(&code);
            unsafe { device.create_shader_module(&info, None) }.unwrap()
        };

        let sm_act = load_shader(include_bytes!("../shaders/activation.wgsl.spv"));
        let sm_reduce = load_shader(include_bytes!("../shaders/reduce.wgsl.spv"));
        let sm_elementwise = load_shader(include_bytes!("../shaders/elementwise.comp.spv"));
        let sm_softmax = load_shader(include_bytes!("../shaders/softmax.wgsl.spv"));
        let sm_matmul = load_shader(include_bytes!("../shaders/matmul_tiled.comp.spv"));
        let sm_bit_linear = load_shader(include_bytes!("../shaders/bit_linear.comp.spv"));
        let sm_bit_linear_fast = load_shader(include_bytes!("../shaders/bit_linear_fast.comp.spv"));
        let sm_bit_linear_lut = load_shader(include_bytes!("../shaders/bit_linear_lut.comp.spv"));
        let sm_layer_norm = load_shader(include_bytes!("../shaders/layer_norm.comp.spv"));
        let sm_rms_norm = load_shader(include_bytes!("../shaders/rms_norm.comp.spv"));
        let sm_index_select = load_shader(include_bytes!("../shaders/index_select.comp.spv"));

        let entry_main = CString::new("main").unwrap();
        let entry_relu = CString::new("relu_main").unwrap();
        let entry_sigm = CString::new("sigmoid_main").unwrap();
        let entry_silu = CString::new("silu_main").unwrap();
        let entry_gelu = CString::new("gelu_main").unwrap();
        let entry_leaky = CString::new("leaky_relu_main").unwrap();
        let entry_elu  = CString::new("elu_main").unwrap();
        let entry_tanh = CString::new("tanh_main").unwrap();
        let entry_clamp = CString::new("clamp_main").unwrap();
        let entry_neg  = CString::new("neg_main").unwrap();
        let entry_pow  = CString::new("pow_main").unwrap();

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

        let pipe_elementwise = create_pipe(sm_elementwise, &entry_main, pipe_layout_elementwise);
        let pipe_softmax = create_pipe(sm_softmax, &entry_main, pipe_layout_act);
        let pipe_matmul = create_pipe(sm_matmul, &entry_main, pipe_layout_matmul);
        let pipe_bit_linear = create_pipe(sm_bit_linear, &entry_main, pipe_layout_bit_linear);
        let pipe_bit_linear_fast = create_pipe(sm_bit_linear_fast, &entry_main, pipe_layout_bit_linear_fast);
        let pipe_bit_linear_lut = create_pipe(sm_bit_linear_lut, &entry_main, pipe_layout_bit_linear);
        let pipe_layer_norm = create_pipe(sm_layer_norm, &entry_main, pipe_layout_layer_norm);
        let pipe_rms_norm = create_pipe(sm_rms_norm, &entry_main, pipe_layout_rms_norm);
        let pipe_index_select = create_pipe(sm_index_select, &entry_main, pipe_layout_index_select);
        let pipe_relu = create_pipe(sm_act, &entry_relu, pipe_layout_act);
        let pipe_sigmoid = create_pipe(sm_act, &entry_sigm, pipe_layout_act);
        let pipe_silu = create_pipe(sm_act, &entry_silu, pipe_layout_act);
        let pipe_gelu = create_pipe(sm_act, &entry_gelu, pipe_layout_act);
        let pipe_leaky_relu = create_pipe(sm_act, &entry_leaky, pipe_layout_act);
        let pipe_elu = create_pipe(sm_act, &entry_elu, pipe_layout_act);
        let pipe_tanh = create_pipe(sm_act, &entry_tanh, pipe_layout_act);
        let pipe_clamp = create_pipe(sm_act, &entry_clamp, pipe_layout_act);
        let pipe_neg = create_pipe(sm_act, &entry_neg, pipe_layout_act);
        let pipe_pow = create_pipe(sm_act, &entry_pow, pipe_layout_act);

        let entry_sum_redu = CString::new("sum_main").unwrap();
        let entry_max_redu = CString::new("max_main").unwrap();
        let entry_min_redu = CString::new("min_main").unwrap();
        let pipe_reduce_sum = create_pipe(sm_reduce, &entry_sum_redu, pipe_layout_reduce);
        let pipe_reduce_max = create_pipe(sm_reduce, &entry_max_redu, pipe_layout_reduce);
        let pipe_reduce_min = create_pipe(sm_reduce, &entry_min_redu, pipe_layout_reduce);
        let entry_argmax = std::ffi::CString::new("argmax_main").unwrap();
        let pipe_reduce_argmax = create_pipe(sm_reduce, &entry_argmax, pipe_layout_reduce);

        unsafe {
            device.destroy_shader_module(sm_act, None);
            device.destroy_shader_module(sm_reduce, None);
            device.destroy_shader_module(sm_elementwise, None);
            device.destroy_shader_module(sm_softmax, None);
            device.destroy_shader_module(sm_matmul, None);
            device.destroy_shader_module(sm_bit_linear, None);
            device.destroy_shader_module(sm_bit_linear_fast, None);
            device.destroy_shader_module(sm_layer_norm, None);
            device.destroy_shader_module(sm_rms_norm, None);
            device.destroy_shader_module(sm_index_select, None);
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
        let timeline_semaphore = unsafe { device.create_semaphore(&sem_info, None) }.expect("Failed to create Timeline Semaphore");

        let mut descriptor_pools = vec![desc_pool];
        let create_pool = |device: &ash::Device, pools: &mut Vec<vk::DescriptorPool>, layout: vk::DescriptorSetLayout, count: usize| -> DescriptorSetPool {
            let layouts = vec![layout; count];
            for pool in pools.iter().rev() {
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*pool)
                    .set_layouts(&layouts);
                if let Ok(sets) = unsafe { device.allocate_descriptor_sets(&alloc_info) } {
                    return DescriptorSetPool { sets, current: 0 };
                }
            }
            let pool_sizes = [
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(4096),
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1024),
            ];
            let pool_info = vk::DescriptorPoolCreateInfo::default().max_sets(512).pool_sizes(&pool_sizes);
            let new_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }.unwrap();
            pools.push(new_pool);
            let alloc_info = vk::DescriptorSetAllocateInfo::default().descriptor_pool(new_pool).set_layouts(&layouts);
            let sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }.expect("Failed to allocate descriptor pool during init");
            DescriptorSetPool { sets, current: 0 }
        };

        let pool_desc_act = create_pool(&device, &mut descriptor_pools, dsl_act, 128);
        let pool_desc_reduce = create_pool(&device, &mut descriptor_pools, dsl_reduce, 64);
        let pool_desc_elementwise = create_pool(&device, &mut descriptor_pools, dsl_elementwise, 64);
        let pool_desc_linear = create_pool(&device, &mut descriptor_pools, dsl_linear, 64);
        let pool_desc_matmul = create_pool(&device, &mut descriptor_pools, dsl_matmul, 64);
        let pool_desc_bit_linear = create_pool(&device, &mut descriptor_pools, dsl_bit_linear, 32);
        let pool_desc_layer_norm = create_pool(&device, &mut descriptor_pools, dsl_layer_norm, 32);
        let pool_desc_rms_norm = create_pool(&device, &mut descriptor_pools, dsl_rms_norm, 32);
        let pool_desc_index_select = create_pool(&device, &mut descriptor_pools, dsl_index_select, 32);

        AshBackend {
            _entry: entry,
            _instance: instance,
            _pdevice: pdevice,
            device,
            compute_queue,
            _compute_family: compute_family,
            transfer_queue,
            _transfer_family: transfer_family,
            allocator: Mutex::new(allocator),
            desc_pools: Mutex::new(descriptor_pools),
            pipe_layout_act,
            pipe_layout_reduce,
            pipe_layout_elementwise,
            pipe_layout_matmul,
            pipe_layout_bit_linear,
            pipe_layout_layer_norm,
            pipe_layout_rms_norm,
            pipe_layout_index_select,
            pipe_layout_bit_linear_fast,
            pipe_layout_bit_linear_lut: pipe_layout_bit_linear,
            pipe_elementwise,
            pipe_relu,
            pipe_softmax,
            pipe_matmul,
            pipe_bit_linear,
            pipe_bit_linear_fast,
            pipe_bit_linear_lut,
            pipe_layer_norm,
            pipe_rms_norm,
            pipe_index_select,
            pipe_sigmoid,
            pipe_silu,
            pipe_gelu,
            pipe_leaky_relu,
            pipe_elu,
            pipe_tanh,
            pipe_clamp,
            pipe_neg,
            pipe_pow,
            pipe_reduce_sum,
            pipe_reduce_max,
            pipe_reduce_min,
            pipe_reduce_argmax,
            compute_cmd_pool,
            transfer_cmd_pool,
            buffer_cache: Mutex::new(Vec::new()),
            pool_desc_act: Mutex::new(pool_desc_act),
            pool_desc_reduce: Mutex::new(pool_desc_reduce),
            pool_desc_elementwise: Mutex::new(pool_desc_elementwise),
            pool_desc_linear: Mutex::new(pool_desc_linear),
            pool_desc_matmul: Mutex::new(pool_desc_matmul),
            pool_desc_bit_linear: Mutex::new(pool_desc_bit_linear),
            pool_desc_layer_norm: Mutex::new(pool_desc_layer_norm),
            pool_desc_rms_norm: Mutex::new(pool_desc_rms_norm),
            pool_desc_index_select: Mutex::new(pool_desc_index_select),
            timeline_semaphore,
            timeline_value: AtomicU64::new(0),
            pending_ops: Mutex::new(Vec::new()),
            pool_buffer: vk::Buffer::null(),
            pool_free_list: Mutex::new(Vec::new()),
        }
    });
}
