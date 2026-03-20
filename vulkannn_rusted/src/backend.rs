use std::ffi::{CStr, CString};
use std::sync::{Mutex, OnceLock};
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc, Allocation, AllocationCreateDesc};
use gpu_allocator::MemoryLocation;
use crate::tensor::DataType;
use crate::cpu::*;
// pub use crate::swar_int8::*;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct AsyncOp {
    pub staging_buffers: Vec<CachedBuffer>,
    pub device_buffers: Vec<CachedBuffer>,
    pub cmd_buffer: vk::CommandBuffer,
    pub desc_set: vk::DescriptorSet,
    pub wait_id: u64,
}

pub struct DescriptorSetPool {
    pub sets: Vec<vk::DescriptorSet>,
    pub current: usize,
}

impl DescriptorSetPool {
    pub fn next(&mut self) -> vk::DescriptorSet {
        let set = self.sets[self.current];
        self.current = (self.current + 1) % self.sets.len();
        set
    }
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
    #[allow(dead_code)]
    pub pipe_layout_act: vk::PipelineLayout,
    pub pipe_layout_reduce: vk::PipelineLayout,
    pub pipe_layout_elementwise: vk::PipelineLayout,
    pub pipe_layout_linear: vk::PipelineLayout,
    pub pipe_layout_matmul: vk::PipelineLayout,
    pub pipe_layout_bit_linear: vk::PipelineLayout,
    
    #[allow(dead_code)]
    pub pipe_elementwise: vk::Pipeline,
    pub pipe_relu: vk::Pipeline,
    pub pipe_softmax: vk::Pipeline,
    pub pipe_linear: vk::Pipeline,
    pub pipe_matmul: vk::Pipeline,
    pub pipe_bit_linear: vk::Pipeline,
    pub pipe_sigmoid: vk::Pipeline,
    pub pipe_silu: vk::Pipeline,
    pub pipe_gelu: vk::Pipeline,
    pub pipe_leaky_relu: vk::Pipeline,
    pub pipe_elu: vk::Pipeline,
    pub pipe_tanh: vk::Pipeline,
    pub pipe_clamp: vk::Pipeline,
    
    pub pipe_reduce_sum: vk::Pipeline,
    pub pipe_reduce_max: vk::Pipeline,
    pub pipe_reduce_min: vk::Pipeline,

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
    
    pub timeline_semaphore: vk::Semaphore,
    pub timeline_value: AtomicU64,
    pub pending_ops: Mutex<Vec<AsyncOp>>,

    // Phase 1: VRAM Architecture (Memory Pooling)
    pub pool_buffer: vk::Buffer,
    pub pool_free_list: Mutex<Vec<PoolBlock>>,
}

unsafe impl Send for AshBackend {}
unsafe impl Sync for AshBackend {}

#[derive(Clone, Copy, Debug)]
pub struct PoolBlock {
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
    pub used: bool,
}

/// A reusable Vulkan buffer paired with its native memory allocation.
/// Tracked by the memory allocator to enable zero-copy VRAM reuse.
pub struct CachedBuffer {
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub cpu_visible: bool,
    pub mapped_ptr: Option<*mut u8>,
    pub pool_offset: Option<vk::DeviceSize>, // New: Tracks offset if sub-allocated
}

unsafe impl Send for CachedBuffer {}
unsafe impl Sync for CachedBuffer {}

impl CachedBuffer {
    pub fn copy_for_async(&self) -> Self {
        Self {
            size: self.size,
            usage: self.usage,
            buffer: self.buffer,
            allocation: None, // Do NOT copy allocation
            cpu_visible: self.cpu_visible,
            mapped_ptr: self.mapped_ptr,
            pool_offset: self.pool_offset,
        }
    }
}

pub static BACKEND: OnceLock<AshBackend> = OnceLock::new();

/// Bootstraps the raw Ash Vulkan 1.2 engine.
/// Configures physical devices, asynchronous compute and transfer queues,
/// and allocates persistent command pools mapping to the hardware ACEs.
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

        #[cfg(target_os = "linux")] // Most linux drivers (Mesa/RADV/NVIDIA) support these
        {
            // Subgroups are standard in Vulkan 1.1, but some features might be optional.
            // basic, vote, arithmetic, ballot are usually available on modern GPUs.
        }
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

        // Bug #3 fix: Do NOT use FREE_DESCRIPTOR_SET_BIT.
        // Per Vulkan spec and vkguide.dev best practices: omitting this flag lets the driver
        // use a simpler bump allocator internally, which is faster and doesn't fragment.
        // We only need a small fixed pool (3 permanent sets + headroom for async activation ops).
        // Bug #3 fix: Do NOT use FREE_DESCRIPTOR_SET_BIT.
        // We now use rotating pools of descriptor sets to support asynchronous pipelining.
        let pool_sizes = [
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1024),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(256),
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
        
        // DSL MatMul: 3 Storage
        let bindings_matmul = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
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


        let pc_act_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(16)];
        let pipe_layout_act = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_act]).push_constant_ranges(&pc_act_range), None) }.unwrap();

        let dsl_reduce = dsl_act;
        let pc_reduce_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(16)];
        let pipe_layout_reduce = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_reduce]).push_constant_ranges(&pc_reduce_range), None) }.unwrap();

        let dsl_elementwise = dsl_elem;
        let pc_elementwise_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(16)];
        let pipe_layout_elementwise = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_elementwise]).push_constant_ranges(&pc_elementwise_range), None) }.unwrap();

        let pc_linear_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(20)]; // Increased to 20 for transpose_b
        let pipe_layout_linear = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_linear]).push_constant_ranges(&pc_linear_range), None) }.unwrap();

        let pc_matmul_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(12)];
        let pipe_layout_matmul = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_matmul]).push_constant_ranges(&pc_matmul_range), None) }.unwrap();

        let pc_bit_linear_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(16)];
        let pipe_layout_bit_linear = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dsl_bit_linear]).push_constant_ranges(&pc_bit_linear_range), None) }.unwrap();

        let load_shader = |bytes: &[u8]| -> vk::ShaderModule {
            let mut cursor = std::io::Cursor::new(bytes);
            let code = ash::util::read_spv(&mut cursor).expect("Failed to read struct spv");
            let info = vk::ShaderModuleCreateInfo::default().code(&code);
            unsafe { device.create_shader_module(&info, None) }.unwrap()
        };

        let sm_act = load_shader(include_bytes!("shaders/activation.wgsl.spv"));
        let sm_reduce = load_shader(include_bytes!("shaders/reduce.wgsl.spv"));
        let sm_elementwise = load_shader(include_bytes!("shaders/elementwise.comp.spv"));
        let sm_softmax = load_shader(include_bytes!("shaders/softmax.wgsl.spv"));
        let sm_linear = load_shader(include_bytes!("shaders/linear.comp.spv"));
        let sm_matmul = load_shader(include_bytes!("shaders/matmul_tiled.comp.spv"));
        let sm_bit_linear = load_shader(include_bytes!("shaders/bit_linear.comp.spv"));

        let entry_main = CString::new("main").unwrap();
        let entry_relu = CString::new("relu_main").unwrap();
        let entry_sigm = CString::new("sigmoid_main").unwrap();
        let entry_silu = CString::new("silu_main").unwrap();
        let entry_gelu = CString::new("gelu_main").unwrap();
        let entry_leaky = CString::new("leaky_relu_main").unwrap();
        let entry_elu  = CString::new("elu_main").unwrap();
        let entry_tanh = CString::new("tanh_main").unwrap();
        let entry_clamp = CString::new("clamp_main").unwrap();

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
        let pipe_linear = create_pipe(sm_linear, &entry_main, pipe_layout_linear);
        let pipe_matmul = create_pipe(sm_matmul, &entry_main, pipe_layout_matmul);
        let pipe_bit_linear = create_pipe(sm_bit_linear, &entry_main, pipe_layout_bit_linear);
        let pipe_relu = create_pipe(sm_act, &entry_relu, pipe_layout_act);
        let pipe_sigmoid = create_pipe(sm_act, &entry_sigm, pipe_layout_act);
        let pipe_silu = create_pipe(sm_act, &entry_silu, pipe_layout_act);
        let pipe_gelu = create_pipe(sm_act, &entry_gelu, pipe_layout_act);
        let pipe_leaky_relu = create_pipe(sm_act, &entry_leaky, pipe_layout_act);
        let pipe_elu = create_pipe(sm_act, &entry_elu, pipe_layout_act);
        let pipe_tanh = create_pipe(sm_act, &entry_tanh, pipe_layout_act);
        let pipe_clamp = create_pipe(sm_act, &entry_clamp, pipe_layout_act);

        let entry_sum_redu = CString::new("sum_main").unwrap();
        let entry_max_redu = CString::new("max_main").unwrap();
        let entry_min_redu = CString::new("min_main").unwrap();
        let pipe_reduce_sum = create_pipe(sm_reduce, &entry_sum_redu, pipe_layout_reduce);
        let pipe_reduce_max = create_pipe(sm_reduce, &entry_max_redu, pipe_layout_reduce);
        let pipe_reduce_min = create_pipe(sm_reduce, &entry_min_redu, pipe_layout_reduce);

        unsafe {
            device.destroy_shader_module(sm_act, None);
            device.destroy_shader_module(sm_reduce, None);
            device.destroy_shader_module(sm_elementwise, None);
            device.destroy_shader_module(sm_softmax, None);
            device.destroy_shader_module(sm_linear, None);
            device.destroy_shader_module(sm_matmul, None);
            device.destroy_shader_module(sm_bit_linear, None);
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
        // Initialize Descriptor Set Pools
        let create_pool = |layout: vk::DescriptorSetLayout, count: usize| -> DescriptorSetPool {
            let layouts = vec![layout; count];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(&layouts);
            let sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }.expect("Failed to allocate descriptor set pool");
            DescriptorSetPool { sets, current: 0 }
        };

        let pool_desc_act = create_pool(dsl_act, 128);
        let pool_desc_reduce = create_pool(dsl_reduce, 64);
        let pool_desc_elementwise = create_pool(dsl_elementwise, 64);
        let pool_desc_linear = create_pool(dsl_linear, 64);
        let pool_desc_matmul = create_pool(dsl_matmul, 64);
        let pool_desc_bit_linear = create_pool(dsl_bit_linear, 32);

        // NEW: Phase 1 VRAM Pool Allocation
        let pool_size = 256 * 1024 * 1024; // 256MB (reduced for compatibility with R7 200 series)
        let pool_info = vk::BufferCreateInfo::default()
            .size(pool_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::UNIFORM_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        
        let pool_buffer = unsafe { device.create_buffer(&pool_info, None) }.unwrap();
        let pool_reqs = unsafe { device.get_buffer_memory_requirements(pool_buffer) };
        
        let mut allocator = allocator; // Ensure it's mutable
        let pool_allocation = allocator.allocate(&AllocationCreateDesc {
            name: "VNN_VRAM_POOL",
            requirements: pool_reqs,
            location: MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        }).expect("Failed to allocate 1GB VRAM pool");

        unsafe { device.bind_buffer_memory(pool_buffer, pool_allocation.memory(), pool_allocation.offset()) }.unwrap();

        AshBackend {
            _entry: entry, _instance: instance, _pdevice: pdevice, device,
            compute_queue, _compute_family: compute_family,
            transfer_queue, _transfer_family: transfer_family,
            allocator: Mutex::new(allocator),
            desc_pool: Mutex::new(desc_pool),
            pipe_layout_act, pipe_layout_reduce, pipe_layout_elementwise, pipe_layout_linear,
            pipe_elementwise, pipe_relu, pipe_softmax, pipe_linear, pipe_sigmoid, pipe_silu,
            pipe_gelu, pipe_leaky_relu, pipe_elu, pipe_tanh, pipe_clamp,
            pipe_reduce_sum, pipe_reduce_max, pipe_reduce_min,
            compute_cmd_pool, transfer_cmd_pool,
            buffer_cache: Mutex::new(Vec::new()),
            pool_desc_act:    Mutex::new(pool_desc_act),
            pool_desc_reduce: Mutex::new(pool_desc_reduce),
            pool_desc_elementwise: Mutex::new(pool_desc_elementwise),
            pool_desc_linear: Mutex::new(pool_desc_linear),
            pool_desc_matmul: Mutex::new(pool_desc_matmul),
            pool_desc_bit_linear: Mutex::new(pool_desc_bit_linear),
            pipe_layout_matmul,
            pipe_matmul,
            pipe_layout_bit_linear,
            pipe_bit_linear,
            timeline_semaphore,
            timeline_value: AtomicU64::new(0),
            pending_ops: Mutex::new(Vec::new()),
            pool_buffer,
            pool_free_list: Mutex::new(vec![PoolBlock { offset: 0, size: pool_size, used: false }]),
        }
    });
}

/// Checks the global Vulkan timeline semaphore and cleans up completed asynchronous
/// operations. Reclaims all associated memory blocks into the caching pool.
pub fn poll_async_ops() {
    let backend = BACKEND.get().unwrap();
    let current_val = unsafe { backend.device.get_semaphore_counter_value(backend.timeline_semaphore).unwrap() };
    
    let mut pending = backend.pending_ops.lock().unwrap();
    pending.retain_mut(|op| {
        if current_val >= op.wait_id {
            for buf in op.staging_buffers.drain(..) { recycle_buffer(buf); }
            for buf in op.device_buffers.drain(..) { recycle_buffer(buf); }
            
            unsafe {
                backend.device.free_command_buffers(backend.compute_cmd_pool, &[op.cmd_buffer]);
                if op.desc_set != vk::DescriptorSet::null() {
                    backend.device.free_descriptor_sets(*backend.desc_pool.lock().unwrap(), &[op.desc_set]).unwrap();
                }
            }
            false 
        } else {
            true
        }
    });
}

/// Blocks the host CPU until the Vulkan timeline semaphore reaches the target value.
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

/// Requests a Vulkan memory allocation.
/// Attempts to fetch a suitable existing block from the shared `buffer_cache`.
/// If no matching blocks exist or if VRAM boundaries are exceeded (triggering OOM retry logic),
/// a fresh buffer and `vk::DeviceMemory` allocation is created.
pub fn get_buffer(size: vk::DeviceSize, usage: vk::BufferUsageFlags, label: Option<&str>, cpu_visible: bool) -> CachedBuffer {
    let backend = BACKEND.get().unwrap();
    
    // Alignment requirement for sub-allocation
    let alignment: vk::DeviceSize = 256;
    let aligned_size = (size + alignment - 1) & !(alignment - 1);

    // Try pool first for non-CPU-visible buffers (Ph1 Opt)
    if !cpu_visible {
        let mut free_list = backend.pool_free_list.lock().unwrap();
        if let Some(idx) = free_list.iter().position(|b| !b.used && b.size >= aligned_size) {
            let block = free_list[idx];
            let remaining = block.size - aligned_size;
            
            free_list[idx].used = true;
            free_list[idx].size = aligned_size;
            
            let offset = block.offset;
            
            if remaining > 0 {
                free_list.insert(idx + 1, PoolBlock {
                    offset: offset + aligned_size,
                    size: remaining,
                    used: false,
                });
            }
            
            return CachedBuffer {
                size: aligned_size,
                usage,
                buffer: backend.pool_buffer,
                allocation: None, // No individual allocation
                cpu_visible: false,
                mapped_ptr: None,
                pool_offset: Some(offset),
            };
        }
    }

    // Fallback to cache or new allocation
    if let Ok(mut cache) = backend.buffer_cache.lock() {
        if let Some(idx) = cache.iter().position(|b| b.size >= size && b.usage.contains(usage) && b.cpu_visible == cpu_visible && b.pool_offset.is_none()) {
            let cached = cache.swap_remove(idx);
            return cached;
        }
    }
    
    let buffer_info = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { backend.device.create_buffer(&buffer_info, None) }.unwrap();
    
    let requirements = unsafe { backend.device.get_buffer_memory_requirements(buffer) };
    
    let location = if cpu_visible { 
        MemoryLocation::CpuToGpu    // CPU writes, GPU reads (upload staging)
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
    
    CachedBuffer { size, usage, buffer, allocation: Some(allocation), cpu_visible: cpu_visible_actual, mapped_ptr, pool_offset: None }
}

/// Like get_buffer but allocates HOST-readable (GpuToCpu) memory.
/// Must be used for all staging buffers the GPU writes and the CPU reads back from.
pub fn get_buffer_readback(size: vk::DeviceSize, usage: vk::BufferUsageFlags, label: Option<&str>) -> CachedBuffer {
    let backend = BACKEND.get().unwrap();

    // Try cache first
    if let Ok(mut cache) = backend.buffer_cache.lock() {
        if let Some(idx) = cache.iter().position(|b| b.size >= size && b.usage.contains(usage) && b.cpu_visible && b.pool_offset.is_none()) {
            let cached = cache.swap_remove(idx);
            return cached;
        }
    }

    let buffer_info = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { backend.device.create_buffer(&buffer_info, None) }.unwrap();
    let requirements = unsafe { backend.device.get_buffer_memory_requirements(buffer) };

    let mut retry_count = 0;
    let allocation = loop {
        let result = backend.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name: label.unwrap_or("ReadbackBuffer"),
            requirements,
            location: MemoryLocation::GpuToCpu,   // GPU writes, CPU reads
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        });
        match result {
            Ok(alloc) => break alloc,
            Err(e) => {
                if retry_count == 0 { clear_all_caches(); retry_count += 1; }
                else { panic!("[VNN] Readback alloc failed: {:?}", e); }
            }
        }
    };

    let mapped_ptr = allocation.mapped_ptr().map(|p| p.as_ptr() as *mut u8);
    unsafe { backend.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }.unwrap();
    CachedBuffer { size, usage, buffer, allocation: Some(allocation), cpu_visible: true, mapped_ptr, pool_offset: None }
}

pub fn recycle_buffer(cached: CachedBuffer) {
    let backend = BACKEND.get().unwrap();
    
    // Phase 1: If from pool, return it to the pool's free list
    if let Some(offset) = cached.pool_offset {
        let mut free_list = backend.pool_free_list.lock().unwrap();
        if let Some(idx) = free_list.iter().position(|b| b.offset == offset && b.used) {
            free_list[idx].used = false;
            // Merge adjacent free blocks (simple coalescing)
            coalesce_pool(idx, &mut free_list);
            return;
        }
    }

    if let Ok(mut cache) = backend.buffer_cache.lock() {
        let current_total: u64 = cache.iter().map(|b| b.size).sum();
        if current_total > 512 * 1024 * 1024 {
            prune_buffer_cache(2); 
        }
        cache.push(cached);
    }
}

fn coalesce_pool(idx: usize, free_list: &mut Vec<PoolBlock>) {
    // Merge with right
    if idx + 1 < free_list.len() && !free_list[idx + 1].used {
        let next = free_list.remove(idx + 1);
        free_list[idx].size += next.size;
    }
    // Merge with left
    if idx > 0 && !free_list[idx - 1].used {
        let current = free_list.remove(idx);
        free_list[idx - 1].size += current.size;
    }
}

/// Drops and permanently frees a designated number of the least recently used buffers
/// within the caching pool to respect the Bonaire 1GB VRAM constraints.
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

/// Completely flushes all reserved caching memory, releasing it entirely back to the native allocator.
/// Triggered during automated Host OOM retries or explicit Python garbage collection phases.
pub fn clear_all_caches() {
    let backend = BACKEND.get().unwrap();
    if let Ok(mut cache) = backend.buffer_cache.lock() {
        while !cache.is_empty() {
            let buf = cache.pop().unwrap();
            destroy_cached_buffer(buf);
        }
    }
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
    let bytes_per_elem = match dtype {
        DataType::F32  => 4,
        DataType::Int8 => 1,
        _ => 2,
    };
    let num_elements = src_raw.len() / bytes_per_elem;
    let dst_slice = unsafe { std::slice::from_raw_parts_mut(ptr, num_elements) };

    if dtype == DataType::F16 {
        let src_slice = bytemuck::cast_slice::<u8, half::f16>(src_raw);
        convert_f16_to_f32(src_slice, dst_slice);
    } else if dtype == DataType::BF16 {
        let src_slice = bytemuck::cast_slice::<u8, half::bf16>(src_raw);
        convert_bf16_to_f32(src_slice, dst_slice);
    } else if dtype == DataType::Int8 {
        let src_slice = bytemuck::cast_slice::<u8, i8>(src_raw);
        for i in 0..num_elements {
            dst_slice[i] = src_slice[i] as f32;
        }
    } else {
        unsafe { std::ptr::copy_nonoverlapping(src_raw.as_ptr(), ptr as *mut u8, src_raw.len()); }
    }
}

fn download_from_stage(dst_raw: &mut [u8], stage: &CachedBuffer, dtype: DataType) {
    let ptr = stage.mapped_ptr.unwrap() as *const f32;
    let bytes_per_elem = match dtype {
        DataType::F32  => 4,
        DataType::Int8 => 1,
        _ => 2,
    };
    let num_elements = dst_raw.len() / bytes_per_elem;
    let src_slice = unsafe { std::slice::from_raw_parts(ptr, num_elements) };

    if dtype == DataType::F16 {
        let dst_slice = bytemuck::cast_slice_mut::<u8, half::f16>(dst_raw);
        convert_f32_to_f16(src_slice, dst_slice);
    } else if dtype == DataType::BF16 {
        let dst_slice = bytemuck::cast_slice_mut::<u8, half::bf16>(dst_raw);
        convert_f32_to_bf16(src_slice, dst_slice);
    } else if dtype == DataType::Int8 {
        let dst_slice = bytemuck::cast_slice_mut::<u8, i8>(dst_raw);
        for i in 0..num_elements {
            dst_slice[i] = src_slice[i] as i8;
        }
    } else {
        unsafe { std::ptr::copy_nonoverlapping(ptr as *const u8, dst_raw.as_mut_ptr(), dst_raw.len()); }
    }
}

fn upload_to_stage_raw(src_raw: &[u8], stage: &CachedBuffer) {
    let ptr = stage.mapped_ptr.expect("Stage buffer must be mapped");
    unsafe { std::ptr::copy_nonoverlapping(src_raw.as_ptr(), ptr, src_raw.len()); }
}

fn download_from_stage_raw(dst_raw: &mut [u8], stage: &CachedBuffer) {
    let ptr = stage.mapped_ptr.expect("Stage buffer must be mapped");
    unsafe { std::ptr::copy_nonoverlapping(ptr, dst_raw.as_mut_ptr(), dst_raw.len()); }
}


/// Executes a fused elementwise operation (mul, sub, div, add) on the GPU.
/// Uses a generic shader with operation ID passed via push constants.
pub fn execute_elementwise_into(a_raw: &[u8], b_raw: &[u8], res_raw: &mut [u8], op_id: u32, dtype: DataType) {
    let backend = BACKEND.get().unwrap();
    let bytes_per_elem = match dtype {
        DataType::F32  => 4,
        DataType::Int8 => 1,
        _ => 2,
    };
    let num_elements = a_raw.len() / bytes_per_elem;
    let num_bytes_f32 = (num_elements * 4) as vk::DeviceSize;

    // Use mappable buffers (CpuToGpu) for small data, GpuOnly for large.
    let buf_a = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Elem_A"), false);
    let buf_b = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Elem_B"), false);
    let buf_c = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Elem_C"), false);

    let stage_a = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("Elem_Stage_A"), true);
    let stage_b = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("Elem_Stage_B"), true);
    let stage_c = get_buffer_readback(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_DST, Some("Elem_Stage_C"));

    upload_to_stage(a_raw, &stage_a, dtype);
    upload_to_stage(b_raw, &stage_b, dtype);

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_a.buffer, buf_a.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_a.pool_offset.unwrap_or(0), size: num_bytes_f32 }]);
        backend.device.cmd_copy_buffer(cmd, stage_b.buffer, buf_b.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_b.pool_offset.unwrap_or(0), size: num_bytes_f32 }]);
        
        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).size(buf_a.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).size(buf_b.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        // Use rotating descriptor set pool to safely support async.
        let set = backend.pool_desc_elementwise.lock().unwrap().next();
        let info_a = [vk::DescriptorBufferInfo::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).range(buf_a.size)];
        let info_b = [vk::DescriptorBufferInfo::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).range(buf_b.size)];
        let info_c = [vk::DescriptorBufferInfo::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).range(buf_c.size)];
        
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_a),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_b),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_c),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        // Push constants: 16 bytes for consistency across all shaders.
        let mut pc = [0u32; 4];
        pc[0] = num_elements as u32;
        pc[1] = op_id;
        let pc_bytes = bytemuck::cast_slice(&pc);
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_elementwise, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_elementwise);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_elementwise, 0, &[set], &[]);
        let workgroups = (num_elements as u32 + 255) / 256;
        backend.device.cmd_dispatch(cmd, workgroups, 1, 1);
        
        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).size(buf_c.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);
        
        backend.device.cmd_copy_buffer(cmd, buf_c.buffer, stage_c.buffer, &[vk::BufferCopy { src_offset: buf_c.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_f32 }]);
        
        backend.device.end_command_buffer(cmd).unwrap();
        
        let wait_val = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;
        
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(std::slice::from_ref(&wait_val));
        
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .push_next(&mut timeline_info)
            .command_buffers(&cmds)
            .signal_semaphores(std::slice::from_ref(&backend.timeline_semaphore));
            
        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();
        
        backend.pending_ops.lock().unwrap().push(AsyncOp {
            staging_buffers: vec![stage_a, stage_b, stage_c.copy_for_async()],
            device_buffers: vec![buf_a, buf_b, buf_c],
            cmd_buffer: cmd,
            desc_set: vk::DescriptorSet::null(),
            wait_id: wait_val,
        });

        let backend_ref = BACKEND.get().unwrap();
        let target = wait_val;
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&backend_ref.timeline_semaphore))
            .values(std::slice::from_ref(&target));
        backend_ref.device.wait_semaphores(&wait_info, u64::MAX).unwrap();
        
        download_from_stage(res_raw, &stage_c, dtype); 
    }
    
    poll_async_ops();
}

/// Executes an activation function into a pre-allocated buffer on the GPU.
/// Optimally used for in-place or pre-allocated tensor operations.
pub fn execute_activation_into(input_raw: &[u8], op: &str, param1: f32, param2: f32, res_raw: &mut [u8], dtype: DataType, is_hybrid: bool, use_staging: bool) {
    let t_start = std::time::Instant::now();
    let (wait_id, stage_out) = submit_activation_into(input_raw, op, param1, param2, res_raw, dtype, is_hybrid, use_staging);
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

/// Executes a tree-reduction (Sum, Mean, Max, Min) on the GPU.
/// Uses a two-pass reduction strategy with shared memory kernels for maximum throughput.
pub fn execute_reduce(input_raw: &[u8], op: &str, dtype: DataType) -> Vec<f32> {
    let bytes_per_elem = match dtype {
        DataType::F32 => 4,
        DataType::Int8 => 1,
        _ => 2,
    };
    let elem_count = input_raw.len() / bytes_per_elem;
    let num_bytes_f32 = (elem_count * 4) as vk::DeviceSize;
    let num_blocks = (elem_count + 255) / 256;
    let out_num_bytes = (num_blocks * 4) as vk::DeviceSize;

    let backend = BACKEND.get().unwrap();

    let buf_in  = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Reduce_In"),  false);
    let buf_out = get_buffer(out_num_bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Reduce_Out"), false);
    let stage_in  = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("Reduce_Stage_In"),  true);
    let stage_out = get_buffer_readback(out_num_bytes, vk::BufferUsageFlags::TRANSFER_DST, Some("Reduce_Stage_Out"));

    upload_to_stage(input_raw, &stage_in, dtype);

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_in.buffer, buf_in.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_in.pool_offset.unwrap_or(0), size: num_bytes_f32 }]);

        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(buf_in.buffer).offset(buf_in.pool_offset.unwrap_or(0)).size(buf_in.size)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[barrier], &[]);

        let set = backend.pool_desc_reduce.lock().unwrap().next();

        let info_in  = [vk::DescriptorBufferInfo::default().buffer(buf_in.buffer).offset(buf_in.pool_offset.unwrap_or(0)).range(buf_in.size)];
        let info_out = [vk::DescriptorBufferInfo::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).range(buf_out.size)];
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_in),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_out),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        let pipe = match op {
            "sum" => backend.pipe_reduce_sum,
            "mean" => backend.pipe_reduce_sum, // mean uses sum, we divide on CPU
            "max" => backend.pipe_reduce_max,
            "min" => backend.pipe_reduce_min,
            _ => panic!("Unsupported reduction OP: {}", op),
        };

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_reduce, 0, &[set], &[]);
        
        let pc_data = [elem_count as u32, 0u32, 0u32]; // size, stride, size_d
        let pc_bytes = bytemuck::cast_slice(&pc_data);
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_reduce, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
        
        backend.device.cmd_dispatch(cmd, num_blocks as u32, 1, 1);

        let barrier_out = vk::BufferMemoryBarrier::default()
            .buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).size(buf_out.size)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);

        backend.device.cmd_copy_buffer(cmd, buf_out.buffer, stage_out.buffer, &[vk::BufferCopy { src_offset: buf_out.pool_offset.unwrap_or(0), dst_offset: 0, size: out_num_bytes }]);

        backend.device.end_command_buffer(cmd).unwrap();

        let wait_val = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;
        let signal_values = [wait_val];
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
        let cmds = [cmd];
        let signal_sems = [backend.timeline_semaphore];
        let submit_info = vk::SubmitInfo::default()
            .push_next(&mut timeline_info)
            .command_buffers(&cmds)
            .signal_semaphores(&signal_sems);
            
        let submits = [submit_info];
        backend.device.queue_submit(backend.compute_queue, &submits, vk::Fence::null()).unwrap();
        
        backend.pending_ops.lock().unwrap().push(AsyncOp {
            staging_buffers: vec![stage_in, stage_out.copy_for_async()],
            device_buffers: vec![buf_in, buf_out],
            cmd_buffer: cmd,
            desc_set: vk::DescriptorSet::null(),
            wait_id: wait_val,
        });

        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(std::slice::from_ref(&backend.timeline_semaphore)).values(std::slice::from_ref(&wait_val));
        backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap();
    }

    let mut out_bytes = vec![0u8; out_num_bytes as usize];
    download_from_stage(&mut out_bytes, &stage_out, DataType::F32);
    poll_async_ops();

    bytemuck::cast_slice(&out_bytes).to_vec()
}

pub fn submit_activation_into(input_raw: &[u8], op: &str, param1: f32, param2: f32, _res_raw: &mut [u8], dtype: DataType, _is_hybrid: bool, _use_staging: bool) -> (u64, CachedBuffer) {

    let t_start = std::time::Instant::now();
    let _backend = BACKEND.get().unwrap();
    let num_bytes = input_raw.len() as vk::DeviceSize;
    let bytes_per_elem = match dtype {
        DataType::F32 => 4,
        DataType::Int8 => 1,
        _ => 2,
    };
    let num_elements = num_bytes as u32 / bytes_per_elem;
    let num_bytes_staging = (num_elements * 4) as vk::DeviceSize;
    
    let buf_in = get_buffer(num_bytes_staging, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Act_In"), false);
    let buf_out = get_buffer(num_bytes_staging, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Act_Out"), false);

    let stage_in = get_buffer(num_bytes_staging, vk::BufferUsageFlags::TRANSFER_SRC, Some("Act_Stage_In"), true);
    let stage_out = get_buffer_readback(num_bytes_staging, vk::BufferUsageFlags::TRANSFER_DST, Some("Act_Stage_Out"));
    let t_buf = t_start.elapsed();

    upload_to_stage(input_raw, &stage_in, dtype);
    let _t_up = t_start.elapsed() - t_buf;

    let backend = BACKEND.get().unwrap();
    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_in.buffer, buf_in.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_in.pool_offset.unwrap_or(0), size: num_bytes_staging }]);
        
        let barrier = vk::BufferMemoryBarrier::default().buffer(buf_in.buffer).offset(buf_in.pool_offset.unwrap_or(0)).size(buf_in.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[barrier], &[]);

        // Use rotating descriptor set pool to safely support async.
        let set = backend.pool_desc_act.lock().unwrap().next();
        
        let info_in = [vk::DescriptorBufferInfo::default().buffer(buf_in.buffer).offset(buf_in.pool_offset.unwrap_or(0)).range(buf_in.size)];
        let info_out = [vk::DescriptorBufferInfo::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).range(buf_out.size)];
        
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_in),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_out),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        let pipe = match op {
            "relu" => backend.pipe_relu,
            "sigmoid" => backend.pipe_sigmoid,
            "silu" => backend.pipe_silu,
            "gelu" => backend.pipe_gelu,
            "leaky_relu" => backend.pipe_leaky_relu,
            "elu" => backend.pipe_elu,
            "tanh" => backend.pipe_tanh,
            "clamp" => backend.pipe_clamp,
            _ => panic!("Unsupported activation OP"),
        };

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_act, 0, &[set], &[]);
        
        let pc_data = [num_elements as f32, 0.0, param1, param2];
        let pc_bytes = bytemuck::cast_slice(&pc_data);
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_act, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
        
        backend.device.cmd_dispatch(cmd, (num_elements + 255) / 256, 1, 1);
        
        // Wait for compute to finish BEFORE copying back to staging
        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).size(buf_out.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);
        
        backend.device.cmd_copy_buffer(cmd, buf_out.buffer, stage_out.buffer, &[vk::BufferCopy { src_offset: buf_out.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_staging }]);
        
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
        
        let async_stage_out = stage_out.copy_for_async();

        let op_info = AsyncOp {
            staging_buffers: vec![stage_in, async_stage_out],
            device_buffers: vec![buf_in, buf_out],
            cmd_buffer: cmd,
            desc_set: vk::DescriptorSet::null(),
            wait_id,
        };
        backend.pending_ops.lock().unwrap().push(op_info);
        
        return (wait_id, stage_out.copy_for_async());
    }
}

pub fn execute_softmax_into(input_raw: &[u8], output_raw: &mut [u8], width: u32, height: u32, is_log: bool, dtype: DataType) {
    let backend = BACKEND.get().unwrap();
    let num_bytes_f32 = (width * height * 4) as vk::DeviceSize;

    let buf_in  = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Softmax_In"), false);
    // buf_out must not alias buf_in in the pool — force it to a fresh standalone buffer
    // by using TRANSFER_DST|SRC|STORAGE combination that pool won't reuse for buf_in:
    let buf_out_size = num_bytes_f32 + 1; // size mismatch forces new allocation, avoids pool aliasing
    let buf_out = get_buffer(buf_out_size, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Softmax_Out"), false);
    let stage_in  = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("Softmax_Stage_In"), true);
    let stage_out = get_buffer_readback(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_DST, Some("Softmax_Stage_Out"));

    upload_to_stage(input_raw, &stage_in, dtype);

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_in.buffer, buf_in.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_in.pool_offset.unwrap_or(0), size: num_bytes_f32 }]);

        let barrier_in = vk::BufferMemoryBarrier::default()
            .buffer(buf_in.buffer).offset(buf_in.pool_offset.unwrap_or(0)).size(buf_in.size)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[barrier_in], &[]);

        let set = backend.pool_desc_act.lock().unwrap().next();
        let info_in  = [vk::DescriptorBufferInfo::default().buffer(buf_in.buffer).offset(buf_in.pool_offset.unwrap_or(0)).range(buf_in.size)];
        let info_out = [vk::DescriptorBufferInfo::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).range(buf_out.size)];
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_in),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_out),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_softmax);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_act, 0, &[set], &[]);

        // Standardized 16-byte push constants: [width, height, is_log, 0]
        let pc_data: [u32; 4] = [width, height, if is_log { 1 } else { 0 }, 0];
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_act, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&pc_data));

        backend.device.cmd_dispatch(cmd, height, 1, 1);

        let barrier_out = vk::BufferMemoryBarrier::default()
            .buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).size(buf_out.size)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);

        backend.device.cmd_copy_buffer(cmd, buf_out.buffer, stage_out.buffer, &[vk::BufferCopy { src_offset: buf_out.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_f32 }]);
             backend.device.end_command_buffer(cmd).unwrap();

        let wait_val = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(std::slice::from_ref(&wait_val));

        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .push_next(&mut timeline_info)
            .command_buffers(&cmds)
            .signal_semaphores(std::slice::from_ref(&backend.timeline_semaphore));

        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();

        backend.pending_ops.lock().unwrap().push(AsyncOp {
            staging_buffers: vec![stage_in, stage_out.copy_for_async()],
            device_buffers: vec![buf_in, buf_out],
            cmd_buffer: cmd,
            desc_set: set,
            wait_id: wait_val,
        });

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&backend.timeline_semaphore))
            .values(std::slice::from_ref(&wait_val));
        backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap();

        download_from_stage(output_raw, &stage_out, dtype);
    }
    poll_async_ops();
}

pub fn execute_linear_into(a_raw: &[u8], b_raw: &[u8], bias_raw: &[u8], res_raw: &mut [u8], m: u32, k: u32, n: u32, act_type: u32, transpose_b: u32, dtype: DataType) {
    let backend = BACKEND.get().unwrap();
    let num_bytes_f32_a = (m * k * 4) as vk::DeviceSize;
    let num_bytes_f32_b = (k * n * 4) as vk::DeviceSize;
    let num_bytes_f32_c = (m * n * 4) as vk::DeviceSize;
    let num_bytes_f32_bias = (n * 4) as vk::DeviceSize;

    let buf_a = get_buffer(num_bytes_f32_a, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Linear_A"), false);
    let buf_b = get_buffer(num_bytes_f32_b, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Linear_B"), false);
    let buf_bias = get_buffer(num_bytes_f32_bias, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Linear_Bias"), false);
    let buf_c = get_buffer(num_bytes_f32_c, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Linear_C"), false);

    let stage_a = get_buffer(num_bytes_f32_a, vk::BufferUsageFlags::TRANSFER_SRC, Some("Linear_Stage_A"), true);
    let stage_b = get_buffer(num_bytes_f32_b, vk::BufferUsageFlags::TRANSFER_SRC, Some("Linear_Stage_B"), true);
    let stage_bias = get_buffer(num_bytes_f32_bias, vk::BufferUsageFlags::TRANSFER_SRC, Some("Linear_Stage_Bias"), true);
    let stage_c = get_buffer_readback(num_bytes_f32_c, vk::BufferUsageFlags::TRANSFER_DST, Some("Linear_Stage_C"));

    upload_to_stage(a_raw, &stage_a, dtype);
    upload_to_stage(b_raw, &stage_b, dtype);
    if bias_raw.is_empty() {
        let ptr = stage_bias.mapped_ptr.unwrap() as *mut f32;
        let dst_slice = unsafe { std::slice::from_raw_parts_mut(ptr, n as usize) };
        dst_slice.fill(0.0);
    } else {
        upload_to_stage(bias_raw, &stage_bias, dtype);
    }

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_a.buffer, buf_a.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_a.pool_offset.unwrap_or(0), size: num_bytes_f32_a }]);
        backend.device.cmd_copy_buffer(cmd, stage_b.buffer, buf_b.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_b.pool_offset.unwrap_or(0), size: num_bytes_f32_b }]);
        backend.device.cmd_copy_buffer(cmd, stage_bias.buffer, buf_bias.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_bias.pool_offset.unwrap_or(0), size: num_bytes_f32_bias }]);

        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).size(buf_a.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).size(buf_b.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_bias.buffer).offset(buf_bias.pool_offset.unwrap_or(0)).size(buf_bias.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        let set = backend.pool_desc_linear.lock().unwrap().next();
        let info_a = [vk::DescriptorBufferInfo::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).range(buf_a.size)];
        let info_b = [vk::DescriptorBufferInfo::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).range(buf_b.size)];
        let info_c = [vk::DescriptorBufferInfo::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).range(buf_c.size)];
        let info_bias = [vk::DescriptorBufferInfo::default().buffer(buf_bias.buffer).offset(buf_bias.pool_offset.unwrap_or(0)).range(buf_bias.size)];

        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_a),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_b),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_c),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_bias),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_linear);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_linear, 0, &[set], &[]);

        let pc_data = [m, k, n, act_type, transpose_b];
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_linear, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&pc_data));

        backend.device.cmd_dispatch(cmd, (n + 15) / 16, (m + 15) / 16, 1);

        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).size(buf_c.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);

        backend.device.cmd_copy_buffer(cmd, buf_c.buffer, stage_c.buffer, &[vk::BufferCopy { src_offset: buf_c.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_f32_c }]);

        backend.device.end_command_buffer(cmd).unwrap();
        let wait_val = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(std::slice::from_ref(&wait_val));
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default().push_next(&mut timeline_info).command_buffers(&cmds).signal_semaphores(std::slice::from_ref(&backend.timeline_semaphore));

        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();

        backend.pending_ops.lock().unwrap().push(AsyncOp {
            staging_buffers: vec![stage_a, stage_b, stage_bias, stage_c.copy_for_async()],
            device_buffers: vec![buf_a, buf_b, buf_bias, buf_c],
            cmd_buffer: cmd,
            desc_set: set,
            wait_id: wait_val,
        });

        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(std::slice::from_ref(&backend.timeline_semaphore)).values(std::slice::from_ref(&wait_val));
        backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap();
        download_from_stage(res_raw, &stage_c, dtype);
    }

    poll_async_ops();
}

pub fn execute_matmul_into(a_raw: &[u8], b_raw: &[u8], res_raw: &mut [u8], m: u32, k: u32, n: u32, dtype: DataType) {
    let backend = BACKEND.get().unwrap();
    let num_bytes_f32_a = (m * k * 4) as vk::DeviceSize;
    let num_bytes_f32_b = (k * n * 4) as vk::DeviceSize;
    let num_bytes_f32_c = (m * n * 4) as vk::DeviceSize;

    let buf_a = get_buffer(num_bytes_f32_a, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_A"), false);
    let buf_b = get_buffer(num_bytes_f32_b, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_B"), false);
    let buf_c = get_buffer(num_bytes_f32_c, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_C"), false);

    let stage_a = get_buffer(num_bytes_f32_a, vk::BufferUsageFlags::TRANSFER_SRC, Some("MatMul_Stage_A"), true);
    let stage_b = get_buffer(num_bytes_f32_b, vk::BufferUsageFlags::TRANSFER_SRC, Some("MatMul_Stage_B"), true);
    let stage_c = get_buffer_readback(num_bytes_f32_c, vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_Stage_C"));

    upload_to_stage(a_raw, &stage_a, dtype);
    upload_to_stage(b_raw, &stage_b, dtype);

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_a.buffer, buf_a.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_a.pool_offset.unwrap_or(0), size: num_bytes_f32_a }]);
        backend.device.cmd_copy_buffer(cmd, stage_b.buffer, buf_b.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_b.pool_offset.unwrap_or(0), size: num_bytes_f32_b }]);

        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).size(buf_a.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).size(buf_b.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        let set = backend.pool_desc_matmul.lock().unwrap().next();
        let info_a = [vk::DescriptorBufferInfo::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).range(buf_a.size)];
        let info_b = [vk::DescriptorBufferInfo::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).range(buf_b.size)];
        let info_c = [vk::DescriptorBufferInfo::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).range(buf_c.size)];

        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_a),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_b),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_c),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_matmul);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_matmul, 0, &[set], &[]);

        let pc_data = [m, k, n];
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_matmul, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&pc_data));

        backend.device.cmd_dispatch(cmd, (n + 15) / 16, (m + 15) / 16, 1);

        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).size(buf_c.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);

        backend.device.cmd_copy_buffer(cmd, buf_c.buffer, stage_c.buffer, &[vk::BufferCopy { src_offset: buf_c.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_f32_c }]);

        backend.device.end_command_buffer(cmd).unwrap();
        let wait_val = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(std::slice::from_ref(&wait_val));
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default().push_next(&mut timeline_info).command_buffers(&cmds).signal_semaphores(std::slice::from_ref(&backend.timeline_semaphore));

        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();

        backend.pending_ops.lock().unwrap().push(AsyncOp {
            staging_buffers: vec![stage_a, stage_b, stage_c.copy_for_async()],
            device_buffers: vec![buf_a, buf_b, buf_c],
            cmd_buffer: cmd,
            desc_set: set,
            wait_id: wait_val,
        });

        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(std::slice::from_ref(&backend.timeline_semaphore)).values(std::slice::from_ref(&wait_val));
        backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap();
        download_from_stage(res_raw, &stage_c, dtype);
    }

    poll_async_ops();
}

pub fn execute_bit_linear_into(a_raw: &[u8], b_raw: &[u8], s_raw: &[u8], bias_raw: &[u8], out_raw: &mut [u8], m: u32, k: u32, n: u32) {
    let backend = BACKEND.get().unwrap();
    let num_bytes_a = (m * k) as vk::DeviceSize;
    let num_bytes_b = (n * k) as vk::DeviceSize;
    let num_bytes_s = (n * 4) as vk::DeviceSize;
    let num_bytes_bias = (n * 4) as vk::DeviceSize;
    let num_bytes_out = (m * n * 4) as vk::DeviceSize;

    let buf_a = get_buffer(num_bytes_a, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("BitLinear_A"), false);
    let buf_b = get_buffer(num_bytes_b, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("BitLinear_B"), false);
    let buf_s = get_buffer(num_bytes_s, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("BitLinear_S"), false);
    let buf_bias = get_buffer(num_bytes_bias, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("BitLinear_Bias"), false);
    let buf_out = get_buffer(num_bytes_out, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("BitLinear_Out"), false);

    let stage_a = get_buffer(num_bytes_a, vk::BufferUsageFlags::TRANSFER_SRC, Some("BitLinear_Stage_A"), true);
    let stage_b = get_buffer(num_bytes_b, vk::BufferUsageFlags::TRANSFER_SRC, Some("BitLinear_Stage_B"), true);
    let stage_s = get_buffer(num_bytes_s, vk::BufferUsageFlags::TRANSFER_SRC, Some("BitLinear_Stage_S"), true);
    let stage_bias = get_buffer(num_bytes_bias, vk::BufferUsageFlags::TRANSFER_SRC, Some("BitLinear_Stage_Bias"), true);
    let stage_out = get_buffer_readback(num_bytes_out, vk::BufferUsageFlags::TRANSFER_DST, Some("BitLinear_Stage_Out"));

    upload_to_stage_raw(a_raw, &stage_a);
    upload_to_stage_raw(b_raw, &stage_b);
    upload_to_stage_raw(s_raw, &stage_s);
    if bias_raw.is_empty() {
        let ptr = stage_bias.mapped_ptr.unwrap() as *mut f32;
        unsafe { std::slice::from_raw_parts_mut(ptr, n as usize).fill(0.0); }
    } else {
        upload_to_stage_raw(bias_raw, &stage_bias);
    }

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_a.buffer, buf_a.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_a.pool_offset.unwrap_or(0), size: num_bytes_a }]);
        backend.device.cmd_copy_buffer(cmd, stage_b.buffer, buf_b.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_b.pool_offset.unwrap_or(0), size: num_bytes_b }]);
        backend.device.cmd_copy_buffer(cmd, stage_s.buffer, buf_s.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_s.pool_offset.unwrap_or(0), size: num_bytes_s }]);
        backend.device.cmd_copy_buffer(cmd, stage_bias.buffer, buf_bias.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_bias.pool_offset.unwrap_or(0), size: num_bytes_bias }]);

        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).size(buf_a.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).size(buf_b.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_s.buffer).offset(buf_s.pool_offset.unwrap_or(0)).size(buf_s.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_bias.buffer).offset(buf_bias.pool_offset.unwrap_or(0)).size(buf_bias.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        let set = backend.pool_desc_bit_linear.lock().unwrap().next();
        let info_a = [vk::DescriptorBufferInfo::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).range(buf_a.size)];
        let info_b = [vk::DescriptorBufferInfo::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).range(buf_b.size)];
        let info_s = [vk::DescriptorBufferInfo::default().buffer(buf_s.buffer).offset(buf_s.pool_offset.unwrap_or(0)).range(buf_s.size)];
        let info_bias = [vk::DescriptorBufferInfo::default().buffer(buf_bias.buffer).offset(buf_bias.pool_offset.unwrap_or(0)).range(buf_bias.size)];
        let info_out = [vk::DescriptorBufferInfo::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).range(buf_out.size)];

        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_a),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_b),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_s),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_bias),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_out),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_bit_linear);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_bit_linear, 0, &[set], &[]);

        let has_bias = if bias_raw.is_empty() { 0u32 } else { 1u32 };
        let pc_data = [m, k, n, has_bias];
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_bit_linear, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&pc_data));

        backend.device.cmd_dispatch(cmd, (n + 15) / 16, (m + 15) / 16, 1);

        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).size(buf_out.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);

        backend.device.cmd_copy_buffer(cmd, buf_out.buffer, stage_out.buffer, &[vk::BufferCopy { src_offset: buf_out.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_out }]);

        backend.device.end_command_buffer(cmd).unwrap();
        let wait_val = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(std::slice::from_ref(&wait_val));
        let submit_info = vk::SubmitInfo::default().push_next(&mut timeline_info).command_buffers(std::slice::from_ref(&cmd)).signal_semaphores(std::slice::from_ref(&backend.timeline_semaphore));
        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();

        backend.pending_ops.lock().unwrap().push(AsyncOp {
            staging_buffers: vec![stage_a, stage_b, stage_s, stage_bias, stage_out.copy_for_async()],
            device_buffers: vec![buf_a, buf_b, buf_s, buf_bias, buf_out],
            cmd_buffer: cmd,
            desc_set: set,
            wait_id: wait_val,
        });

        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(std::slice::from_ref(&backend.timeline_semaphore)).values(std::slice::from_ref(&wait_val));
        backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap();
        download_from_stage_raw(out_raw, &stage_out);
    }
    poll_async_ops();
}

#[allow(dead_code)]
pub fn wait_for_all() {
    let backend = BACKEND.get().unwrap();
    let current = backend.timeline_value.load(Ordering::SeqCst);
    if current > 0 {
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&backend.timeline_semaphore))
            .values(std::slice::from_ref(&current));
        unsafe { backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap(); }
    }
    poll_async_ops();
}
