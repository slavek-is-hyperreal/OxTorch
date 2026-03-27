use std::fs;
use std::sync::OnceLock;
use ash::Entry;

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub has_avx2: bool,
    pub has_neon: bool,
    pub ram_total_gb: f64,
    pub ram_available_gb: f64,
    pub gpu_name: String,
    pub vulkan_api_version: String,
    pub has_fp16: bool,
    pub has_cooperative_matrix: bool,
    pub is_nvme: bool,
}

pub static SYS_INFO: OnceLock<SystemInfo> = OnceLock::new();

pub fn get_sys_info() -> &'static SystemInfo {
    SYS_INFO.get_or_init(detect_system)
}

fn detect_system() -> SystemInfo {
    let mut info = SystemInfo {
        cpu_model: "Unknown".to_string(),
        cpu_cores: 0,
        has_avx2: false,
        has_neon: false,
        ram_total_gb: 0.0,
        ram_available_gb: 0.0,
        gpu_name: "None/CPU".to_string(),
        vulkan_api_version: "N/A".to_string(),
        has_fp16: false,
        has_cooperative_matrix: false,
        is_nvme: false,
    };

    // 1. CPU Detection
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = fs::read_to_string("/proc/cpuinfo") {
            for line in content.lines() {
                if line.starts_with("model name") {
                    info.cpu_model = line.split(':').nth(1).unwrap_or("Unknown").trim().to_string();
                }
                if line.starts_with("processor") {
                    info.cpu_cores += 1;
                }
                if line.starts_with("flags") || line.starts_with("Features") {
                    info.has_avx2 = line.contains("avx2");
                    info.has_neon = line.contains("neon") || line.contains("asimd");
                }
            }
        }
    }

    // 2. RAM Detection
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            let mut mem_avail = 0;
            let mut mem_total = 0;
            for line in content.lines() {
                if line.starts_with("MemAvailable:") {
                    mem_avail = line.split_whitespace().nth(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0) * 1024;
                }
                if line.starts_with("MemTotal:") {
                    mem_total = line.split_whitespace().nth(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0) * 1024;
                }
            }
            info.ram_total_gb = mem_total as f64 / 1024.0 / 1024.0 / 1024.0;
            
            // Logika OxTorch: rezerwa 2GB i uwzględnienie Capacitora (jeśli już istnieje)
            let mut avail_bytes = mem_avail;
            if let Some(cap) = crate::tensor::capacitor::GLOBAL_CAPACITOR.get() {
                avail_bytes += cap.capacity;
            }
            info.ram_available_gb = avail_bytes.saturating_sub(2 * 1024 * 1024 * 1024) as f64 / 1024.0 / 1024.0 / 1024.0;
        }
    }

    // 3. GPU / Vulkan Detection (Minimal Ash)
    let entry = unsafe { Entry::load() };
    if let Ok(entry) = entry {
        if let Ok(instance) = unsafe { entry.create_instance(&ash::vk::InstanceCreateInfo::default(), None) } {
            if let Ok(pdevices) = unsafe { instance.enumerate_physical_devices() } {
                if let Some(&pdevice) = pdevices.first() {
                    let props = unsafe { instance.get_physical_device_properties(pdevice) };
                    info.gpu_name = unsafe {
                        std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                            .to_string_lossy()
                            .into_owned()
                    };
                    info.vulkan_api_version = format!(
                        "{}.{}.{}",
                        ash::vk::api_version_major(props.api_version),
                        ash::vk::api_version_minor(props.api_version),
                        ash::vk::api_version_patch(props.api_version)
                    );

                    // Check for FP16 / Coop Matrix (properly via features2 and pointers)
                    let mut feat16 = ash::vk::PhysicalDeviceShaderFloat16Int8FeaturesKHR::default();
                    let mut feat2 = ash::vk::PhysicalDeviceFeatures2::default();
                    feat2.p_next = &mut feat16 as *mut _ as *mut std::ffi::c_void;
                    
                    unsafe { instance.get_physical_device_features2(pdevice, &mut feat2) };
                    info.has_fp16 = feat16.shader_float16 == ash::vk::TRUE;
                    
                    // Cooperative matrix check requires extension enumeration (simplified here)
                    if let Ok(exts) = unsafe { instance.enumerate_device_extension_properties(pdevice) } {
                        for ext in exts {
                            let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()).to_string_lossy() };
                            if name == "VK_KHR_cooperative_matrix" || name == "VK_NV_cooperative_matrix" {
                                info.has_cooperative_matrix = true;
                            }
                        }
                    }
                }
            }
            unsafe { instance.destroy_instance(None) };
        }
    }

    // 4. SSD Detection (NVMe vs SATA)
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = fs::read_dir("/sys/block") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with("nvme") {
                    info.is_nvme = true;
                    break;
                }
            }
        }
    }

    info
}

pub fn print_sys_info() {
    let info = get_sys_info();
    println!("--- OxTorch System Diagnostic ---");
    println!("CPU: {} ({} cores, AVX2: {}, NEON: {})", info.cpu_model, info.cpu_cores, info.has_avx2, info.has_neon);
    println!("RAM: {:.2} GB Available / {:.2} GB Total", info.ram_available_gb, info.ram_total_gb);
    println!("GPU: {} (Vulkan {})", info.gpu_name, info.vulkan_api_version);
    println!("Features: FP16: {}, CoopMatrix: {}", info.has_fp16, info.has_cooperative_matrix);
    println!("Disk: NVMe Detected: {}", info.is_nvme);
    println!("---------------------------------");
}
