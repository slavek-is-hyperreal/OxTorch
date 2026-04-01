use std::fs;
use std::sync::{OnceLock, Mutex};

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

pub static SYS_INFO_STATIC: OnceLock<SystemInfo> = OnceLock::new();
pub static GPU_INFO_VOLATILE: Mutex<(String, String)> = Mutex::new((String::new(), String::new()));

pub fn get_sys_info() -> SystemInfo {
    let mut info = SYS_INFO_STATIC.get_or_init(detect_static_system).clone();
    
    // 1. Dynamic RAM Detection (Always fresh)
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
            info.ram_available_gb = mem_avail.saturating_sub(512 * 1024 * 1024) as f64 / 1024.0 / 1024.0 / 1024.0;
        }
    }

    // 2. Dynamic GPU Info (Updated by backend)
    let gpu = GPU_INFO_VOLATILE.lock().unwrap();
    if !gpu.0.is_empty() {
        info.gpu_name = gpu.0.clone();
        info.vulkan_api_version = gpu.1.clone();
    }

    info
}

pub fn update_gpu_info(name: String, api_version: String) {
    let mut gpu = GPU_INFO_VOLATILE.lock().unwrap();
    *gpu = (name, api_version);
}

fn detect_static_system() -> SystemInfo {
    let mut info = SystemInfo {
        cpu_model: "Unknown".to_string(),
        cpu_cores: 0,
        has_avx2: false,
        has_neon: false,
        ram_total_gb: 0.0,
        ram_available_gb: 0.0,
        gpu_name: "Detection Deferred...".to_string(),
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

    // 2. Static SSD Detection
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
    println!("Disk: NVMe Detected: {}", info.is_nvme);
    println!("---------------------------------");
}
