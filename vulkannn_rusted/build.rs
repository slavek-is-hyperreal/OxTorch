use std::fs;

fn main() {
    // 1. Compile Shaders (existing logic)
    let shader_dir = "src/shaders";
    let entries = fs::read_dir(shader_dir).expect("Failed to read shaders directory");

    for entry in entries {
        let entry = entry.expect("Failed to read directory entry");
        let path_buf = entry.path();
        let path = path_buf.to_str().unwrap();

        if !path.ends_with(".wgsl") && !path.ends_with(".comp") {
            continue;
        }

        println!("cargo:rerun-if-changed={}", path);
        let log_msg = format!("Processing shader: {}\n", path);
        let mut file = std::fs::OpenOptions::new().append(true).create(true).open("/tmp/vnn_build.log").unwrap();
        use std::io::Write;
        file.write_all(log_msg.as_bytes()).unwrap();
        
        let source = fs::read_to_string(path).unwrap();
        
        let module = if path.ends_with(".wgsl") {
            match naga::front::wgsl::parse_str(&source) {
                Ok(m) => m,
                Err(e) => panic!("Failed to parse WGSL {}: {:?}", path, e),
            }
        } else if path.ends_with(".comp") {
            let mut parser = naga::front::glsl::Frontend::default();
            let options = naga::front::glsl::Options {
                stage: naga::ShaderStage::Compute,
                defines: Default::default(),
            };
            match parser.parse(&options, &source) {
                Ok(m) => m,
                Err(e) => {
                    let err_msg = format!("NAGA PARSE ERROR in {}: {:?}\n", path, e);
                    let mut file = std::fs::OpenOptions::new().append(true).create(true).open("/tmp/vnn_build.log").unwrap();
                    file.write_all(err_msg.as_bytes()).unwrap();
                    panic!("Failed to parse GLSL {}: {:?}", path, e);
                }
            }
        } else {
            panic!("Unsupported shader extension: {}", path);
        };
        
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = match validator.validate(&module) {
            Ok(i) => i,
            Err(e) => panic!("Failed to validate {}: {:?}", path, e),
        };

        let mut flags = naga::back::spv::WriterFlags::empty();
        flags.set(naga::back::spv::WriterFlags::DEBUG, true);
        
        let options = naga::back::spv::Options {
            lang_version: (1, 3), // SPV 1.3
            flags,
            ..Default::default()
        };

        let spv = match naga::back::spv::write_vec(
            &module,
            &info,
            &options,
            None,
        ) {
            Ok(s) => s,
            Err(e) => panic!("Failed to write SPV for {}: {:?}", path, e),
        };

        let out_path = format!("{}.spv", path);
        let bytes = unsafe {
            std::slice::from_raw_parts(
                spv.as_ptr() as *const u8,
                spv.len() * 4,
            )
        };
        fs::write(&out_path, bytes).unwrap();
    }

    // 2. MSTS Hardware Dispatch Constants
    // These are used to "burn-in" optimal thresholds for the current CPU.
    println!("cargo:rerun-if-env-changed=MSTS_TILE_BYTES");
    println!("cargo:rerun-if-env-changed=MSTS_RING_DEPTH");

    let l2_kb = read_cache_size(2).unwrap_or(256); // Default 256KB L2
    let l3_mb = read_cache_size(3).map(|k| k / 1024).unwrap_or(6); // Default 6MB L3

    // Logic from docs/implementation_guides.md
    let direct_max = (l3_mb * 1024 * 1024) / 2;                // 50% of L3
    let tile_small = (l2_kb * 1024 * 3) / 4;                   // 75% of L2
    let ring_small = 2usize;                                   // ping-pong buffer
    let tile_large = 8 * 1024 * 1024usize;                      // 8MB (for high-speed RAID/NVMe)
    let ring_large = std::cmp::min(l3_mb / 2, 8).max(2);       // bound by L3

    // Allow override via ENV (for build server)
    let direct_max = std::env::var("MSTS_DIRECT_MAX").map(|s| s.parse().unwrap()).unwrap_or(direct_max);
    let tile_small = std::env::var("MSTS_TILE_SMALL").map(|s| s.parse().unwrap()).unwrap_or(tile_small);
    let ring_small = std::env::var("MSTS_RING_SMALL").map(|s| s.parse().unwrap()).unwrap_or(ring_small);
    let tile_large = std::env::var("MSTS_TILE_BYTES").map(|s| s.parse().unwrap()).unwrap_or(tile_large);
    let ring_large = std::env::var("MSTS_RING_DEPTH").map(|s| s.parse().unwrap()).unwrap_or(ring_large);

    // 3. Static Disk Detection (NVMe vs SATA ZFS)
    // Read .env to find our target cache path
    let env_content = fs::read_to_string(".env").unwrap_or_default();
    let mut cache_path = String::new();
    for line in env_content.lines() {
        if line.starts_with("VNN_CACHE_DIR=") {
            cache_path = line.split('=').nth(1).unwrap_or("").to_string();
            break;
        }
    }

    let is_ssd = cache_path.to_lowercase().contains("ssd") || check_if_nvme();
    let uring_depth = if is_ssd { 64u32 } else { 16u32 }; // SATA SSD depth (64) is safer for ZFS

    // 4. Capacitor Static Safety Floor (50% of build-time RAM)
    let mem_total_kb = read_mem_total().unwrap_or(8 * 1024 * 1024); // Default 8GB
    let capacitor_floor_mb = (mem_total_kb / 1024) / 4; // 25% safety floor

    println!("cargo:rustc-env=MSTS_DIRECT_MAX={}", direct_max);
    println!("cargo:rustc-env=MSTS_TILE_SMALL={}", tile_small);
    println!("cargo:rustc-env=MSTS_RING_SMALL={}", ring_small);
    println!("cargo:rustc-env=MSTS_TILE_BYTES={}", tile_large);
    println!("cargo:rustc-env=MSTS_RING_DEPTH={}", ring_large);
    println!("cargo:rustc-env=VNN_URING_DEPTH={}", uring_depth);
    println!("cargo:rustc-env=VNN_CAPACITOR_FLOOR_MB={}", capacitor_floor_mb);

    // Also write a constants file to OUT_DIR for easier inclusion
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("msts_constants.rs");
    fs::write(&dest_path, format!(
        "pub const DIRECT_MAX: usize = {};\n\
         pub const TILE_SMALL: usize = {};\n\
         pub const RING_SMALL: usize = {};\n\
         pub const TILE_LARGE: usize = {};\n\
         pub const RING_LARGE: usize = {};\n\
         pub const URING_DEPTH: u32 = {};\n\
         pub const CAPACITOR_FLOOR_MB: usize = {};\n",
        direct_max, tile_small, ring_small, tile_large, ring_large, uring_depth, capacitor_floor_mb
    )).unwrap();
}

fn check_if_nvme() -> bool {
    if let Ok(entries) = fs::read_dir("/sys/block") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("nvme") { return true; }
        }
    }
    false
}

fn read_mem_total() -> Option<usize> {
    if let Ok(content) = fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                return line.split_whitespace().nth(1).and_then(|s| s.parse::<usize>().ok());
            }
        }
    }
    None
}

fn read_cache_size(level: u32) -> Option<usize> {
    // Standard Linux path for CPU 0 cache info
    // levels are usually: index1=L1, index2=L2, index3=L3
    let index = match level {
        2 => 2,
        3 => 3,
        _ => return None,
    };
    let path = format!("/sys/devices/system/cpu/cpu0/cache/index{}/size", index);
    if let Ok(content) = fs::read_to_string(path) {
        let s = content.trim();
        if s.ends_with('K') {
            return s.trim_end_matches('K').parse::<usize>().ok();
        } else if s.ends_with('M') {
            return s.trim_end_matches('M').parse::<usize>().map(|v| v * 1024).ok();
        }
    }
    None
}
