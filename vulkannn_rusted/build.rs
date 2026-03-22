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
                Err(e) => panic!("Failed to parse GLSL {}: {:?}", path, e),
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

    println!("cargo:rustc-env=MSTS_DIRECT_MAX={}", direct_max);
    println!("cargo:rustc-env=MSTS_TILE_SMALL={}", tile_small);
    println!("cargo:rustc-env=MSTS_RING_SMALL={}", ring_small);
    println!("cargo:rustc-env=MSTS_TILE_BYTES={}", tile_large);
    println!("cargo:rustc-env=MSTS_RING_DEPTH={}", ring_large);

    // Also write a constants file to OUT_DIR for easier inclusion
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("msts_constants.rs");
    fs::write(&dest_path, format!(
        "pub const DIRECT_MAX: usize = {};\npub const TILE_SMALL: usize = {};\npub const RING_SMALL: usize = {};\npub const TILE_LARGE: usize = {};\npub const RING_LARGE: usize = {};\n",
        direct_max, tile_small, ring_small, tile_large, ring_large
    )).unwrap();
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
