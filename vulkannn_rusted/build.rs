use std::fs;

fn main() {
    let wgsl_files = [
        "src/shaders/add.wgsl",
        "src/shaders/matmul.wgsl",
        "src/shaders/activation.wgsl",
        "src/shaders/reduce.wgsl",
    ];

    for path in wgsl_files {
        println!("cargo:rerun-if-changed={}", path);
        let source = fs::read_to_string(path).unwrap();
        let module = match naga::front::wgsl::parse_str(&source) {
            Ok(m) => m,
            Err(e) => panic!("Failed to parse {}: {:?}", path, e),
        };
        
        // Disable strict validation just to ensure we can parse what wgpu was parsing
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
}
