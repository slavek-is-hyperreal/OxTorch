use std::fs;

fn main() {
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
