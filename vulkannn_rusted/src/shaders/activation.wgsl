@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

@compute @workgroup_size(64)
fn relu_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 64u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        out_data[index] = max(0.0, in_data[index]);
    }
}

@compute @workgroup_size(64)
fn sigmoid_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 64u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        let x = in_data[index];
        out_data[index] = 1.0 / (1.0 + exp(-x));
    }
}

@compute @workgroup_size(64)
fn silu_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 64u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        let x = in_data[index];
        out_data[index] = x * (1.0 / (1.0 + exp(-x)));
    }
}
