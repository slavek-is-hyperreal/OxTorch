@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

struct PushConstants {
    num_elements: f32,
    reserved: f32,
    param1: f32,
    param2: f32,
}
var<push_constant> pc: PushConstants;

@compute @workgroup_size(256)
fn relu_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        out_data[index] = max(0.0, in_data[index]);
    }
}

@compute @workgroup_size(256)
fn sigmoid_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        let x = in_data[index];
        out_data[index] = 1.0 / (1.0 + exp(-x));
    }
}

@compute @workgroup_size(256)
fn silu_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        let x = in_data[index];
        out_data[index] = x * (1.0 / (1.0 + exp(-x)));
    }
}

@compute @workgroup_size(256)
fn gelu_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        let x = in_data[index];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_pi = 0.7978845608028654;
        let inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
        let exp2_inner = exp(2.0 * inner);
        let tanh_inner = (exp2_inner - 1.0) / (exp2_inner + 1.0);
        out_data[index] = 0.5 * x * (1.0 + tanh_inner);
    }
}

@compute @workgroup_size(256)
fn leaky_relu_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        let x = in_data[index];
        out_data[index] = select(pc.param1 * x, x, x > 0.0);
    }
}

@compute @workgroup_size(256)
fn elu_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        let x = in_data[index];
        out_data[index] = select(pc.param1 * (exp(x) - 1.0), x, x > 0.0);
    }
}

@compute @workgroup_size(256)
fn tanh_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        let x = in_data[index];
        let e2x = exp(2.0 * x);
        out_data[index] = (e2x - 1.0) / (e2x + 1.0);
    }
}

@compute @workgroup_size(256)
fn clamp_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        let x = in_data[index];
        out_data[index] = clamp(x, pc.param1, pc.param2);
    }
}

@compute @workgroup_size(256)
fn neg_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        out_data[index] = -in_data[index];
    }
}

@compute @workgroup_size(256)
fn pow_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    if (index < arrayLength(&out_data)) {
        out_data[index] = pow(in_data[index], pc.param1);
    }
}
