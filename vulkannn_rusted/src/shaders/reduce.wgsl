@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

struct PushConstants {
    total_elements: u32,
    stride: u32,      // if needed for dim reduce, later
    size_d: u32,      // if needed for dim reduce, later
}
var<push_constant> pc: PushConstants;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    let tid = local_id.x;

    if (index < pc.total_elements) {
        sdata[tid] = in_data[index];
    } else {
        sdata[tid] = 0.0;
    }
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let block_idx = group_id.y * num_workgroups.x + group_id.x;
        out_data[block_idx] = sdata[0];
    }
}

@compute @workgroup_size(256)
fn max_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    let tid = local_id.x;

    if (index < pc.total_elements) {
        sdata[tid] = in_data[index];
    } else {
        sdata[tid] = -3.402823466e+38; // ~neg_inf
    }
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let block_idx = group_id.y * num_workgroups.x + group_id.x;
        out_data[block_idx] = sdata[0];
    }
}

@compute @workgroup_size(256)
fn min_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    let tid = local_id.x;

    if (index < pc.total_elements) {
        sdata[tid] = in_data[index];
    } else {
        sdata[tid] = 3.402823466e+38; // ~pos_inf
    }
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let block_idx = group_id.y * num_workgroups.x + group_id.x;
        out_data[block_idx] = sdata[0];
    }
}
