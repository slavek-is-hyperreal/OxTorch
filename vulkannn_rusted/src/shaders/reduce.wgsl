@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

struct PushConstants {
    total_elements: u32,
    stride: u32,
    size_d: u32,
}
var<push_constant> pc: PushConstants;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    let tid = local_id.x;

    var val: f32 = 0.0;
    if (index < pc.total_elements) {
        val = in_data[index];
    }
    
    // Subgroup reduction (warp-level)
    let sg_val = subgroupAdd(val);
    
    // First thread of each subgroup writes to shared memory
    if (subgroup_local_id == 0u) {
        sdata[subgroup_id] = sg_val;
    }
    
    workgroupBarrier();

    // Final reduction by the first subgroup
    if (subgroup_id == 0u) {
        let num_subgroups = (256u + subgroup_size - 1u) / subgroup_size;
        var final_val: f32 = 0.0;
        if (tid < num_subgroups) {
            final_val = sdata[tid];
        }
        
        let res = subgroupAdd(final_val);
        if (tid == 0u) {
            let block_idx = group_id.y * num_workgroups.x + group_id.x;
            out_data[block_idx] = res;
        }
    }
}

@compute @workgroup_size(256)
fn max_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    let tid = local_id.x;

    var val: f32 = -3.402823466e+38; 
    if (index < pc.total_elements) {
        val = in_data[index];
    }
    
    let sg_val = subgroupMax(val);
    if (subgroup_local_id == 0u) {
        sdata[subgroup_id] = sg_val;
    }
    workgroupBarrier();

    if (subgroup_id == 0u) {
        let num_subgroups = (256u + subgroup_size - 1u) / subgroup_size;
        var final_val: f32 = -3.402823466e+38;
        if (tid < num_subgroups) {
            final_val = sdata[tid];
        }
        let res = subgroupMax(final_val);
        if (tid == 0u) {
            let block_idx = group_id.y * num_workgroups.x + group_id.x;
            out_data[block_idx] = res;
        }
    }
}

@compute @workgroup_size(256)
fn min_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
) {
    let index = global_id.y * (num_workgroups.x * 256u) + global_id.x;
    let tid = local_id.x;

    var val: f32 = 3.402823466e+38; 
    if (index < pc.total_elements) {
        val = in_data[index];
    }
    
    let sg_val = subgroupMin(val);
    if (subgroup_local_id == 0u) {
        sdata[subgroup_id] = sg_val;
    }
    workgroupBarrier();

    if (subgroup_id == 0u) {
        let num_subgroups = (256u + subgroup_size - 1u) / subgroup_size;
        var final_val: f32 = 3.402823466e+38;
        if (tid < num_subgroups) {
            final_val = sdata[tid];
        }
        let res = subgroupMin(final_val);
        if (tid == 0u) {
            let block_idx = group_id.y * num_workgroups.x + group_id.x;
            out_data[block_idx] = res;
        }
    }
}
