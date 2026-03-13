// Softmax compute shader (WGSL) — written natively for reliable naga compilation.
// Each workgroup handles one row; 256 threads per workgroup.
// Barrier-based parallel reduction for max and sum.

struct PushConsts {
    width:  u32,
    height: u32,
    _pad1:  u32,
    _pad2:  u32,
}

@group(0) @binding(0) var<storage, read>       in_buf:  array<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<f32>;

var<push_constant> params: PushConsts;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id)       wg_id:    vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row: u32 = wg_id.x;
    let tid: u32 = local_id.x;
    let width: u32 = params.width;

    if row >= params.height { return; }

    let base: u32 = row * width;

    // ── Pass 1: row max ──────────────────────────────────────────────────────
    var local_max: f32 = -1.0e38;
    var i: u32 = tid;
    while i < width {
        local_max = max(local_max, in_buf[base + i]);
        i += 256u;
    }
    shared_data[tid] = local_max;
    workgroupBarrier();

    var s: u32 = 128u;
    while s > 0u {
        if tid < s {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let row_max: f32 = shared_data[0];

    // ── Pass 2: sum of exp(x - max) ──────────────────────────────────────────
    var local_sum: f32 = 0.0;
    i = tid;
    while i < width {
        local_sum += exp(in_buf[base + i] - row_max);
        i += 256u;
    }
    shared_data[tid] = local_sum;
    workgroupBarrier();

    s = 128u;
    while s > 0u {
        if tid < s {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let row_sum: f32 = shared_data[0];

    // ── Pass 3: normalize and write ──────────────────────────────────────────
    i = tid;
    while i < width {
        out_buf[base + i] = exp(in_buf[base + i] - row_max) / row_sum;
        i += 256u;
    }
}
