struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Uniforms;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    let M = dims.M;
    let K = dims.K;
    let N = dims.N;

    var sum: f32 = 0.0;
    
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile A
        let k_idx = t * TILE_SIZE + local_col;
        if (row < M && k_idx < K) {
            tile_a[local_row][local_col] = a[row * K + k_idx];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        // Load tile B
        let b_row_idx = t * TILE_SIZE + local_row;
        if (b_row_idx < K && col < N) {
            tile_b[local_row][local_col] = b[b_row_idx * N + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_row][k] * tile_b[k][local_col];
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        c[row * N + col] = c[row * N + col] + sum;
    }
}
