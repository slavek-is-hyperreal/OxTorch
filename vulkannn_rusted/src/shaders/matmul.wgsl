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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    let M = dims.M;
    let K = dims.K;
    let N = dims.N;

    if (row >= M || col >= N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        sum = sum + a[row * K + k] * b[k * N + col];
    }
    
    c[row * N + col] = sum;
}
