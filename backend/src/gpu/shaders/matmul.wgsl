// Matrix multiplication shader
// Computes C = A * B where A is (M x K) and B is (K x N)
//
// Uniform buffer layout:
// - dims: [M, K, N] - matrix dimensions
//
// Storage buffers:
// - lhs: array<f32> - matrix A (M x K) in row-major order
// - rhs: array<f32> - matrix B (K x N) in row-major order
// - output: array<f32> - matrix C (M x N) in row-major order

@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> dims: vec3<u32>; // [M, K, N]

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x; // 0 to M-1
    let col = global_id.y; // 0 to N-1

    let M = dims.x;
    let K = dims.y;
    let N = dims.z;

    // Check bounds
    if (row >= M || col >= N) {
        return;
    }

    // Compute dot product of row 'row' from lhs and column 'col' from rhs
    var sum = 0.0;
    for (var k = 0u; k < K; k = k + 1u) {
        let lhs_idx = row * K + k;
        let rhs_idx = k * N + col;
        sum = sum + lhs[lhs_idx] * rhs[rhs_idx];
    }

    // Store result
    let output_idx = row * N + col;
    output[output_idx] = sum;
}
