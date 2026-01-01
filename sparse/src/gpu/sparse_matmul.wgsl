// Sparse Matrix-Matrix Multiplication (SpMM) Kernel
// CSR format: Sparse (A) x Dense (B) = Dense (C)

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_data: array<f32>;
@group(0) @binding(2) var<storage, read> a_indices: array<u32>;
@group(0) @binding(3) var<storage, read> a_indptr: array<u32>;
@group(0) @binding(4) var<storage, read> b_data: array<f32>;
@group(0) @binding(5) var<storage, read_write> c_data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.m) {
        return;
    }

    let start = a_indptr[row];
    let end = a_indptr[row + 1];

    // For each column in dense matrix B
    for (var col: u32 = 0u; col < params.n; col++) {
        var sum: f32 = 0.0;
        
        // Iterate over non-zeros in row of A
        for (var i: u32 = start; i < end; i++) {
            let a_col = a_indices[i];
            let a_val = a_data[i];
            
            // B is dense [k, n], row-major
            let b_val = b_data[a_col * params.n + col];
            sum += a_val * b_val;
        }
        
        c_data[row * params.n + col] = sum;
    }
}
