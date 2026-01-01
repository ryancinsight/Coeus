//! GPU kernels for sparse matrix operations and automatic differentiation
//!
//! These WGSL shaders implement high-performance sparse matrix operations
//! for GPU-accelerated automatic differentiation, providing significant
//! speedups for sparse tensor computations.

struct CsrMatrix {
    rows: u32,
    cols: u32,
    nnz: u32,
}

struct CooMatrix {
    rows: u32,
    cols: u32,
    nnz: u32,
}

@group(0) @binding(0) var<storage, read> csr_data: array<f32>;
@group(0) @binding(1) var<storage, read> csr_indices: array<u32>;
@group(0) @binding(2) var<storage, read> csr_indptr: array<u32>;
@group(0) @binding(3) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(4) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(5) var<uniform> matrix_info: CsrMatrix;

@compute @workgroup_size(256, 1, 1)
fn spmm_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= matrix_info.rows) { return; }

    let row_start = csr_indptr[row];
    let row_end = csr_indptr[row + 1];

    for (var idx = row_start; idx < row_end; idx = idx + 1u) {
        let col_a = csr_indices[idx];
        let val_a = csr_data[idx];
        let b_row_start = col_a * matrix_info.cols;

        for (var col_c = 0u; col_c < matrix_info.cols; col_c = col_c + 1u) {
            let b_val = matrix_b[b_row_start + col_c];
            let c_index = row * matrix_info.cols + col_c;
            matrix_c[c_index] = matrix_c[c_index] + val_a * b_val;
        }
    }
}

// ... Additional kernels from original file can be added as needed
