//! GPU kernels for sparse matrix operations and automatic differentiation
//!
//! These WGSL shaders implement high-performance sparse matrix operations
//! for GPU-accelerated automatic differentiation, providing significant
//! speedups for sparse tensor computations.

/// Structure for sparse matrix data in CSR format
struct CsrMatrix {
    /// Number of rows in the matrix
    rows: u32,
    /// Number of columns in the matrix
    cols: u32,
    /// Number of non-zero elements
    nnz: u32,
}

/// Structure for sparse matrix data in COO format
struct CooMatrix {
    /// Number of rows in the matrix
    rows: u32,
    /// Number of columns in the matrix
    cols: u32,
    /// Number of non-zero elements
    nnz: u32,
}

/// Compute sparse-dense matrix multiplication: C = A @ B
/// where A is sparse (CSR format) and B is dense
///
/// Workgroup dimensions: [workgroup_size_x, workgroup_size_y, 1]
/// Each workgroup processes one row of the output matrix
@group(0) @binding(0)
var<storage, read> csr_data: array<f32>;      // Non-zero values of A

@group(0) @binding(1)
var<storage, read> csr_indices: array<u32>;   // Column indices of A

@group(0) @binding(2)
var<storage, read> csr_indptr: array<u32>;    // Row pointers of A

@group(0) @binding(3)
var<storage, read> matrix_b: array<f32>;      // Dense matrix B

@group(0) @binding(4)
var<storage, read_write> matrix_c: array<f32>; // Output matrix C

@group(0) @binding(5)
var<uniform> matrix_info: CsrMatrix;         // Matrix metadata

/// Sparse-dense matrix multiplication kernel
@compute
@workgroup_size(256, 1, 1)
fn spmm_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.x;

    // Bounds check
    if (row >= matrix_info.rows) {
        return;
    }

    // Get row bounds for this sparse matrix row
    let row_start = csr_indptr[row];
    let row_end = csr_indptr[row + 1];

    // Process each non-zero element in this row
    for (var idx = row_start; idx < row_end; idx = idx + 1u) {
        let col_a = csr_indices[idx];
        let val_a = csr_data[idx];

        // Multiply with corresponding column of matrix B
        let b_row_start = col_a * matrix_info.cols;  // Assume B has same cols as A->B transformation

        // Sum over all columns of result matrix
        for (var col_c = 0u; col_c < matrix_info.cols; col_c = col_c + 1u) {
            let b_val = matrix_b[b_row_start + col_c];
            let c_index = row * matrix_info.cols + col_c;
            matrix_c[c_index] = matrix_c[c_index] + val_a * b_val;
        }
    }
}

/// Element-wise sparse gradient accumulation
///
/// Accumulates gradients from multiple sources into a sparse result
@group(1) @binding(0)
var<storage, read> grad_accumulators: array<f32>;    // Gradient values to accumulate

@group(1) @binding(1)
var<storage, read> grad_row_indices: array<u32>;    // Row indices for accumulation

@group(1) @binding(2)
var<storage, read> grad_col_indices: array<u32>;    // Column indices for accumulation

@group(1) @binding(3)
var<storage, read_write> sparse_grad_output: array<f32>; // Accumulated sparse gradients

/// Sparse gradient accumulation kernel
@compute
@workgroup_size(256, 1, 1)
fn sparse_grad_accumulate_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let grad_idx = global_id.x;

    // Accumulate gradients for each non-zero position
    let row = grad_row_indices[grad_idx];
    let col = grad_col_indices[grad_idx];
    let orig_matrix_cols = 1024u; // This should be passed as uniform

    let flat_idx = row * orig_matrix_cols + col;
    sparse_grad_output[flat_idx] = sparse_grad_output[flat_idx] + grad_accumulators[grad_idx];
}

/// Compute sparse element-wise operations (activation functions)
///
/// Applies activation functions element-wise to sparse matrices
@group(2) @binding(0)
var<storage, read> sparse_input: array<f32>;       // Sparse input values

@group(2) @binding(1)
var<storage, read_write> sparse_output: array<f32>; // Sparse output values

/// Element-wise tangent (activation derivative)
@compute
@workgroup_size(256, 1, 1)
fn sparse_tanh_backward_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Only process non-zero elements (input should be filtered)
    if (idx < arrayLength(&sparse_input)) {
        let input_val = sparse_input[idx];
        // tanh derivative: 1 - tanh²(x)
        let tanh_val = tanh(input_val);
        let derivative = 1.0 - tanh_val * tanh_val;

        sparse_output[idx] = derivative;
    }
}

/// Element-wise sigmoid (activation derivative)
@compute
@workgroup_size(256, 1, 1)
fn sparse_sigmoid_backward_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Only process non-zero elements
    if (idx < arrayLength(&sparse_input)) {
        let input_val = sparse_input[idx];
        // sigmoid derivative: σ(x) * (1 - σ(x))
        let sigmoid_val = 1.0 / (1.0 + exp(-input_val));
        let derivative = sigmoid_val * (1.0 - sigmoid_val);

        sparse_output[idx] = derivative;
    }
}

/// Element-wise ReLU (activation derivative)
@compute
@workgroup_size(256, 1, 1)
fn sparse_relu_backward_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Only process non-zero elements
    if (idx < arrayLength(&sparse_input)) {
        let input_val = sparse_input[idx];
        // ReLU derivative: 1 if x > 0, 0 otherwise
        let derivative = select(0.0, 1.0, input_val > 0.0);

        sparse_output[idx] = derivative;
    }
}

/// Sparse matrix transpose operation
///
/// Transposes sparse matrix between CSR and CSC formats
@group(3) @binding(0)
var<storage, read> transpose_input_data: array<f32>;

@group(3) @binding(1)
var<storage, read> transpose_input_indices: array<u32>;

@group(3) @binding(2)
var<storage, read> transpose_input_indptr: array<u32>;

@group(3) @binding(3)
var<storage, read_write> transpose_output_data: array<f32>;

@group(3) @binding(4)
var<storage, read_write> transpose_output_indices: array<u32>;

@group(3) @binding(5)
var<storage, read_write> transpose_output_indptr: array<u32>;

@group(3) @binding(6)
var<uniform> transpose_info: CsrMatrix;

/// Sparse matrix transpose kernel
@compute
@workgroup_size(256, 1, 1)
fn sparse_transpose_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let element_idx = global_id.x;

    if (element_idx >= transpose_info.nnz) {
        return;
    }

    // For CSR->CSC transpose, swap row/col indices
    let row = element_idx;  // In CSR context, this represents the sparse element position
    let data_val = transpose_input_data[element_idx];
    let col_idx = transpose_input_indices[element_idx];

    // Store transposed values
    transpose_output_data[element_idx] = data_val;
    transpose_output_indices[element_idx] = row; // Original row becomes new column
    // Note: indptr needs separate pass to rebuild
}

/// Build row pointers for transposed sparse matrix
@compute
@workgroup_size(256, 1, 1)
fn build_transpose_indptr_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;

    if (row > transpose_info.cols) {
        return;
    }

    // Count elements per column in original matrix to build new indptr
    var count = 0u;
    for (var i = 0u; i < transpose_info.nnz; i = i + 1u) {
        if (transpose_input_indices[i] == row) {
            count = count + 1u;
        }
    }

    // This is a simplification - real implementation needs atomic operations
    transpose_output_indptr[row] = count;
}

/// Sparse matrix addition kernel
///
/// Adds two sparse matrices element-wise
@group(4) @binding(0)
var<storage, read> add_a_data: array<f32>;

@group(4) @binding(1)
var<storage, read> add_a_rows: array<u32>;

@group(4) @binding(2)
var<storage, read> add_a_cols: array<u32>;

@group(4) @binding(3)
var<storage, read> add_b_data: array<f32>;

@group(4) @binding(4)
var<storage, read> add_b_rows: array<u32>;

@group(4) @binding(5)
var<storage, read> add_b_cols: array<u32>;

@group(4) @binding(6)
var<storage, read_write> add_result_data: array<f32>;

@group(4) @binding(7)
var<storage, read_write> add_result_rows: array<u32>;

@group(4) @binding(8)
var<storage, read_write> add_result_cols: array<u32>;

/// Sparse matrix addition kernel (simplified - assumes matching sparsity patterns)
@compute
@workgroup_size(256, 1, 1)
fn sparse_add_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Assume equal sparsity patterns for simplicity
    let max_nnz = min(arrayLength(&add_a_data), arrayLength(&add_b_data));

    if (idx < max_nnz &&
        add_a_rows[idx] == add_b_rows[idx] &&
        add_a_cols[idx] == add_b_cols[idx]) {

        // Add elements at same positions
        add_result_data[idx] = add_a_data[idx] + add_b_data[idx];
        add_result_rows[idx] = add_a_rows[idx];
        add_result_cols[idx] = add_a_cols[idx];
    }
}

/// Sparse matrix scaling (scalar multiplication)
@group(5) @binding(0)
var<storage, read> scale_input: array<f32>;

@group(5) @binding(1)
var<storage, read_write> scale_output: array<f32>;

@group(5) @binding(2)
var<uniform> scale_factor: f32;

/// Scalar multiplication kernel for sparse matrices
@compute
@workgroup_size(256, 1, 1)
fn sparse_scale_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx < arrayLength(&scale_input)) {
        scale_output[idx] = scale_input[idx] * scale_factor;
    }
}

/// Compute sparse matrix Frobenius norm (for gradient clipping)
///
/// Returns the norm of the sparse matrix for regularization
@group(6) @binding(0)
var<storage, read> norm_input: array<f32>;

@group(6) @binding(1)
var<storage, read_write> norm_output: array<f32>; // Single element

/// Sparse matrix norm computation kernel
@compute
@workgroup_size(256, 1, 1)
fn sparse_norm_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // Shared memory for workgroup reduction
    var<workgroup> shared_sum: array<f32, 256>;

    let tid = local_id.x;
    let gid = global_id.x;

    var sum_squares = 0.0f;

    // Compute sum of squares for this thread's elements
    if (gid < arrayLength(&norm_input)) {
        let val = norm_input[gid];
        sum_squares = val * val;
    }

    // Store in shared memory
    shared_sum[tid] = sum_squares;
    workgroupBarrier();

    // Parallel reduction within workgroup
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        stride = stride >> 1u;
        workgroupBarrier();
    }

    // First thread in workgroup writes result
    if (tid == 0u) {
        let workgroup_idx = workgroup_id.x;
        norm_output[workgroup_idx] = shared_sum[0];
    }
}

/// Sparse matrix element-wise square operation
@group(7) @binding(0)
var<storage, read> square_input: array<f32>;

@group(7) @binding(1)
var<storage, read_write> square_output: array<f32>;

/// Element-wise square kernel for sparse matrices
@compute
@workgroup_size(256, 1, 1)
fn sparse_square_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx < arrayLength(&square_input)) {
        let val = square_input[idx];
        square_output[idx] = val * val;
    }
}

/// Sparse matrix gradient clipping by global norm
///
/// Clips gradients to prevent exploding gradients in sparse matrices
@group(8) @binding(0)
var<storage, read> clip_input: array<f32>;

@group(8) @binding(1)
var<storage, read> global_norm: array<f32>; // Single element

@group(8) @binding(2)
var<storage, read> clip_threshold: array<f32>; // Single element

@group(8) @binding(3)
var<storage, read_write> clip_output: array<f32>;

/// Gradient clipping kernel for sparse matrices
@compute
@workgroup_size(256, 1, 1)
fn sparse_gradient_clip_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let max_clip_ratio = clip_threshold[0];
    let current_norm = global_norm[0];

    if (idx < arrayLength(&clip_input)) {
        let grad_val = clip_input[idx];

        // Clip gradient if norm exceeds threshold
        var clipped_val = grad_val;
        if (current_norm > max_clip_ratio) {
            let scale_factor = max_clip_ratio / current_norm;
            clipped_val = grad_val * scale_factor;
        }

        clip_output[idx] = clipped_val;
    }
}
