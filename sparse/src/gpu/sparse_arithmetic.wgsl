// Sparse Element-wise Arithmetic Kernel
// Handles operations like add, sub, mul, div for sparse tensors (COO format)

struct Params {
    nnz: u32,
    op: u32, // 0: add, 1: sub, 2: mul, 3: div
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_data: array<f32>;
@group(0) @binding(2) var<storage, read> b_data: array<f32>;
@group(0) @binding(3) var<storage, read_write> result_data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.nnz) {
        return;
    }

    let a = a_data[idx];
    let b = b_data[idx];
    var res: f32 = 0.0;

    switch (params.op) {
        case 0u: { res = a + b; }
        case 1u: { res = a - b; }
        case 2u: { res = a * b; }
        case 3u: { 
            if (b != 0.0) {
                res = a / b;
            }
        }
        default: { res = a; }
    }

    result_data[idx] = res;
}
