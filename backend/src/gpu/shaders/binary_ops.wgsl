// Binary element-wise operations shader
// Performs binary operations (add, multiply) on two input arrays

@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> op_type: u32; // 0: add, 1: multiply

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&lhs)) {
        return;
    }

    let a = lhs[index];
    let b = rhs[index];

    // Perform operation based on op_type
    var result = a; // default to lhs
    if (op_type == 0u) {
        // Addition: a + b
        result = a + b;
    } else if (op_type == 1u) {
        // Multiplication: a * b
        result = a * b;
    }

    output[index] = result;
}
