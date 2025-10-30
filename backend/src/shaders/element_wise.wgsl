// Element-wise mathematical operations shader
// Performs unary operations on input arrays

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> op_type: u32; // 0: log, 1: sin, 2: cos, 3: exp, etc.

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }

    let x = input[index];

    // Perform operation based on op_type
    var result = x; // default passthrough
    if (op_type == 0u) {
        // Natural logarithm: log(x)
        // Handle domain: x > 0
        if (x > 0.0) {
            result = log(x);
        } else {
            result = -999999.0; // NaN equivalent for invalid input
        }
    } else if (op_type == 1u) {
        // Sine: sin(x)
        result = sin(x);
    } else if (op_type == 2u) {
        // Cosine: cos(x)
        result = cos(x);
    } else if (op_type == 3u) {
        // Exponential: exp(x)
        result = exp(x);
    } else if (op_type == 4u) {
        // Square root: sqrt(x)
        if (x >= 0.0) {
            result = sqrt(x);
        } else {
            result = -999999.0; // NaN for negative input
        }
    } else if (op_type == 5u) {
        // Tanh: tanh(x)
        result = tanh(x);
    } else if (op_type == 6u) {
        // Sigmoid: 1 / (1 + exp(-x))
        result = 1.0 / (1.0 + exp(-x));
    } else if (op_type == 7u) {
        // ReLU: max(0, x)
        result = max(0.0, x);
    }

    output[index] = result;
}
