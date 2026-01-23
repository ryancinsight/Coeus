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
    } else if (op_type == 8u) {
        // Tan: tan(x)
        result = tan(x);
    } else if (op_type == 9u) {
        // Asin: asin(x)
        // Domain -1 to 1
        if (x >= -1.0 && x <= 1.0) {
            result = asin(x);
        } else {
            result = 0.0; // Standard behavior or NaN
        }
    } else if (op_type == 10u) {
        // Acos: acos(x)
        // Domain -1 to 1
        if (x >= -1.0 && x <= 1.0) {
            result = acos(x);
        } else {
            result = 0.0;
        }
    } else if (op_type == 11u) {
        // Atan: atan(x)
        result = atan(x);
    } else if (op_type == 12u) {
        // Sinh: sinh(x)
        result = sinh(x);
    } else if (op_type == 13u) {
        // Cosh: cosh(x)
        result = cosh(x);
    } else if (op_type == 14u) {
        // Abs: abs(x)
        result = abs(x);
    } else if (op_type == 15u) {
        // Ceil: ceil(x)
        result = ceil(x);
    } else if (op_type == 16u) {
        // Floor: floor(x)
        result = floor(x);
    } else if (op_type == 17u) {
        // Round: round(x)
        result = round(x);
    } else if (op_type == 18u) {
        // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_over_pi = 0.7978845608;
        let coeff = 0.044715;
        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + coeff * x3);
        result = 0.5 * x * (1.0 + tanh(inner));
    }

    output[index] = result;
}
