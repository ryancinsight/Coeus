struct Uniforms {
    m: u32,
    k: u32,
    n: u32,
    lhs_scale: f32,
    lhs_zero_point: f32,
    rhs_scale: f32,
    rhs_zero_point: f32,
    bits: u32,
}

@group(0) @binding(0) var<storage, read> lhs: array<u32>; // Quantized LHS
@group(0) @binding(1) var<storage, read> rhs: array<u32>; // Quantized RHS
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // FP32 output
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> bias: array<f32>; // Optional bias

fn dequantize_lhs(quantized: u32) -> f32 {
    return (f32(quantized) - uniforms.lhs_zero_point) * uniforms.lhs_scale;
}

fn dequantize_rhs(quantized: u32) -> f32 {
    return (f32(quantized) - uniforms.rhs_zero_point) * uniforms.rhs_scale;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    
    if (row >= uniforms.m || col >= uniforms.n) {
        return;
    }
    
    // Compute dot product: C[row, col] = sum(A[row, i] * B[i, col])
    var sum = 0.0;
    for (var i = 0u; i < uniforms.k; i = i + 1u) {
        let lhs_idx = row * uniforms.k + i;
        let rhs_idx = i * uniforms.n + col;
        
        // Extract quantized values (8-bit example)
        let lhs_quantized = lhs[lhs_idx] & 0xFFu;
        let rhs_quantized = rhs[rhs_idx] & 0xFFu;
        
        // Dequantize and accumulate
        sum = sum + dequantize_lhs(lhs_quantized) * dequantize_rhs(rhs_quantized);
    }
    
    // Add bias if present
    let has_bias = arrayLength(&bias) > 0u;
    if (has_bias) {
        sum = sum + bias[col];
    }
    
    output[row * uniforms.n + col] = sum;
}
