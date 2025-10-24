struct Uniforms {
    scale: f32,
    zero_point: f32,
    bits: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input)) {
        return;
    }
    
    // Quantize: q = round(x / scale + zero_point)
    let scaled = input[idx] / uniforms.scale + uniforms.zero_point;
    let max_val = (1u << uniforms.bits) - 1u;
    let quantized = u32(clamp(scaled, 0.0, f32(max_val)));
    
    // Pack based on bitwidth
    if (uniforms.bits == 4u) {
        let byte_idx = idx / 2u;
        let is_high_nibble = (idx % 2u) == 0u;
        if (is_high_nibble) {
            output[byte_idx] = (output[byte_idx] & 0x0Fu) | (quantized << 4u);
        } else {
            output[byte_idx] = (output[byte_idx] & 0xF0u) | quantized;
        }
    } else if (uniforms.bits == 8u) {
        output[idx] = quantized;
    } else if (uniforms.bits == 16u) {
        let word_idx = idx / 2u;
        let is_low_word = (idx % 2u) == 0u;
        if (is_low_word) {
            output[word_idx] = (output[word_idx] & 0xFFFF0000u) | quantized;
        } else {
            output[word_idx] = (output[word_idx] & 0x0000FFFFu) | (quantized << 16u);
        }
    }
}
