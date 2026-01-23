struct Uniforms {
    scale: f32,
    zero_point: f32,
    bits: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {
        return;
    }
    
    // Unpack based on bitwidth
    var quantized: u32;
    if (uniforms.bits == 4u) {
        let byte_idx = idx / 2u;
        let is_high_nibble = (idx % 2u) == 0u;
        if (is_high_nibble) {
            quantized = (input[byte_idx] >> 4u) & 0x0Fu;
        } else {
            quantized = input[byte_idx] & 0x0Fu;
        }
    } else if (uniforms.bits == 8u) {
        quantized = input[idx] & 0xFFu;
    } else if (uniforms.bits == 16u) {
        let word_idx = idx / 2u;
        let is_low_word = (idx % 2u) == 0u;
        if (is_low_word) {
            quantized = input[word_idx] & 0xFFFFu;
        } else {
            quantized = (input[word_idx] >> 16u) & 0xFFFFu;
        }
    }
    
    // Dequantize: x = (q - zero_point) * scale
    output[idx] = (f32(quantized) - uniforms.zero_point) * uniforms.scale;
}
