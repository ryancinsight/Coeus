struct ConvUniforms {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    input_h: u32,
    input_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    output_h: u32,
    output_w: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: ConvUniforms;

@compute @workgroup_size(8, 8, 1) // Dispatch over Output H, Output W
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_w_idx = global_id.x;
    let out_h_idx = global_id.y;
    let out_c_idx = global_id.z; // We might need to dispatch Z as (batch * out_channels)

    // Since workgroup size is small, we map Z to batch * out_channels
    // But typical dispatch limit: we can dispatch (W, H, B*C).
    
    // Decompose z index
    let batch_idx = out_c_idx / uniforms.out_channels;
    let channel_idx = out_c_idx % uniforms.out_channels;

    if (out_w_idx >= uniforms.output_w || out_h_idx >= uniforms.output_h || batch_idx >= uniforms.batch_size) {
        return;
    }

    // Convolution: sum over in_channels, kernel_h, kernel_w
    var sum = 0.0;
    
    // Precompute input base offset: batch * in_channels * input_h * input_w
    let input_batch_offset = batch_idx * uniforms.in_channels * uniforms.input_h * uniforms.input_w;
    
    // Precompute weight base offset for this output channel: channel_idx * in_channels * kh * kw
    let weight_out_c_offset = channel_idx * uniforms.in_channels * uniforms.kernel_h * uniforms.kernel_w;

    for (var ic = 0u; ic < uniforms.in_channels; ic = ic + 1u) {
        let input_channel_offset = input_batch_offset + ic * uniforms.input_h * uniforms.input_w;
        let weight_in_c_offset = weight_out_c_offset + ic * uniforms.kernel_h * uniforms.kernel_w;

        for (var ky = 0u; ky < uniforms.kernel_h; ky = ky + 1u) {
            for (var kx = 0u; kx < uniforms.kernel_w; kx = kx + 1u) {
                let in_y = out_h_idx + ky; // stride=1, padding=0 implies direct mapping + kernel offset? No.
                // Conv definition: sum x[i+k] * w[k]. 
                // So input pixel is (out_h + ky), (out_w + kx).
                // No, padding would shift this. With padding=0, stride=1:
                // input(y, x) corresponds to output(y, x) top-left corner.
                
                // Input index
                let input_idx = input_channel_offset + in_y * uniforms.input_w + (out_w_idx + kx);
                
                // Weight index
                let weight_idx = weight_in_c_offset + ky * uniforms.kernel_w + kx;
                
                sum = sum + input[input_idx] * weight[weight_idx];
            }
        }
    }

    // Output index: batch, out_c, out_h, out_w
    let output_idx = 
        batch_idx * uniforms.out_channels * uniforms.output_h * uniforms.output_w +
        channel_idx * uniforms.output_h * uniforms.output_w +
        out_h_idx * uniforms.output_w +
        out_w_idx;

    output[output_idx] = sum;
}
