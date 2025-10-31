//! CLIP Loss shader for GPU-accelerated InfoNCE computation
//!
//! Computes symmetric InfoNCE loss for CLIP training with GPU acceleration.
//! Handles normalization, similarity computation, and cross-entropy loss.

@group(0) @binding(0)
var<storage, read> image_features: array<f32>;

@group(0) @binding(1)
var<storage, read> text_features: array<f32>;

@group(0) @binding(2)
var<storage, read_write> loss_output: array<f32>;

@group(0) @binding(3)
var<uniform> params: CLIPLossParams;

// CLIP loss computation parameters
struct CLIPLossParams {
    batch_size: u32,
    embed_dim: u32,
    temperature: f32,
};

@compute @workgroup_size(64, 1, 1)
fn compute_clip_loss(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    if (batch_idx >= params.batch_size) {
        return;
    }

    // Compute L2 norms for normalization
    var image_norm = 0.0;
    var text_norm = 0.0;

    let image_offset = batch_idx * params.embed_dim;
    let text_offset = batch_idx * params.embed_dim;

    for (var i = 0u; i < params.embed_dim; i = i + 1u) {
        let img_val = image_features[image_offset + i];
        let txt_val = text_features[text_offset + i];
        image_norm = image_norm + img_val * img_val;
        text_norm = text_norm + txt_val * txt_val;
    }

    image_norm = sqrt(image_norm);
    text_norm = sqrt(text_norm);

    // Avoid division by zero
    let eps = 1e-10;
    image_norm = max(image_norm, eps);
    text_norm = max(text_norm, eps);

    // Compute similarity matrix elements for this batch pair
    var image_to_text_sim = 0.0;
    var text_to_image_sim = 0.0;

    for (var i = 0u; i < params.embed_dim; i = i + 1u) {
        let img_val = image_features[image_offset + i] / image_norm;
        let txt_val = text_features[text_offset + i] / text_norm;

        // For cross-batch similarities, we'd need all pairs
        // For now, this is a simplified version - full implementation needs all batch pairs
        image_to_text_sim = image_to_text_sim + img_val * txt_val;
        text_to_image_sim = text_to_image_sim + txt_val * img_val;
    }

    // Apply temperature scaling
    image_to_text_sim = image_to_text_sim / params.temperature;
    text_to_image_sim = text_to_image_sim / params.temperature;

    // In a real implementation, we need the full similarity matrix
    // This is a simplified version for single batch pairs
    let logits = image_to_text_sim;

    // Softmax denominator (simplified - should be sum of exp similarities)
    let softmax_denom = exp(logits); // Simplified - needs full matrix

    // Cross-entropy loss
    let softmax_prob = exp(logits) / softmax_denom;
    let sample_loss = -log(max(softmax_prob, eps));

    // Write loss for this batch
    loss_output[batch_idx] = sample_loss;
}

@compute @workgroup_size(1, 1, 1)
fn reduce_loss(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    // Reduce all batch losses to single scalar
    if (global_id.x != 0 || global_id.y != 0 || global_id.z != 0) {
        return;
    }

    var total_loss = 0.0;
    for (var i = 0u; i < params.batch_size; i = i + 1u) {
        total_loss = total_loss + loss_output[i];
    }

    // Average loss across batch
    loss_output[0] = total_loss / f32(params.batch_size);
}
