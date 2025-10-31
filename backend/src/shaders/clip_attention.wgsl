//! CLIP Attention shader for GPU-accelerated transformer operations
//!
//! This shader implements efficient attention computation for CLIP vision and text transformers,
//! supporting both self-attention and cross-attention patterns.

@group(0) @binding(0)
var<storage, read> queries: array<f32>;

@group(0) @binding(1)
var<storage, read> keys: array<f32>;

@group(0) @binding(2)
var<storage, read> values: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@group(0) @binding(4)
var<uniform> params: AttentionParams;

// Attention parameters for flexible computation
struct AttentionParams {
    batch_size: u32,
    seq_len_q: u32,
    seq_len_kv: u32,
    embed_dim: u32,
    num_heads: u32,
    head_dim: u32,
    scale_factor: f32, // 1/sqrt(head_dim)
};

@compute @workgroup_size(8, 8, 1)
fn clip_attention(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let batch_idx = group_id.z;
    let head_idx = global_id.z;
    let query_pos = global_id.y;
    let embed_pos = global_id.x;

    // Bounds checking
    if (batch_idx >= params.batch_size ||
        head_idx >= params.num_heads ||
        query_pos >= params.seq_len_q ||
        embed_pos >= params.head_dim) {
        return;
    }

    // Compute indices for multi-head attention
    let head_offset = head_idx * params.head_dim;
    let query_offset = batch_idx * params.seq_len_q * params.embed_dim +
                      query_pos * params.embed_dim + head_offset;

    // For each query position, compute attention over all key-value positions
    var attention_scores = array<f32, 1024>(); // Max sequence length, adjust as needed
    var max_score = -3.4028234663852886e+38; // -FLT_MAX

    // First pass: compute attention logits and find max for numerical stability
    for (var kv_pos = 0u; kv_pos < params.seq_len_kv; kv_pos = kv_pos + 1u) {
        let key_offset = batch_idx * params.seq_len_kv * params.embed_dim +
                        kv_pos * params.embed_dim + head_offset;

        var dot_product = 0.0;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            let q_val = queries[query_offset + d];
            let k_val = keys[key_offset + d];
            dot_product = dot_product + q_val * k_val;
        }

        // Apply scaling
        let scaled_score = dot_product * params.scale_factor;
        attention_scores[kv_pos] = scaled_score;

        if (scaled_score > max_score) {
            max_score = scaled_score;
        }
    }

    // Second pass: compute softmax weights
    var weight_sum = 0.0;
    for (var kv_pos = 0u; kv_pos < params.seq_len_kv; kv_pos = kv_pos + 1u) {
        let exp_score = exp(attention_scores[kv_pos] - max_score);
        attention_scores[kv_pos] = exp_score;
        weight_sum = weight_sum + exp_score;
    }

    // Third pass: apply attention weights to values and accumulate output
    var result = 0.0;
    for (var kv_pos = 0u; kv_pos < params.seq_len_kv; kv_pos = kv_pos + 1u) {
        let value_offset = batch_idx * params.seq_len_kv * params.embed_dim +
                          kv_pos * params.embed_dim + head_offset;

        let weight = attention_scores[kv_pos] / weight_sum;
        result = result + weight * values[value_offset + embed_pos];
    }

    // Write output
    let output_offset = batch_idx * params.seq_len_q * params.embed_dim +
                       query_pos * params.embed_dim + head_offset + embed_pos;
    output[output_offset] = result;
}
