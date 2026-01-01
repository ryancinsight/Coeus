//! WGSL Compute Shaders for RMSprop Optimizer Kernels
struct RMSpropUniforms {
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: u32,
    param_count: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> parameters: array<f32>;
@group(0) @binding(3) var<storage, read_write> square_avg: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_avg: array<f32>;
@group(0) @binding(5) var<storage, read_write> momentum_buffer: array<f32>;
@group(0) @binding(6) var<uniform> rmsprop_config: RMSpropUniforms;

@compute @workgroup_size(256)
fn sparse_rmsprop_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&indices)) {
        return;
    }

    let pos = indices[idx];
    if (pos >= arrayLength(&parameters) || pos >= arrayLength(&square_avg) || pos >= arrayLength(&grad_avg) || pos >= arrayLength(&momentum_buffer)) {
        return;
    }

    let grad = gradients[idx];
    let param = parameters[pos];

    // Apply weight decay
    let effective_grad = grad + rmsprop_config.weight_decay * param;

    // Update square average: square_avg = alpha * square_avg + (1 - alpha) * grad^2
    let old_square_avg = square_avg[pos];
    let new_square_avg = rmsprop_config.alpha * old_square_avg + (1.0 - rmsprop_config.alpha) * effective_grad * effective_grad;
    square_avg[pos] = new_square_avg;

    // Compute denominator
    var denom = sqrt(new_square_avg) + rmsprop_config.eps;

    // For centered RMSprop: denom = sqrt(square_avg - grad_avg^2 + eps)
    if (rmsprop_config.centered == 1u) {
        // Update grad_avg: grad_avg = alpha * grad_avg + (1 - alpha) * grad
        let old_grad_avg = grad_avg[pos];
        let new_grad_avg = rmsprop_config.alpha * old_grad_avg + (1.0 - rmsprop_config.alpha) * effective_grad;
        grad_avg[pos] = new_grad_avg;

        // Centered denominator: sqrt(square_avg - grad_avg^2 + eps)
        let grad_avg_sq = new_grad_avg * new_grad_avg;
        let centered_value = new_square_avg - grad_avg_sq;
        denom = sqrt(max(centered_value, 0.0)) + rmsprop_config.eps;
    }

    let grad_scaled = effective_grad / denom;
    if (rmsprop_config.momentum > 0.0) {
        let velocity = momentum_buffer[pos];
        let new_velocity = rmsprop_config.momentum * velocity + grad_scaled;
        momentum_buffer[pos] = new_velocity;
        parameters[pos] = param - rmsprop_config.lr * new_velocity;
    } else {
        parameters[pos] = param - rmsprop_config.lr * grad_scaled;
    }
}

@compute @workgroup_size(256)
fn dense_rmsprop_batch_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let param_idx = global_id.x;
    if (rmsprop_config.param_count == 0u || param_idx >= rmsprop_config.param_count) {
        return;
    }

    if (param_idx >= arrayLength(&parameters) || param_idx >= arrayLength(&square_avg) || param_idx >= arrayLength(&grad_avg) || param_idx >= arrayLength(&momentum_buffer)) {
        return;
    }

    let batch_size = arrayLength(&gradients) / rmsprop_config.param_count;
    if (batch_size == 0u) {
        return;
    }
    var grad_sum = 0.0;

    // Compute gradient average over batch
    for (var batch = 0u; batch < batch_size; batch = batch + 1u) {
        let grad_idx = batch * rmsprop_config.param_count + param_idx;
        grad_sum = grad_sum + gradients[grad_idx];
    }
    let avg_grad = grad_sum / f32(batch_size);

    let param = parameters[param_idx];

    // Apply weight decay
    let effective_grad = avg_grad + rmsprop_config.weight_decay * param;

    // Update square average: square_avg = alpha * square_avg + (1 - alpha) * grad^2
    let old_square_avg = square_avg[param_idx];
    let new_square_avg = rmsprop_config.alpha * old_square_avg + (1.0 - rmsprop_config.alpha) * effective_grad * effective_grad;
    square_avg[param_idx] = new_square_avg;

    // Compute denominator
    var denom = sqrt(new_square_avg) + rmsprop_config.eps;

    // For centered RMSprop: denom = sqrt(square_avg - grad_avg^2 + eps)
    if (rmsprop_config.centered == 1u) {
        // Update grad_avg: grad_avg = alpha * grad_avg + (1 - alpha) * grad
        let old_grad_avg = grad_avg[param_idx];
        let new_grad_avg = rmsprop_config.alpha * old_grad_avg + (1.0 - rmsprop_config.alpha) * effective_grad;
        grad_avg[param_idx] = new_grad_avg;

        // Centered denominator: sqrt(square_avg - grad_avg^2 + eps)
        let grad_avg_sq = new_grad_avg * new_grad_avg;
        let centered_value = new_square_avg - grad_avg_sq;
        denom = sqrt(max(centered_value, 0.0)) + rmsprop_config.eps;
    }

    let grad_scaled = effective_grad / denom;
    if (rmsprop_config.momentum > 0.0) {
        let velocity = momentum_buffer[param_idx];
        let new_velocity = rmsprop_config.momentum * velocity + grad_scaled;
        momentum_buffer[param_idx] = new_velocity;
        parameters[param_idx] = param - rmsprop_config.lr * new_velocity;
    } else {
        parameters[param_idx] = param - rmsprop_config.lr * grad_scaled;
    }
}
