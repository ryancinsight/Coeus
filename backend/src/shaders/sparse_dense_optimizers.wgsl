//! WGSL Compute Shaders for Adam and SGD Optimizer Kernels
//!
//! This file contains GPU compute shaders for high-performance sparse and dense
//! optimization algorithms using WebGPU compute pipelines.
//!
//! - Sparse kernels: Update individual parameter indices for sparse gradients (>10% sparsity)
//! - Dense kernels: Batched updates for dense gradients
//! - Adam algorithm: Implements bias-corrected first and second moment updates
//! - SGD: Implements momentum-based stochastic gradient descent

// =============================================================================
// DATA STRUCTURES
// =============================================================================

// Adam uniform buffer (must match Rust struct)
struct AdamUniforms {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
};

// SGD uniform buffer (must match Rust struct)
struct SGDUniforms {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    dampening: f32,
    nesterov: u32,
};

// =============================================================================
// SPARSE ADAM KERNEL
// =============================================================================

@group(0) @binding(0) var<storage, read> sparse_indices: array<u32>;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> parameters: array<f32>;
@group(0) @binding(3) var<storage, read_write> exp_avg: array<f32>;
@group(0) @binding(4) var<storage, read_write> exp_avg_sq: array<f32>;
@group(0) @binding(5) var<uniform> adam_config: AdamUniforms;

@compute @workgroup_size(256)
fn sparse_adam_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&sparse_indices)) {
        return;
    }

    let pos = sparse_indices[idx];
    if (pos >= arrayLength(&parameters)) {
        return;
    }

    let grad = gradients[idx];
    let param = parameters[pos];

    // Apply weight decay
    let effective_grad = grad + adam_config.weight_decay * param;

    // Update biased first moment: m = β₁ * m + (1 - β₁) * g
    let m_old = exp_avg[pos];
    let m_new = adam_config.beta1 * m_old + (1.0 - adam_config.beta1) * effective_grad;
    exp_avg[pos] = m_new;

    // Update biased second moment: v = β₂ * v + (1 - β₂) * g²
    let v_old = exp_avg_sq[pos];
    let v_new = adam_config.beta2 * v_old + (1.0 - adam_config.beta2) * effective_grad * effective_grad;
    exp_avg_sq[pos] = v_new;

    // Bias-corrected moments
    let step_float = f32(adam_config.step + 1u);
    let bias_correction1 = 1.0 / (1.0 - pow(adam_config.beta1, step_float));
    let bias_correction2 = 1.0 / (1.0 - pow(adam_config.beta2, step_float));

    let m_hat = m_new * bias_correction1;
    let v_hat = v_new * bias_correction2;

    // Parameter update: θ = θ - α * m̂ / (√v̂ + ε)
    let denom = sqrt(v_hat) + adam_config.eps;
    let update = adam_config.lr * m_hat / denom;

    parameters[pos] = param - update;
}

// =============================================================================
// SPARSE SGD KERNEL
// =============================================================================

@group(0) @binding(0) var<storage, read> sgd_sparse_indices: array<u32>;
@group(0) @binding(1) var<storage, read> sgd_gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> sgd_parameters: array<f32>;
@group(0) @binding(3) var<storage, read_write> momentum_buffer: array<f32>;
@group(0) @binding(4) var<uniform> sgd_config: SGDUniforms;

// =============================================================================
// RMSprop CONFIGURATION
// =============================================================================

// RMSprop uniform buffer (must match Rust struct)
struct RMSpropUniforms {
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: u32,
    param_count: u32,
};

@compute @workgroup_size(256)
fn sparse_sgd_momentum_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&sgd_sparse_indices)) {
        return;
    }

    let pos = sgd_sparse_indices[idx];
    if (pos >= arrayLength(&sgd_parameters)) {
        return;
    }

    let grad = sgd_gradients[idx];
    let param = sgd_parameters[pos];
    let velocity = momentum_buffer[pos];

    // Apply weight decay
    let effective_grad = grad + sgd_config.weight_decay * param;

    // Momentum update
    let new_velocity = sgd_config.momentum * velocity + (1.0 - sgd_config.dampening) * effective_grad;
    momentum_buffer[pos] = new_velocity;

    var update_grad = new_velocity;

    // Nesterov momentum
    if (sgd_config.nesterov == 1u) {
        update_grad = sgd_config.momentum * new_velocity + effective_grad;
    }

    // Parameter update
    let update = sgd_config.lr * update_grad;
    sgd_parameters[pos] = param - update;
}

// =============================================================================
// DENSE ADAM KERNEL
// =============================================================================

@group(0) @binding(0) var<storage, read> dense_gradients: array<f32>; // [batch_size, param_count]
@group(0) @binding(1) var<storage, read_write> dense_parameters: array<f32>; // [param_count]
@group(0) @binding(2) var<storage, read_write> dense_exp_avg: array<f32>; // [param_count]
@group(0) @binding(3) var<storage, read_write> dense_exp_avg_sq: array<f32>; // [param_count]
@group(0) @binding(4) var<uniform> dense_adam_config: AdamUniforms;

@compute @workgroup_size(256)
fn dense_adam_batch_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let param_idx = global_id.x;
    if (param_idx >= dense_adam_config.param_count) {
        return;
    }

    let batch_size = arrayLength(&dense_gradients) / dense_adam_config.param_count;
    var grad_sum = 0.0;

    // Compute gradient average over batch
    for (var batch = 0u; batch < batch_size; batch = batch + 1u) {
        let grad_idx = batch * dense_adam_config.param_count + param_idx;
        grad_sum = grad_sum + dense_gradients[grad_idx];
    }
    let avg_grad = grad_sum / f32(batch_size);

    let param = dense_parameters[param_idx];

    // Apply weight decay
    let effective_grad = avg_grad + dense_adam_config.weight_decay * param;

    // Update biased first moment: m = β₁ * m + (1 - β₁) * g
    let m_old = dense_exp_avg[param_idx];
    let m_new = dense_adam_config.beta1 * m_old + (1.0 - dense_adam_config.beta1) * effective_grad;
    dense_exp_avg[param_idx] = m_new;

    // Update biased second moment: v = β₂ * v + (1 - β₂) * g²
    let v_old = dense_exp_avg_sq[param_idx];
    let v_new = dense_adam_config.beta2 * v_old + (1.0 - dense_adam_config.beta2) * effective_grad * effective_grad;
    dense_exp_avg_sq[param_idx] = v_new;

    // Bias-corrected moments
    let step_float = f32(dense_adam_config.step + 1u);
    let bias_correction1 = 1.0 / (1.0 - pow(dense_adam_config.beta1, step_float));
    let bias_correction2 = 1.0 / (1.0 - pow(dense_adam_config.beta2, step_float));

    let m_hat = m_new * bias_correction1;
    let v_hat = v_new * bias_correction2;

    // Parameter update: θ = θ - α * m̂ / (√v̂ + ε)
    let denom = sqrt(v_hat) + dense_adam_config.eps;
    let update = dense_adam_config.lr * m_hat / denom;

    dense_parameters[param_idx] = param - update;
}

// =============================================================================
// DENSE SGD KERNEL
// =============================================================================

@group(0) @binding(0) var<storage, read> dense_sgd_gradients: array<f32>; // [batch_size, param_count]
@group(0) @binding(1) var<storage, read_write> dense_sgd_parameters: array<f32>; // [param_count]
@group(0) @binding(2) var<storage, read_write> dense_momentum_buffer: array<f32>; // [param_count]
@group(0) @binding(3) var<uniform> dense_sgd_config: SGDUniforms;

@compute @workgroup_size(256)
fn dense_sgd_batch_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let param_idx = global_id.x;
    if (param_idx >= dense_sgd_config.param_count) {
        return;
    }

    let batch_size = arrayLength(&dense_sgd_gradients) / dense_sgd_config.param_count;
    var grad_sum = 0.0;

    // Compute gradient average over batch
    for (var batch = 0u; batch < batch_size; batch = batch + 1u) {
        let grad_idx = batch * dense_sgd_config.param_count + param_idx;
        grad_sum = grad_sum + dense_sgd_gradients[grad_idx];
    }
    let avg_grad = grad_sum / f32(batch_size);

    let param = dense_sgd_parameters[param_idx];
    let velocity = dense_momentum_buffer[param_idx];

    // Apply weight decay
    let effective_grad = avg_grad + dense_sgd_config.weight_decay * param;

    // Momentum update
    let new_velocity = dense_sgd_config.momentum * velocity + (1.0 - dense_sgd_config.dampening) * effective_grad;
    dense_momentum_buffer[param_idx] = new_velocity;

    var update_grad = new_velocity;

    // Nesterov momentum
    if (dense_sgd_config.nesterov == 1u) {
        update_grad = dense_sgd_config.momentum * new_velocity + effective_grad;
    }

    // Parameter update
    let update = dense_sgd_config.lr * update_grad;
    dense_sgd_parameters[param_idx] = param - update;
}

// =============================================================================
// SPARSE RMSprop KERNEL
// =============================================================================

@group(0) @binding(0) var<storage, read> rmsprop_sparse_indices: array<u32>;
@group(0) @binding(1) var<storage, read> rmsprop_gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> rmsprop_parameters: array<f32>;
@group(0) @binding(3) var<storage, read_write> square_avg: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_avg: array<f32>;
@group(0) @binding(5) var<storage, read_write> momentum_buffer: array<f32>;
@group(0) @binding(6) var<uniform> rmsprop_config: RMSpropUniforms;

@compute @workgroup_size(256)
fn sparse_rmsprop_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&rmsprop_sparse_indices)) {
        return;
    }

    let pos = rmsprop_sparse_indices[idx];
    if (pos >= arrayLength(&rmsprop_parameters)) {
        return;
    }

    let grad = rmsprop_gradients[idx];
    let param = rmsprop_parameters[pos];

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

    // Parameter update: param = param - lr * grad / denom
    let update = rmsprop_config.lr * effective_grad / denom;
    rmsprop_parameters[pos] = param - update;
}

// =============================================================================
// DENSE RMSprop KERNEL
// =============================================================================

@group(0) @binding(0) var<storage, read> dense_rmsprop_gradients: array<f32>; // [batch_size, param_count]
@group(0) @binding(1) var<storage, read_write> dense_rmsprop_parameters: array<f32>; // [param_count]
@group(0) @binding(2) var<storage, read_write> dense_square_avg: array<f32>; // [param_count]
@group(0) @binding(3) var<storage, read_write> dense_grad_avg: array<f32>; // [param_count]
@group(0) @binding(4) var<storage, read_write> dense_momentum_buffer: array<f32>; // [param_count]
@group(0) @binding(5) var<uniform> dense_rmsprop_config: RMSpropUniforms;

@compute @workgroup_size(256)
fn dense_rmsprop_batch_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let param_idx = global_id.x;
    if (param_idx >= dense_rmsprop_config.param_count) {
        return;
    }

    let batch_size = arrayLength(&dense_rmsprop_gradients) / dense_rmsprop_config.param_count;
    var grad_sum = 0.0;

    // Compute gradient average over batch
    for (var batch = 0u; batch < batch_size; batch = batch + 1u) {
        let grad_idx = batch * dense_rmsprop_config.param_count + param_idx;
        grad_sum = grad_sum + dense_rmsprop_gradients[grad_idx];
    }
    let avg_grad = grad_sum / f32(batch_size);

    let param = dense_rmsprop_parameters[param_idx];

    // Apply weight decay
    let effective_grad = avg_grad + dense_rmsprop_config.weight_decay * param;

    // Update square average: square_avg = alpha * square_avg + (1 - alpha) * grad^2
    let old_square_avg = dense_square_avg[param_idx];
    let new_square_avg = dense_rmsprop_config.alpha * old_square_avg + (1.0 - dense_rmsprop_config.alpha) * effective_grad * effective_grad;
    dense_square_avg[param_idx] = new_square_avg;

    // Compute denominator
    var denom = sqrt(new_square_avg) + dense_rmsprop_config.eps;

    // For centered RMSprop: denom = sqrt(square_avg - grad_avg^2 + eps)
    if (dense_rmsprop_config.centered == 1u) {
        // Update grad_avg: grad_avg = alpha * grad_avg + (1 - alpha) * grad
        let old_grad_avg = dense_grad_avg[param_idx];
        let new_grad_avg = dense_rmsprop_config.alpha * old_grad_avg + (1.0 - dense_rmsprop_config.alpha) * effective_grad;
        dense_grad_avg[param_idx] = new_grad_avg;

        // Centered denominator: sqrt(square_avg - grad_avg^2 + eps)
        let grad_avg_sq = new_grad_avg * new_grad_avg;
        let centered_value = new_square_avg - grad_avg_sq;
        denom = sqrt(max(centered_value, 0.0)) + dense_rmsprop_config.eps;
    }

    // Parameter update: param = param - lr * grad / denom
    let update = dense_rmsprop_config.lr * effective_grad / denom;
    dense_rmsprop_parameters[param_idx] = param - update;
}
