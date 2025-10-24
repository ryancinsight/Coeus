//! Fuzzing targets for neural network operations.
//!
//! This module provides fuzzing targets that can be used with cargo-fuzz
//! to find edge cases and potential crashes in neural network operations.

use arbitrary::{Arbitrary, Unstructured};

/// Arbitrary tensor data for fuzzing
#[derive(Debug, Clone)]
pub struct ArbitraryTensorData {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl<'a> Arbitrary<'a> for ArbitraryTensorData {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Generate a reasonable shape (1-4 dimensions, reasonable sizes)
        let num_dims = u.int_in_range(1..=4)?;
        let mut shape = Vec::with_capacity(num_dims);

        for _ in 0..num_dims {
            // Keep dimensions reasonable for fuzzing (1-64 elements per dimension)
            shape.push(u.int_in_range(1..=64)?);
        }

        // Generate data for the tensor
        let total_elements: usize = shape.iter().product();
        let mut data = Vec::with_capacity(total_elements);

        for _ in 0..total_elements {
            // Generate float values, including edge cases
            let val = match u.int_in_range(0..=10)? {
                0 => 0.0,                   // Zero
                1 => 1.0,                   // One
                2 => -1.0,                  // Negative one
                3 => f32::INFINITY,         // Positive infinity
                4 => f32::NEG_INFINITY,     // Negative infinity
                5 => f32::NAN,              // NaN
                6 => f32::MIN,              // Minimum finite value
                7 => f32::MAX,              // Maximum finite value
                8 => f32::MIN_POSITIVE,     // Minimum positive value
                _ => u.arbitrary::<f32>()?, // Random finite float
            };
            data.push(val);
        }

        Ok(ArbitraryTensorData { shape, data })
    }
}

/// Fuzz target for ReLU activation
pub fn fuzz_relu(data: &[u8]) {
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_nn::functional_activations::relu;
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    let arb_data = match ArbitraryTensorData::arbitrary(&mut Unstructured::new(data)) {
        Ok(d) => d,
        Err(_) => return, // Skip invalid inputs
    };

    // Convert f32 data to Float32
    let float_data: Vec<Float32> = arb_data.data.iter().map(|&x| Float32::new(x)).collect();

    // Create tensor (skip if shape is too large)
    let total_elements: usize = arb_data.shape.iter().product();
    if total_elements > 10000 || total_elements == 0 {
        return;
    }

    let tensor = match Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        float_data,
        &arb_data.shape,
    ) {
        Ok(t) => t,
        Err(_) => return, // Skip invalid tensor creation
    };

    // Apply ReLU - this should not panic or crash
    let _result = relu(&tensor);
}

/// Fuzz target for convolution operations
pub fn fuzz_conv2d(data: &[u8]) {
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_nn::functional_conv::conv2d;
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    let mut u = Unstructured::new(data);

    // Generate input tensor
    let input_data = match ArbitraryTensorData::arbitrary(&mut u) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Only test 4D tensors for conv2d
    if input_data.shape.len() != 4 || input_data.shape.iter().any(|&x| x == 0) {
        return;
    }

    let batch_size = input_data.shape[0];
    let in_channels = input_data.shape[1];

    // Generate weight tensor with compatible dimensions
    let out_channels = u.int_in_range(1..=32).unwrap_or(1);
    let kernel_h = u.int_in_range(1..=7).unwrap_or(3);
    let kernel_w = u.int_in_range(1..=7).unwrap_or(3);

    let weight_shape = vec![out_channels, in_channels, kernel_h, kernel_w];
    let weight_elements: usize = weight_shape.iter().product();

    if weight_elements > 10000 {
        return; // Skip too large weights
    }

    let mut weight_data = Vec::with_capacity(weight_elements);
    for _ in 0..weight_elements {
        weight_data.push(Float32::new(u.arbitrary::<f32>().unwrap_or(0.0)));
    }

    // Convert data to tensors
    let input_float_data: Vec<Float32> = input_data.data.iter().map(|&x| Float32::new(x)).collect();

    let input = match Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_float_data,
        &input_data.shape,
    ) {
        Ok(t) => t,
        Err(_) => return,
    };

    let weight = match Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        weight_data,
        &weight_shape,
    ) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Generate random stride and padding
    let stride_h = u.int_in_range(1..=3).unwrap_or(1);
    let stride_w = u.int_in_range(1..=3).unwrap_or(1);
    let padding_h = u.int_in_range(0..=2).unwrap_or(0);
    let padding_w = u.int_in_range(0..=2).unwrap_or(0);

    // Apply convolution - should not panic
    let _result = conv2d(
        &input,
        &weight,
        None,
        Some((stride_h, stride_w)),
        Some((padding_h, padding_w)),
    );
}

/// Fuzz target for max pooling
pub fn fuzz_max_pool2d(data: &[u8]) {
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_nn::functional_pooling::max_pool2d;
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    let arb_data = match ArbitraryTensorData::arbitrary(&mut Unstructured::new(data)) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Only test 4D tensors for pooling
    if arb_data.shape.len() != 4 || arb_data.shape.iter().any(|&x| x == 0) {
        return;
    }

    // Check if tensor is too large
    let total_elements: usize = arb_data.shape.iter().product();
    if total_elements > 50000 {
        return;
    }

    let float_data: Vec<Float32> = arb_data.data.iter().map(|&x| Float32::new(x)).collect();
    let tensor = match Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        float_data,
        &arb_data.shape,
    ) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Generate pooling parameters
    let mut u = Unstructured::new(data);
    let kernel_h = u.int_in_range(2..=4).unwrap_or(2);
    let kernel_w = u.int_in_range(2..=4).unwrap_or(2);
    let stride_h = u.int_in_range(1..=2).unwrap_or(2);
    let stride_w = u.int_in_range(1..=2).unwrap_or(2);
    let padding_h = u.int_in_range(0..=1).unwrap_or(0);
    let padding_w = u.int_in_range(0..=1).unwrap_or(0);

    // Apply max pooling - should not panic
    let _result = max_pool2d(
        &tensor,
        (kernel_h, kernel_w),
        Some((stride_h, stride_w)),
        (padding_h, padding_w),
    );
}

/// Fuzz target for linear transformations
pub fn fuzz_linear(data: &[u8]) {
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_nn::functional_linear::linear;
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    let mut u = Unstructured::new(data);

    // Generate input tensor
    let input_data = match ArbitraryTensorData::arbitrary(&mut u) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Flatten input for linear layer
    let in_features = input_data.shape.iter().product::<usize>();
    if in_features == 0 || in_features > 1000 {
        return;
    }

    let input_flat = input_data.data;

    // Generate weight matrix
    let out_features = u.int_in_range(1..=100).unwrap_or(10);
    let weight_elements = out_features * in_features;

    if weight_elements > 10000 {
        return; // Skip too large matrices
    }

    let mut weight_data = Vec::with_capacity(weight_elements);
    for _ in 0..weight_elements {
        weight_data.push(Float32::new(u.arbitrary::<f32>().unwrap_or(0.0)));
    }

    // Convert to tensors
    let input_float_data: Vec<Float32> = input_flat.iter().map(|&x| Float32::new(x)).collect();
    let input = match Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_float_data,
        &[1, in_features], // Add batch dimension
    ) {
        Ok(t) => t,
        Err(_) => return,
    };

    let weight_float_data: Vec<Float32> = weight_data;
    let weight = match Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        weight_float_data,
        &[out_features, in_features],
    ) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Apply linear transformation - should not panic
    let _result = linear(&input, &weight, None);
}

/// Fuzz target for loss functions
pub fn fuzz_mse_loss(data: &[u8]) {
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_nn::functional_loss::mse_loss;
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    let arb_data = match ArbitraryTensorData::arbitrary(&mut Unstructured::new(data)) {
        Ok(d) => d,
        Err(_) => return,
    };

    let total_elements: usize = arb_data.shape.iter().product();
    if total_elements == 0 || total_elements > 10000 {
        return;
    }

    let float_data: Vec<Float32> = arb_data.data.iter().map(|&x| Float32::new(x)).collect();
    let pred = match Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        float_data.clone(),
        &arb_data.shape,
    ) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Create target with same shape but different values
    let target_data: Vec<Float32> = float_data.iter().map(|&x| x + Float32::new(1.0)).collect();
    let target = match Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        target_data,
        &arb_data.shape,
    ) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Compute MSE loss - should not panic
    let _result = mse_loss(&pred, &target);
}

/// Fuzz target for softmax
pub fn fuzz_softmax(data: &[u8]) {
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_nn::functional_attention::softmax;
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    let arb_data = match ArbitraryTensorData::arbitrary(&mut Unstructured::new(data)) {
        Ok(d) => d,
        Err(_) => return,
    };

    let total_elements: usize = arb_data.shape.iter().product();
    if total_elements == 0 || total_elements > 1000 {
        return;
    }

    let float_data: Vec<Float32> = arb_data.data.iter().map(|&x| Float32::new(x)).collect();

    // Create tensor with last dimension for softmax
    let mut shape = arb_data.shape;
    if shape.is_empty() {
        shape = vec![total_elements];
    }

    let tensor = match Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        float_data, &shape,
    ) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Apply softmax - should not panic
    let _result = softmax(&tensor);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Unit tests for fuzz targets (to ensure they don't panic with known inputs)
    #[test]
    fn test_fuzz_targets_basic() {
        // Test relu with basic input
        let data = vec![0, 1, 2, 3]; // Simple arbitrary data
        fuzz_relu(&data);

        // Test conv2d with basic input
        let data = vec![0; 100]; // More data for conv2d
        fuzz_conv2d(&data);

        // Test max_pool2d
        fuzz_max_pool2d(&data);

        // Test linear
        fuzz_linear(&data);

        // Test mse_loss
        fuzz_mse_loss(&data);

        // Test softmax
        fuzz_softmax(&data);
    }
}
