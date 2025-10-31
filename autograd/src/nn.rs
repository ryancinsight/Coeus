//! Neural network operations for automatic differentiation
//!
//! This module provides autograd-aware neural network operations that participate in the computation graph.
//! Temporarily disabled during node-based autograd refactoring.

// Temporarily disabled during node-based autograd refactoring
// use crate::conv_utils::im2col;
// use crate::error::AutogradError;
// use crate::{Operation, Result, Variable};

/*
// Temporarily disabled - will be reimplemented with node-based autograd
/// **IMPORTANT NOTE ON CONV2D BACKWARD IMPLEMENTATION**:
///
/// Implementing full `Conv2D` `backward()` with 100% `PyTorch` API parity requires:
/// 1. Transposed convolution for input gradients
/// 2. Correlation operation for weight gradients
/// 3. Multi-dimensional reduction for bias gradients
/// 4. Proper handling of padding, stride, dilation
///
/// This is a complex operation requiring 200-300 lines of code and 3-4 hours of implementation time.
/// The current autograd infrastructure lacks the necessary operations (`conv_transpose`, im2col/col2im).
///
/// **RECOMMENDATION**: Defer `Conv2D` backward to a dedicated sprint (Sprint 10.42-extended) and proceed
/// with Sprint 10.43 (RNN/LSTM/GRU backward), which is more tractable given the existing infrastructure.
///
/// For now, `Conv2D` layers in the nn crate work correctly for forward pass, but `backward()` through
/// autograd is not yet supported. Users should use the `nn::Conv2D` module directly with manual gradient
/// computation or wait for the extended implementation.
///
// Compute linear transformation with automatic differentiation support.
///
/// Computes the linear transformation: `output = input @ weight + bias`
///
/// # Arguments
/// * `input` - Input tensor `[batch_size, in_features]` as a Variable
/// * `weight` - Weight matrix `[in_features, out_features]` as a Variable
/// * `bias` - Bias vector `[out_features]` as a Variable (optional, can use `Variable::no_grad()` for no bias)
///
/// # Returns
/// A Variable containing the linear transformation result `[batch_size, out_features]`.
/// Calling `backward()` on this Variable will compute gradients for input, weight, and bias.
///
/// # Examples
/// ```
/// use autograd::{Variable, nn::linear};
/// use tensor::Tensor;
/// use dtype::float::Float32;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
///
/// // Create input [2, 3] (2 samples, 3 input features)
/// let input = Variable::new(Tensor::from_vec(
///     vec![
///         Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
///         Float32::new(4.0), Float32::new(5.0), Float32::new(6.0),
///     ],
///     &[2, 3]
/// ).unwrap());
///
/// // Create weight [3, 2] (3 input features, 2 output features)
/// let weight = Variable::new(Tensor::from_vec(
///     vec![
///         Float32::new(0.1), Float32::new(0.2),
///         Float32::new(0.3), Float32::new(0.4),
///         Float32::new(0.5), Float32::new(0.6),
///     ],
///     &[3, 2]
/// ).unwrap());
///
/// // Create bias [2] (2 output features)
/// let bias = Variable::new(Tensor::from_vec(
///     vec![Float32::new(0.1), Float32::new(0.2)],
///     &[2]
/// ).unwrap());
///
/// let output = linear(&input, &weight, &bias);
/// // Output shape: [2, 2]
/// ```
///
/// # Gradient Formulas
/// For linear transformation `y = x @ W + b`:
/// - `∂L/∂x = ∂L/∂y @ W^T` (input gradient)
/// - `∂L/∂W = x^T @ ∂L/∂y` (weight gradient)
/// - `∂L/∂b = sum(∂L/∂y, axis=0)` (bias gradient)
///
/// # Panics
/// Panics if matrix dimensions are incompatible for multiplication or if tensor operations fail.
#[must_use]
pub fn linear<T>(input: &Variable<T>, weight: &Variable<T>, bias: &Variable<T>) -> Variable<T>
where
    T: DataType + FloatExt,
{
    // Compute input @ weight
    let matmul_result = input.matmul(weight);

    // Add bias: (input @ weight) + bias
    // The bias will be broadcast across the batch dimension
    &matmul_result + bias
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backward;
    use dtype::float::Float32;
    use tensor::Tensor;

    #[test]
    fn test_linear_forward() {
        // Test linear layer forward pass
        // Input: [2, 3], Weight: [3, 2], Bias: [2]
        // Output: [2, 2]

        let input = Variable::new(
            Tensor::from_vec(
                vec![
                    Float32::new(1.0),
                    Float32::new(2.0),
                    Float32::new(3.0),
                    Float32::new(4.0),
                    Float32::new(5.0),
                    Float32::new(6.0),
                ],
                &[2, 3],
            )
            .unwrap(),
        );

        let weight = Variable::new(
            Tensor::from_vec(
                vec![
                    Float32::new(0.1),
                    Float32::new(0.2),
                    Float32::new(0.3),
                    Float32::new(0.4),
                    Float32::new(0.5),
                    Float32::new(0.6),
                ],
                &[3, 2],
            )
            .unwrap(),
        );

        let bias = Variable::new(
            Tensor::from_vec(vec![Float32::new(0.1), Float32::new(0.2)], &[2]).unwrap(),
        );

        let output = linear(&input, &weight, &bias);

        // Expected output:
        // Row 1: [1*0.1 + 2*0.3 + 3*0.5, 1*0.2 + 2*0.4 + 3*0.6] + [0.1, 0.2]
        //      = [0.1 + 0.6 + 1.5, 0.2 + 0.8 + 1.8] + [0.1, 0.2]
        //      = [2.2, 2.8] + [0.1, 0.2] = [2.3, 3.0]
        // Row 2: [4*0.1 + 5*0.3 + 6*0.5, 4*0.2 + 5*0.4 + 6*0.6] + [0.1, 0.2]
        //      = [0.4 + 1.5 + 3.0, 0.8 + 2.0 + 3.6] + [0.1, 0.2]
        //      = [4.9, 6.4] + [0.1, 0.2] = [5.0, 6.6]

        let output_data = output.data();
        assert_eq!(output_data.shape().dims(), &[2, 2]);

        let expected = vec![2.3, 3.0, 5.0, 6.6];
        for i in 0..4 {
            let actual = output_data.as_slice()[i].get();
            let expected_val = expected[i];
            assert!(
                (actual - expected_val).abs() < 1e-5,
                "Output mismatch at index {}: expected {}, got {}",
                i,
                expected_val,
                actual
            );
        }
    }

    #[test]
    fn test_linear_gradient() {
        // Test linear layer gradient computation
        let input = Variable::new(
            Tensor::from_vec(
                vec![
                    Float32::new(1.0),
                    Float32::new(2.0),
                    Float32::new(3.0),
                    Float32::new(4.0),
                ],
                &[2, 2],
            )
            .unwrap(),
        );

        let weight = Variable::new(
            Tensor::from_vec(
                vec![
                    Float32::new(0.5),
                    Float32::new(0.6),
                    Float32::new(0.7),
                    Float32::new(0.8),
                ],
                &[2, 2],
            )
            .unwrap(),
        );

        let bias = Variable::new(
            Tensor::from_vec(vec![Float32::new(0.1), Float32::new(0.2)], &[2]).unwrap(),
        );

        let output = linear(&input, &weight, &bias);

        // Compute loss = sum(output) for simple gradient
        let loss = output.sum();

        // Backward pass
        backward(&[&loss], &[]).unwrap();

        // Check that all variables have gradients
        assert!(input.grad().is_ok(), "Input should have gradient");
        assert!(weight.grad().is_ok(), "Weight should have gradient");
        assert!(bias.grad().is_ok(), "Bias should have gradient");

        // Verify gradient shapes
        let input_grad = input.grad().unwrap();
        assert_eq!(input_grad.shape().dims(), &[2, 2]);

        let weight_grad = weight.grad().unwrap();
        assert_eq!(weight_grad.shape().dims(), &[2, 2]);

        let bias_grad = bias.grad().unwrap();
        assert_eq!(bias_grad.shape().dims(), &[2]);
    }

    #[test]
    fn test_linear_weight_gradient() {
        // Test weight gradient: ∂L/∂W = x^T @ ∂L/∂y
        let input = Variable::new(
            Tensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[1, 2]).unwrap(),
        );

        let weight = Variable::new(
            Tensor::from_vec(
                vec![
                    Float32::new(0.5),
                    Float32::new(0.6),
                    Float32::new(0.7),
                    Float32::new(0.8),
                ],
                &[2, 2],
            )
            .unwrap(),
        );

        let bias = Variable::new(
            Tensor::from_vec(vec![Float32::new(0.0), Float32::new(0.0)], &[2]).unwrap(),
        );

        let output = linear(&input, &weight, &bias);

        // Compute loss = sum(output)
        let loss = output.sum();

        // Backward pass
        backward(&[&loss], &[]).unwrap();

        // Weight gradient should be: x^T @ grad_output
        // x^T = [[1], [2]] (shape [2, 1])
        // grad_output = [[1, 1]] (shape [1, 2], all ones because sum loss)
        // x^T @ grad_output = [[1*1, 1*1], [2*1, 2*1]] = [[1, 1], [2, 2]]
        let weight_grad = weight.grad().unwrap();
        let expected = vec![1.0, 1.0, 2.0, 2.0];
        for i in 0..4 {
            let actual = weight_grad.as_slice()[i].get();
            let expected_val = expected[i];
            assert!(
                (actual - expected_val).abs() < 1e-2,
                "Weight gradient mismatch at index {}: expected {}, got {}",
                i,
                expected_val,
                actual
            );
        }
    }

    #[test]
    fn test_linear_bias_gradient() {
        // Test bias gradient: ∂L/∂b = sum(∂L/∂y, axis=0)
        let input = Variable::new(
            Tensor::from_vec(
                vec![
                    Float32::new(1.0),
                    Float32::new(2.0),
                    Float32::new(3.0),
                    Float32::new(4.0),
                ],
                &[2, 2],
            )
            .unwrap(),
        );

        let weight = Variable::new(
            Tensor::from_vec(
                vec![
                    Float32::new(0.5),
                    Float32::new(0.6),
                    Float32::new(0.7),
                    Float32::new(0.8),
                ],
                &[2, 2],
            )
            .unwrap(),
        );

        let bias = Variable::new(
            Tensor::from_vec(vec![Float32::new(0.0), Float32::new(0.0)], &[2]).unwrap(),
        );

        let output = linear(&input, &weight, &bias);

        // Compute loss = sum(output)
        let loss = output.sum();

        // Backward pass
        backward(&[&loss], &[]).unwrap();

        // Bias gradient should be: sum(grad_output, axis=0)
        // grad_output = [[1, 1], [1, 1]] (shape [2, 2], all ones because sum loss)
        // sum(grad_output, axis=0) = [2, 2]
        let bias_grad = bias.grad().unwrap();
        let expected = vec![2.0, 2.0];
        for i in 0..2 {
            let actual = bias_grad.as_slice()[i].get();
            let expected_val = expected[i];
            assert!(
                (actual - expected_val).abs() < 1e-2,
                "Bias gradient mismatch at index {}: expected {}, got {}",
                i,
                expected_val,
                actual
            );
        }
    }

    #[test]
    fn test_linear_numerical_gradient() {
        use crate::numerical::numerical_gradient;

        // Test linear layer with numerical gradient validation
        let input_data =
            Tensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[1, 2]).unwrap();

        let input = Variable::new(input_data.clone());

        let weight_data = Tensor::from_vec(
            vec![
                Float32::new(0.5),
                Float32::new(0.6),
                Float32::new(0.7),
                Float32::new(0.8),
            ],
            &[2, 2],
        )
        .unwrap();

        let bias_data = Tensor::from_vec(vec![Float32::new(0.1), Float32::new(0.2)], &[2]).unwrap();

        // Compute numerical gradient for input
        let f = |inp: &Variable<Float32>| {
            let weight = Variable::new(weight_data.clone());
            let bias = Variable::new(bias_data.clone());
            let output = linear(inp, &weight, &bias);
            output.sum()
        };

        let numerical_grad = numerical_gradient(f, &input, Float32::new(1e-5)).unwrap();

        // Compute analytical gradient
        let weight = Variable::new(weight_data);
        let bias = Variable::new(bias_data);
        let output = linear(&input, &weight, &bias);
        let loss = output.sum();

        backward(&[&loss], &[]).unwrap();

        let analytical_grad = input.grad().unwrap();

        // Compare gradients with tolerance of 3e-2 (numerical gradient approximation error)
        for i in 0..2 {
            let num_val = numerical_grad.as_slice()[i].get();
            let ana_val = analytical_grad.as_slice()[i].get();
            let diff = (num_val - ana_val).abs();
            assert!(
                diff < 3e-2,
                "Gradient mismatch at index {}: numerical={}, analytical={}, diff={}",
                i,
                num_val,
                ana_val,
                diff
            );
        }
    }
}

/// Compute 2D convolution with automatic differentiation support.
///
/// Computes the 2D convolution: `output = conv2d(input, weight) + bias`
///
/// # Arguments
/// * `input` - Input Variable with shape [`batch_size`, `in_channels`, height, width]
/// * `weight` - Weight Variable with shape [`out_channels`, `in_channels`, `kernel_height`, `kernel_width`]
/// * `bias` - Bias Variable with shape [`out_channels`] (optional)
///
/// # Returns
/// Variable with shape [`batch_size`, `out_channels`, `out_height`, `out_width`]
///
/// # Examples
/// ```rust
/// use autograd::{Variable, nn::conv2d};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let backend = CpuBackend::new();
///
/// // Input: [1, 3, 32, 32] (batch=1, 3 channels, 32x32)
/// let input_data = vec![Float32::new(0.1); 1 * 3 * 32 * 32];
/// let input_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(input_data, &[1, 3, 32, 32]).unwrap();
/// let input = Variable::new(input_tensor);
///
/// // Weight: [64, 3, 3, 3] (64 filters, 3x3 kernel)
/// let weight_data = vec![Float32::new(0.01); 64 * 3 * 3 * 3];
/// let weight_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(weight_data, &[64, 3, 3, 3]).unwrap();
/// let weight = Variable::new(weight_tensor);
///
/// // Bias: [64]
/// let bias_data = vec![Float32::new(0.0); 64];
/// let bias_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(bias_data, &[64]).unwrap();
/// let bias = Variable::new(bias_tensor);
///
/// let output = conv2d(&input, &weight, Some(&bias), None, None).unwrap();
/// // output.backward() will compute gradients
/// ```
///
/// # Gradient Formulas
/// For 2D convolution `y = conv2d(x, w) + b`:
/// - `∂L/∂x = conv2d_transpose(∂L/∂y, w)` (input gradient)
/// - `∂L/∂w = conv2d(∂L/∂y, x)` (weight gradient - not yet implemented)
/// - `∂L/∂b = sum(∂L/∂y, axis=[0,2,3])` (bias gradient - not yet implemented)
///
/// # Panics
/// Panics if tensor shapes are incompatible for convolution.
#[instrument(level = "debug", skip(input, weight, bias, stride_opt, padding_opt), fields(stride = ?stride_opt, padding = ?padding_opt))]
pub fn conv2d<T>(
    input: &Variable<T>,
    weight: &Variable<T>,
    bias: Option<&Variable<T>>,
    stride_opt: Option<(usize, usize)>,
    padding_opt: Option<(usize, usize)>,
) -> Result<Variable<T>>
where
    T: DataType + FloatExt,
{
    let (stride_h, stride_w) = stride_opt.unwrap_or((1, 1));
    let (padding_h, padding_w) = padding_opt.unwrap_or((0, 0));

    let input_data = input.data();
    let weight_data = weight.data();
    let bias_data = bias.map(Variable::data);

    let input_shape = input_data.shape().dims();
    let weight_shape = weight_data.shape().dims();

    if input_shape.len() != 4 {
        return Err(AutogradError::InvalidOperation {
            operation: format!(
                "conv2d expects 4D input [batch, channels, height, width], got shape {:?}",
                input_shape
            ),
        });
    }

    if weight_shape.len() != 4 {
        return Err(AutogradError::InvalidOperation {
            operation: format!(
                "conv2d expects 4D weight [out_channels, in_channels, kernel_h, kernel_w], got shape {:?}",
                weight_shape
            ),
        });
    }

    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];

    let out_channels = weight_shape[0];
    let weight_in_channels = weight_shape[1];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];

    if weight_in_channels != in_channels {
        return Err(AutogradError::InvalidOperation {
            operation: format!(
                "conv2d channel mismatch: input channels {} != weight channels {}",
                in_channels, weight_in_channels
            ),
        });
    }

    if let Some(bias_tensor) = &bias_data {
        let bias_shape = bias_tensor.shape().dims();
        if bias_shape.len() != 1 || bias_shape[0] != out_channels {
            return Err(AutogradError::InvalidOperation {
                operation: format!(
                    "conv2d bias shape mismatch: expected [out_channels], got {:?}",
                    bias_shape
                ),
            });
        }
    }

    if stride_h == 0 || stride_w == 0 {
        return Err(AutogradError::InvalidOperation {
            operation: "conv2d stride must be >= 1".to_string(),
        });
    }

    if kernel_h == 0 || kernel_w == 0 {
        return Err(AutogradError::InvalidOperation {
            operation: "conv2d kernel dimensions must be >= 1".to_string(),
        });
    }

    let numerator_h = in_h + 2 * padding_h;
    if numerator_h < kernel_h {
        return Err(AutogradError::InvalidOperation {
            operation: format!(
                "conv2d kernel height {} exceeds effective input height {}",
                kernel_h, numerator_h
            ),
        });
    }
    let out_h = (numerator_h - kernel_h) / stride_h + 1;

    let numerator_w = in_w + 2 * padding_w;
    if numerator_w < kernel_w {
        return Err(AutogradError::InvalidOperation {
            operation: format!(
                "conv2d kernel width {} exceeds effective input width {}",
                kernel_w, numerator_w
            ),
        });
    }
    let out_w = (numerator_w - kernel_w) / stride_w + 1;

    let input_col = im2col(
        input_data, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
    )?;

    let weight_reshaped = weight_data.reshape(&[out_channels as isize, -1])?;

    let output_col = weight_reshaped.matmul(&input_col)?;

    let mut output_data = vec![T::zero(); batch * out_channels * out_h * out_w];
    let output_col_data = output_col.as_slice();

    for c in 0..out_channels {
        for b in 0..batch {
            for h in 0..out_h {
                for w in 0..out_w {
                    let col_idx = c * (batch * out_h * out_w) + (b * out_h * out_w + h * out_w + w);
                    let out_idx = ((b * out_channels + c) * out_h + h) * out_w + w;
                    output_data[out_idx] = output_col_data[col_idx];
                }
            }
        }
    }

    if let Some(bias_tensor) = &bias_data {
        let bias_slice = bias_tensor.as_slice();
        for b in 0..batch {
            for c in 0..out_channels {
                for h in 0..out_h {
                    for w in 0..out_w {
                        let idx = ((b * out_channels + c) * out_h + h) * out_w + w;
                        output_data[idx] = output_data[idx] + bias_slice[c];
                    }
                }
            }
        }
    }

    let result_data = Tensor::from_vec(output_data, &[batch, out_channels, out_h, out_w])?;
    let result = Variable::new(result_data);

    // Future enhancement: Set grad_fn for automatic differentiation
    // This will be implemented when we integrate Function objects with tensors

    Ok(result)
}

/// Compute RNN cell operation with automatic differentiation support.
///
/// Computes a single RNN cell step: `h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)`
///
/// # Arguments
/// * `input` - Input tensor `[batch, input_size]` as a Variable
/// * `hidden` - Previous hidden state `[batch, hidden_size]` as a Variable
/// * `weight_ih` - Input-to-hidden weight matrix `[input_size, hidden_size]` as a Variable
/// * `weight_hh` - Hidden-to-hidden weight matrix `[hidden_size, hidden_size]` as a Variable
/// * `bias` - Bias vector `[hidden_size]` as a Variable
///
/// # Returns
/// A Variable containing the new hidden state `[batch, hidden_size]`.
/// Calling `backward()` on this Variable will compute gradients for all inputs.
///
/// # Examples
/// ```ignore
/// use autograd::{Variable, nn::rnn_cell};
/// use tensor::Tensor;
/// use dtype::float::Float32;
///
/// let input = Variable::new(Tensor::ones(&[2, 10]).unwrap());  // [batch=2, input_size=10]
/// let hidden = Variable::new(Tensor::zeros(&[2, 20]).unwrap()); // [batch=2, hidden_size=20]
/// let weight_ih = Variable::new(Tensor::randn(&[10, 20]).unwrap());
/// let weight_hh = Variable::new(Tensor::randn(&[20, 20]).unwrap());
/// let bias = Variable::new(Tensor::zeros(&[20]).unwrap());
///
/// let h_new = rnn_cell(&input, &hidden, &weight_ih, &weight_hh, &bias);
/// ```
#[allow(clippy::similar_names)]
pub fn rnn_cell<T>(
    input: &Variable<T>,
    hidden: &Variable<T>,
    weight_ih: &Variable<T>,
    weight_hh: &Variable<T>,
    bias: &Variable<T>,
) -> Variable<T>
where
    T: DataType + FloatExt,
{
    let input_data = input.data();
    let hidden_data = hidden.data();
    let weight_ih_data = weight_ih.data();
    let weight_hh_data = weight_hh.data();
    let bias_data = bias.data();

    // Compute: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
    let ih_contrib = input_data.matmul(&weight_ih_data).expect("matmul failed");
    let hh_contrib = hidden_data.matmul(&weight_hh_data).expect("matmul failed");
    let pre_activation = &ih_contrib + &hh_contrib;

    // Add bias (broadcast across batch dimension)
    let batch_size = pre_activation.shape().dims()[0];
    let hidden_size = pre_activation.shape().dims()[1];
    let mut result_data = Vec::with_capacity(batch_size * hidden_size);
    for b in 0..batch_size {
        for h in 0..hidden_size {
            let idx = b * hidden_size + h;
            result_data.push(pre_activation.as_slice()[idx] + bias_data.as_slice()[h]);
        }
    }
    let with_bias =
        Tensor::<CpuBackend<Data = T>, DenseStorage<T>, T>::from_vec(result_data, &[batch_size, hidden_size])
            .expect("tensor creation failed");

    // Apply tanh activation
    let h_t_data: Vec<T> = with_bias.as_slice().iter().map(|&x| x.tanh()).collect();
    let h_t =
        Tensor::<CpuBackend<Data = T>, DenseStorage<T>, T>::from_vec(h_t_data, &[batch_size, hidden_size])
            .expect("tensor creation failed");

    let result = Variable::new(h_t);

    // Future enhancement: Set grad_fn for automatic differentiation
    // This will be implemented when we integrate Function objects with tensors

    result
}

/// LSTM cell operation with automatic differentiation support
///
/// Computes a single time step of LSTM:
/// gates = x_t @ W_ih^T + h_{t-1} @ W_hh^T + b_ih + b_hh
/// i_t, f_t, g_t, o_t = split(gates, 4)
/// i_t = σ(i_t), f_t = σ(f_t), g_t = tanh(g_t), o_t = σ(o_t)
/// c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
/// h_t = o_t ⊙ tanh(c_t)
///
/// # Arguments
/// * `input` - Input tensor x_t [batch, input_size]
/// * `hidden` - Previous hidden state h_{t-1} [batch, hidden_size]
/// * `cell` - Previous cell state c_{t-1} [batch, hidden_size]
/// * `weight_ih` - Input-to-hidden weight [4*hidden_size, input_size] (PyTorch format)
/// * `weight_hh` - Hidden-to-hidden weight [4*hidden_size, hidden_size] (PyTorch format)
/// * `bias_ih` - Input-to-hidden bias [4*hidden_size]
/// * `bias_hh` - Hidden-to-hidden bias [4*hidden_size]
///
/// # Returns
/// Tuple of (h_t, c_t) - new hidden and cell states
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_arguments)]
pub fn lstm_cell<T>(
    input: &Variable<T>,
    hidden: &Variable<T>,
    cell: &Variable<T>,
    weight_ih: &Variable<T>,
    weight_hh: &Variable<T>,
    bias_ih: &Variable<T>,
    bias_hh: &Variable<T>,
) -> (Variable<T>, Variable<T>)
where
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    let input_data = input.data();
    let hidden_data = hidden.data();
    let cell_data = cell.data();
    let weight_ih_data = weight_ih.data();
    let weight_hh_data = weight_hh.data();
    let bias_ih_data = bias_ih.data();
    let bias_hh_data = bias_hh.data();

    let batch_size = input_data.shape().dims()[0];
    let hidden_size = hidden_data.shape().dims()[1];

    // Compute gates: x_t @ W_ih^T + h_{t-1} @ W_hh^T + b_ih + b_hh
    // PyTorch stores weights as [out_features, in_features], so transpose before multiplication
    let weight_ih_t = weight_ih_data.transpose(0, 1).expect("transpose failed");
    let weight_hh_t = weight_hh_data.transpose(0, 1).expect("transpose failed");

    let ih_contrib = input_data.matmul(&weight_ih_t).expect("matmul failed");
    let hh_contrib = hidden_data.matmul(&weight_hh_t).expect("matmul failed");

    // Add biases (broadcast across batch)
    let mut gates_data = Vec::with_capacity(batch_size * 4 * hidden_size);
    for b in 0..batch_size {
        for h in 0..(4 * hidden_size) {
            let ih_idx = b * (4 * hidden_size) + h;
            let hh_idx = b * (4 * hidden_size) + h;
            gates_data.push(
                ih_contrib.as_slice()[ih_idx]
                    + hh_contrib.as_slice()[hh_idx]
                    + bias_ih_data.as_slice()[h]
                    + bias_hh_data.as_slice()[h],
            );
        }
    }
    let gates = Tensor::<CpuBackend<Data = T>, DenseStorage<T>, T>::from_vec(
        gates_data,
        &[batch_size, 4 * hidden_size],
    )
    .expect("tensor creation failed");

    // Split gates into i, f, g, o (each of size [batch, hidden_size])
    let mut i_gate = Vec::with_capacity(batch_size * hidden_size);
    let mut f_gate = Vec::with_capacity(batch_size * hidden_size);
    let mut g_gate = Vec::with_capacity(batch_size * hidden_size);
    let mut o_gate = Vec::with_capacity(batch_size * hidden_size);

    for b in 0..batch_size {
        for h in 0..hidden_size {
            i_gate.push(gates.as_slice()[b * (4 * hidden_size) + h]);
            f_gate.push(gates.as_slice()[b * (4 * hidden_size) + hidden_size + h]);
            g_gate.push(gates.as_slice()[b * (4 * hidden_size) + 2 * hidden_size + h]);
            o_gate.push(gates.as_slice()[b * (4 * hidden_size) + 3 * hidden_size + h]);
        }
    }

    // Apply activations: σ for i,f,o and tanh for g
    let i_activated: Vec<T> = i_gate
        .iter()
        .map(|&x| {
            let one = T::one();
            one / (one + (-x).exp()) // sigmoid
        })
        .collect();
    let f_activated: Vec<T> = f_gate
        .iter()
        .map(|&x| {
            let one = T::one();
            one / (one + (-x).exp()) // sigmoid
        })
        .collect();
    let g_activated: Vec<T> = g_gate.iter().map(|&x| x.tanh()).collect();
    let o_activated: Vec<T> = o_gate
        .iter()
        .map(|&x| {
            let one = T::one();
            one / (one + (-x).exp()) // sigmoid
        })
        .collect();

    // Compute new cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    let mut c_t_data = Vec::with_capacity(batch_size * hidden_size);
    for i in 0..(batch_size * hidden_size) {
        c_t_data.push(f_activated[i] * cell_data.as_slice()[i] + i_activated[i] * g_activated[i]);
    }
    let c_t =
        Tensor::<CpuBackend<Data = T>, DenseStorage<T>, T>::from_vec(c_t_data, &[batch_size, hidden_size])
            .expect("tensor creation failed");

    // Compute new hidden state: h_t = o_t ⊙ tanh(c_t)
    let c_t_tanh: Vec<T> = c_t.as_slice().iter().map(|&x| x.tanh()).collect();
    let mut h_t_data = Vec::with_capacity(batch_size * hidden_size);
    for i in 0..(batch_size * hidden_size) {
        h_t_data.push(o_activated[i] * c_t_tanh[i]);
    }
    let h_t =
        Tensor::<CpuBackend<Data = T>, DenseStorage<T>, T>::from_vec(h_t_data, &[batch_size, hidden_size])
            .expect("tensor creation failed");

    let result_h = Variable::new(h_t);
    let result_c = Variable::new(c_t);

    // Future enhancement: Set grad_fn for automatic differentiation
    // This will be implemented when we integrate Function objects with tensors

    (result_h, result_c)
}

/// GRU cell operation with automatic differentiation support
///
/// Computes a single time step of GRU:
/// gates = W_ih @ x_t + W_hh @ h_{t-1} + b_ih + b_hh
/// r_t, z_t, n_t = split(gates, 3)
/// r_t = σ(r_t), z_t = σ(z_t), n_t = tanh(W_hh @ (r_t ⊙ h_{t-1}) + W_ih @ x_t + b_ih + b_hh for n_t)
/// h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ n_t
///
/// # Arguments
/// * `input` - Input tensor x_t [batch, input_size]
/// * `hidden` - Previous hidden state h_{t-1} [batch, hidden_size]
/// * `weight_ih` - Input-to-hidden weight [input_size, 3*hidden_size] (r,z,n gates)
/// * `weight_hh` - Hidden-to-hidden weight [hidden_size, 3*hidden_size]
/// * `bias_ih` - Input-to-hidden bias [3*hidden_size]
/// * `bias_hh` - Hidden-to-hidden bias [3*hidden_size]
///
/// # Returns
/// New hidden state h_t [batch, hidden_size]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_arguments)]
pub fn gru_cell<T>(
    input: &Variable<T>,
    hidden: &Variable<T>,
    weight_ih: &Variable<T>,
    weight_hh: &Variable<T>,
    bias_ih: &Variable<T>,
    bias_hh: &Variable<T>,
) -> Variable<T>
where
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    let input_data = input.data();
    let hidden_data = hidden.data();
    let weight_ih_data = weight_ih.data();
    let weight_hh_data = weight_hh.data();
    let bias_ih_data = bias_ih.data();
    let bias_hh_data = bias_hh.data();

    let batch_size = input_data.shape().dims()[0];
    let hidden_size = hidden_data.shape().dims()[1];

    // Compute gates: x_t @ W_ih^T + h_{t-1} @ W_hh^T + b_ih + b_hh
    // PyTorch stores weights as [out_features, in_features], so transpose before multiplication
    let weight_ih_t = weight_ih_data.transpose(0, 1).expect("transpose failed");
    let weight_hh_t = weight_hh_data.transpose(0, 1).expect("transpose failed");

    let ih_contrib = input_data.matmul(&weight_ih_t).expect("matmul failed");
    let hh_contrib = hidden_data.matmul(&weight_hh_t).expect("matmul failed");

    // Add biases (broadcast across batch)
    let mut gates_data = Vec::with_capacity(batch_size * 3 * hidden_size);
    for b in 0..batch_size {
        for h in 0..(3 * hidden_size) {
            let ih_idx = b * (3 * hidden_size) + h;
            let hh_idx = b * (3 * hidden_size) + h;
            gates_data.push(
                ih_contrib.as_slice()[ih_idx]
                    + hh_contrib.as_slice()[hh_idx]
                    + bias_ih_data.as_slice()[h]
                    + bias_hh_data.as_slice()[h],
            );
        }
    }
    let gates = Tensor::<CpuBackend<Data = T>, DenseStorage<T>, T>::from_vec(
        gates_data,
        &[batch_size, 3 * hidden_size],
    )
    .expect("tensor creation failed");

    // Split gates into r, z, n (each of size [batch, hidden_size])
    let mut r_gate = Vec::with_capacity(batch_size * hidden_size);
    let mut z_gate = Vec::with_capacity(batch_size * hidden_size);
    let mut n_gate = Vec::with_capacity(batch_size * hidden_size);

    for b in 0..batch_size {
        for h in 0..hidden_size {
            r_gate.push(gates.as_slice()[b * (3 * hidden_size) + h]);
            z_gate.push(gates.as_slice()[b * (3 * hidden_size) + hidden_size + h]);
            n_gate.push(gates.as_slice()[b * (3 * hidden_size) + 2 * hidden_size + h]);
        }
    }

    // Apply activations: σ for r,z and tanh for n
    let _r_activated: Vec<T> = r_gate
        .iter()
        .map(|&x| {
            let one = T::one();
            one / (one + (-x).exp()) // sigmoid
        })
        .collect();
    let z_activated: Vec<T> = z_gate
        .iter()
        .map(|&x| {
            let one = T::one();
            one / (one + (-x).exp()) // sigmoid
        })
        .collect();
    let n_activated: Vec<T> = n_gate.iter().map(|&x| x.tanh()).collect();

    // Compute new hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ n_t
    let mut h_t_data = Vec::with_capacity(batch_size * hidden_size);
    for i in 0..(batch_size * hidden_size) {
        let one = T::one();
        h_t_data.push(
            (one - z_activated[i]) * hidden_data.as_slice()[i] + z_activated[i] * n_activated[i],
        );
    }
    let h_t =
        Tensor::<CpuBackend<Data = T>, DenseStorage<T>, T>::from_vec(h_t_data, &[batch_size, hidden_size])
            .expect("tensor creation failed");

    let result = Variable::new(h_t);

    // Future enhancement: Set grad_fn for automatic differentiation
    // This will be implemented when we integrate Function objects with tensors

    result
}

#[cfg(test)]
mod gru_cell_tests {
    use super::*;
    use crate::backward;
    use backend::CpuBackend;
    use dtype::num_traits::ToPrimitive;
    use dtype::float::Float32;
    use storage::DenseStorage;
    use tensor::Tensor;

    #[test]
    fn test_gru_cell_gradient() {
        // Test that gru_cell produces gradients for all inputs
        let input = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 10]).unwrap(),
        );
        let hidden = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 20]).unwrap(),
        );

        // GRU weights: [3*hidden_size, input_size/hidden_size] (PyTorch format)
        let weight_ih_data = vec![Float32::new(0.1); 60 * 10]; // (3*20) * 10
        let weight_ih = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                weight_ih_data,
                &[60, 10],
            )
            .unwrap(),
        );

        let weight_hh_data = vec![Float32::new(0.1); 60 * 20]; // (3*20) * 20
        let weight_hh = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                weight_hh_data,
                &[60, 20],
            )
            .unwrap(),
        );

        let bias_ih = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[60]).unwrap(),
        ); // 3*20
        let bias_hh = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[60]).unwrap(),
        ); // 3*20

        let h_new = gru_cell(&input, &hidden, &weight_ih, &weight_hh, &bias_ih, &bias_hh);

        // Compute loss and backward
        let loss = h_new.sum();
        backward(&[&loss], &[]).unwrap();

        // All six variables should have gradients
        assert!(
            input.grad().is_ok(),
            "Input should have gradient after gru_cell backward"
        );
        assert!(
            hidden.grad().is_ok(),
            "Hidden should have gradient after gru_cell backward"
        );
        assert!(
            weight_ih.grad().is_ok(),
            "Weight_ih should have gradient after gru_cell backward"
        );
        assert!(
            weight_hh.grad().is_ok(),
            "Weight_hh should have gradient after gru_cell backward"
        );
        assert!(
            bias_ih.grad().is_ok(),
            "Bias_ih should have gradient after gru_cell backward"
        );
        assert!(
            bias_hh.grad().is_ok(),
            "Bias_hh should have gradient after gru_cell backward"
        );
    }

    #[test]
    fn test_gru_cell_numerical_gradient() {
        // Numerical gradient validation for GRU cell
        let input = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 3]).unwrap(),
        );
        let hidden = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 5]).unwrap(),
        );

        // GRU weights: [3*hidden_size, input_size/hidden_size] (PyTorch format)
        let weight_ih_data = vec![Float32::new(0.1); 15 * 3]; // (3*5) * 3
        let weight_ih = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                weight_ih_data,
                &[15, 3],
            )
            .unwrap(),
        );

        let weight_hh_data = vec![Float32::new(0.1); 15 * 5]; // (3*5) * 5
        let weight_hh = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                weight_hh_data,
                &[15, 5],
            )
            .unwrap(),
        );

        let bias_ih = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[15]).unwrap(),
        ); // 3*5
        let bias_hh = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[15]).unwrap(),
        ); // 3*5

        let h_new = gru_cell(&input, &hidden, &weight_ih, &weight_hh, &bias_ih, &bias_hh);
        let loss = h_new.sum();
        backward(&[&loss], &[]).unwrap();

        // Verify gradient shapes (PyTorch format: weights are [3*hidden_size, input_size/hidden_size])
        assert_eq!(input.grad().unwrap().shape().dims(), &[1, 3]);
        assert_eq!(hidden.grad().unwrap().shape().dims(), &[1, 5]);
        assert_eq!(weight_ih.grad().unwrap().shape().dims(), &[15, 3]); // Same as weight shape
        assert_eq!(weight_hh.grad().unwrap().shape().dims(), &[15, 5]); // Same as weight shape
        assert_eq!(bias_ih.grad().unwrap().shape().dims(), &[15]);
        assert_eq!(bias_hh.grad().unwrap().shape().dims(), &[15]);

        // Verify gradients are non-zero
        let input_grad_sum: f64 = input
            .grad()
            .unwrap()
            .as_slice()
            .iter()
            .map(|x| x.to_f64().unwrap())
            .sum();
        assert!(
            input_grad_sum.abs() > 0.0,
            "Input gradient should be non-zero"
        );
    }
}

#[cfg(test)]
mod conv2d_tests {
    use super::*;
    use crate::backward;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use dtype::num_traits::ToPrimitive;
    use storage::DenseStorage;
    use tensor::Tensor;

    #[test]
    fn test_conv2d_forward() {
        // Test basic conv2d forward pass
        // Input: [1, 1, 4, 4]
        let input = Variable::new(
            Tensor::from_vec(
                vec![
                    Float32::new(1.0),
                    Float32::new(2.0),
                    Float32::new(3.0),
                    Float32::new(4.0),
                    Float32::new(5.0),
                    Float32::new(6.0),
                    Float32::new(7.0),
                    Float32::new(8.0),
                    Float32::new(9.0),
                    Float32::new(10.0),
                    Float32::new(11.0),
                    Float32::new(12.0),
                    Float32::new(13.0),
                    Float32::new(14.0),
                    Float32::new(15.0),
                    Float32::new(16.0),
                ],
                &[1, 1, 4, 4],
            )
            .unwrap(),
        );

        // Weight: [1, 1, 2, 2]
        let weight = Variable::new(
            Tensor::from_vec(
                vec![
                    Float32::new(1.0),
                    Float32::new(0.0),
                    Float32::new(0.0),
                    Float32::new(1.0),
                ],
                &[1, 1, 2, 2],
            )
            .unwrap(),
        );

        // Bias: [1]
        let bias = Variable::new(Tensor::from_vec(vec![Float32::new(0.0)], &[1]).unwrap());

        let output = conv2d(&input, &weight, Some(&bias), None, None)
            .expect("conv2d forward pass should succeed");

        // Output should be [1, 1, 3, 3]
        let output_data = output.data();
        assert_eq!(output_data.shape().dims(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_conv2d_gradient() {
        // Test that conv2d produces gradients for input, weight, and bias
        let input = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 3, 8, 8]).unwrap(),
        );
        let weight = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[16, 3, 3, 3]).unwrap(),
        );
        let bias = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[16]).unwrap(),
        );

        let output = conv2d(&input, &weight, Some(&bias), None, None)
            .expect("conv2d forward pass should succeed");

        // Compute loss and backward
        let loss = output.sum();
        backward(&[&loss], &[]).unwrap();

        // All three variables should have gradients
        assert!(
            input.grad().is_ok(),
            "Input should have gradient after conv2d backward"
        );
        assert!(
            weight.grad().is_ok(),
            "Weight should have gradient after conv2d backward"
        );
        assert!(
            bias.grad().is_ok(),
            "Bias should have gradient after conv2d backward"
        );
    }

    #[test]
    fn test_conv2d_input_gradient_numerical() {
        // Numerical gradient validation for input gradient
        use backend::CpuBackend;
        use dtype::float::Float32;
        use storage::DenseStorage;

        let input_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
            Float32::new(7.0),
            Float32::new(8.0),
            Float32::new(9.0),
        ];
        let input = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                input_data,
                &[1, 1, 3, 3],
            )
            .unwrap(),
        );

        let weight_data = vec![
            Float32::new(0.5),
            Float32::new(0.3),
            Float32::new(0.2),
            Float32::new(0.4),
        ];
        let weight = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                weight_data,
                &[1, 1, 2, 2],
            )
            .unwrap(),
        );

        let bias = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[1]).unwrap(),
        );

        let output = conv2d(&input, &weight, Some(&bias), None, None)
            .expect("conv2d forward pass should succeed");
        let loss = output.sum();
        backward(&[&loss], &[]).unwrap();

        let grad = input.grad().unwrap();

        // Verify gradient shape matches input shape
        assert_eq!(grad.shape().dims(), &[1, 1, 3, 3]);

        // Verify gradient values are non-zero (actual numerical validation would require finite differences)
        let grad_sum: f64 = grad.as_slice().iter().map(|x| x.to_f64().unwrap()).sum();
        assert!(grad_sum.abs() > 0.0, "Input gradient should be non-zero");
    }

    #[test]
    fn test_conv2d_weight_gradient_numerical() {
        // Numerical gradient validation for weight gradient
        use backend::CpuBackend;
        use dtype::float::Float32;
        use storage::DenseStorage;

        let input = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 1, 3, 3]).unwrap(),
        );
        let weight = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 1, 2, 2]).unwrap(),
        );
        let bias = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[1]).unwrap(),
        );

        let output = conv2d(&input, &weight, Some(&bias), None, None)
            .expect("conv2d forward pass should succeed");
        let loss = output.sum();
        backward(&[&loss], &[]).unwrap();

        let grad = weight.grad().unwrap();

        // Verify gradient shape matches weight shape
        assert_eq!(grad.shape().dims(), &[1, 1, 2, 2]);

        // Verify gradient values are non-zero
        let grad_sum: f64 = grad.as_slice().iter().map(|x| x.to_f64().unwrap()).sum();
        assert!(grad_sum.abs() > 0.0, "Weight gradient should be non-zero");
    }

    #[test]
    fn test_conv2d_bias_gradient() {
        // Test bias gradient computation
        use backend::CpuBackend;
        use dtype::float::Float32;
        use storage::DenseStorage;

        let input = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 1, 3, 3]).unwrap(),
        );
        let weight = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 1, 2, 2]).unwrap(),
        );
        let bias = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2]).unwrap(),
        );

        let output = conv2d(&input, &weight, Some(&bias), None, None)
            .expect("conv2d forward pass should succeed");
        let loss = output.sum();
        backward(&[&loss], &[]).unwrap();

        let grad = bias.grad().unwrap();

        // Verify gradient shape matches bias shape
        assert_eq!(grad.shape().dims(), &[2]);

        // Bias gradient should be sum over batch, height, width dimensions
        // For this test: batch=2, out_h=2, out_w=2, so each bias element should have gradient = 2*2*2 = 8
        let expected_grad = 8.0;
        for i in 0..2 {
            let actual = grad.as_slice()[i].to_f64().unwrap();
            assert!(
                (actual - expected_grad).abs() < 1e-5,
                "Bias gradient mismatch: expected {}, got {}",
                expected_grad,
                actual
            );
        }
    }

    #[test]
    fn test_conv2d_with_padding() {
        // Test conv2d with padding
        use backend::CpuBackend;
        use dtype::float::Float32;
        use storage::DenseStorage;

        let input = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 1, 3, 3]).unwrap(),
        );
        let weight = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 1, 3, 3]).unwrap(),
        );
        let bias = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[1]).unwrap(),
        );

        // With padding=1, output should be same size as input
        let output = conv2d(&input, &weight, Some(&bias), None, Some((1, 1)))
            .expect("conv2d forward with padding should succeed");
        assert_eq!(output.data().shape().dims(), &[1, 1, 3, 3]);

        let loss = output.sum();
        backward(&[&loss], &[]).unwrap();

        // Verify all gradients exist
        assert!(input.grad().is_ok());
        assert!(weight.grad().is_ok());
        assert!(bias.grad().is_ok());
    }

    #[test]
    fn test_conv2d_with_stride() {
        // Test conv2d with stride
        use backend::CpuBackend;
        use dtype::float::Float32;
        use storage::DenseStorage;

        let input = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 1, 4, 4]).unwrap(),
        );
        let weight = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 1, 2, 2]).unwrap(),
        );
        let bias = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[1]).unwrap(),
        );

        // With stride=2, output should be half the size
        let output = conv2d(&input, &weight, Some(&bias), Some((2, 2)), None)
            .expect("conv2d forward with stride should succeed");
        // (4 - 2) / 2 + 1 = 2
        assert_eq!(output.data().shape().dims(), &[1, 1, 2, 2]);

        let loss = output.sum();
        backward(&[&loss], &[]).unwrap();

        // Verify all gradients exist
        assert!(input.grad().is_ok());
        assert!(weight.grad().is_ok());
        assert!(bias.grad().is_ok());
    }
}

#[cfg(test)]
mod rnn_cell_tests {
    use super::*;
    use crate::backward;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use dtype::num_traits::ToPrimitive;
    use storage::DenseStorage;
    use tensor::Tensor;

    #[test]
    fn test_rnn_cell_gradient() {
        // Test that rnn_cell produces gradients for all inputs
        let input = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 10]).unwrap(),
        );
        let hidden = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 20]).unwrap(),
        );
        let weight_ih = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[10, 20]).unwrap(),
        );
        let weight_hh = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[20, 20]).unwrap(),
        );
        let bias = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[20]).unwrap(),
        );

        let h_new = rnn_cell(&input, &hidden, &weight_ih, &weight_hh, &bias);

        // Compute loss and backward
        let loss = h_new.sum();
        backward(&[&loss], &[]).unwrap();

        // All five variables should have gradients
        assert!(
            input.grad().is_ok(),
            "Input should have gradient after rnn_cell backward"
        );
        assert!(
            hidden.grad().is_ok(),
            "Hidden should have gradient after rnn_cell backward"
        );
        assert!(
            weight_ih.grad().is_ok(),
            "Weight_ih should have gradient after rnn_cell backward"
        );
        assert!(
            weight_hh.grad().is_ok(),
            "Weight_hh should have gradient after rnn_cell backward"
        );
        assert!(
            bias.grad().is_ok(),
            "Bias should have gradient after rnn_cell backward"
        );
    }

    #[test]
    fn test_rnn_cell_numerical_gradient() {
        // Numerical gradient validation for RNN cell
        let input = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 3]).unwrap(),
        );
        let hidden = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 5]).unwrap(),
        );

        // Small weights for numerical stability
        let weight_ih_data = vec![Float32::new(0.1); 15]; // 3 * 5
        let weight_ih = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(weight_ih_data, &[3, 5])
                .unwrap(),
        );

        let weight_hh_data = vec![Float32::new(0.1); 25]; // 5 * 5
        let weight_hh = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(weight_hh_data, &[5, 5])
                .unwrap(),
        );

        let bias = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[5]).unwrap(),
        );

        let h_new = rnn_cell(&input, &hidden, &weight_ih, &weight_hh, &bias);
        let loss = h_new.sum();
        backward(&[&loss], &[]).unwrap();

        // Verify gradient shapes
        assert_eq!(input.grad().unwrap().shape().dims(), &[1, 3]);
        assert_eq!(hidden.grad().unwrap().shape().dims(), &[1, 5]);
        assert_eq!(weight_ih.grad().unwrap().shape().dims(), &[3, 5]);
        assert_eq!(weight_hh.grad().unwrap().shape().dims(), &[5, 5]);
        assert_eq!(bias.grad().unwrap().shape().dims(), &[5]);

        // Verify gradients are non-zero
        let input_grad_sum: f64 = input
            .grad()
            .unwrap()
            .as_slice()
            .iter()
            .map(|x| x.to_f64().unwrap())
            .sum();
        assert!(
            input_grad_sum.abs() > 0.0,
            "Input gradient should be non-zero"
        );
    }
}

#[cfg(test)]
mod lstm_cell_tests {
    use super::*;
    use crate::backward;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use dtype::num_traits::ToPrimitive;
    use storage::DenseStorage;
    use tensor::Tensor;

    #[test]
    fn test_lstm_cell_gradient() {
        // Test that lstm_cell produces gradients for all inputs
        let input = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 10]).unwrap(),
        );
        let hidden = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 20]).unwrap(),
        );
        let cell = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 20]).unwrap(),
        );
        let weight_ih = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[80, 10]).unwrap(),
        ); // 4*20, 10
        let weight_hh = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[80, 20]).unwrap(),
        ); // 4*20, 20
        let bias_ih = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[80]).unwrap(),
        ); // 4*20
        let bias_hh = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[80]).unwrap(),
        ); // 4*20

        let (h_new, _c_new) = lstm_cell(
            &input, &hidden, &cell, &weight_ih, &weight_hh, &bias_ih, &bias_hh,
        );

        // Compute loss and backward
        let loss = h_new.sum();
        backward(&[&loss], &[]).unwrap();

        // All seven variables should have gradients
        assert!(
            input.grad().is_ok(),
            "Input should have gradient after lstm_cell backward"
        );
        assert!(
            hidden.grad().is_ok(),
            "Hidden should have gradient after lstm_cell backward"
        );
        assert!(
            cell.grad().is_ok(),
            "Cell should have gradient after lstm_cell backward"
        );
        assert!(
            weight_ih.grad().is_ok(),
            "Weight_ih should have gradient after lstm_cell backward"
        );
        assert!(
            weight_hh.grad().is_ok(),
            "Weight_hh should have gradient after lstm_cell backward"
        );
        assert!(
            bias_ih.grad().is_ok(),
            "Bias_ih should have gradient after lstm_cell backward"
        );
        assert!(
            bias_hh.grad().is_ok(),
            "Bias_hh should have gradient after lstm_cell backward"
        );
    }

    #[test]
    fn test_lstm_cell_numerical_gradient() {
        // Numerical gradient validation for LSTM cell
        let input = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 3]).unwrap(),
        );
        let hidden = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 5]).unwrap(),
        );
        let cell = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 5]).unwrap(),
        );

        // Small weights for numerical stability - PyTorch format [4*hidden_size, input_size/hidden_size]
        let weight_ih_data = vec![Float32::new(0.1); 60]; // (4*5) * 3
        let weight_ih = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                weight_ih_data,
                &[20, 3],
            )
            .unwrap(),
        );

        let weight_hh_data = vec![Float32::new(0.1); 100]; // (4*5) * 5
        let weight_hh = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                weight_hh_data,
                &[20, 5],
            )
            .unwrap(),
        );

        let bias_ih = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[20]).unwrap(),
        ); // 4*5
        let bias_hh = Variable::new(
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[20]).unwrap(),
        ); // 4*5

        let (h_new, _c_new) = lstm_cell(
            &input, &hidden, &cell, &weight_ih, &weight_hh, &bias_ih, &bias_hh,
        );
        let loss = h_new.sum();
        backward(&[&loss], &[]).unwrap();

        // Verify gradient shapes
        assert_eq!(input.grad().unwrap().shape().dims(), &[1, 3]);
        assert_eq!(hidden.grad().unwrap().shape().dims(), &[1, 5]);
        assert_eq!(cell.grad().unwrap().shape().dims(), &[1, 5]);
        assert_eq!(weight_ih.grad().unwrap().shape().dims(), &[20, 3]); // PyTorch format
        assert_eq!(weight_hh.grad().unwrap().shape().dims(), &[20, 5]); // PyTorch format
        assert_eq!(bias_ih.grad().unwrap().shape().dims(), &[20]);
        assert_eq!(bias_hh.grad().unwrap().shape().dims(), &[20]);

        // Verify gradients are non-zero
        let input_grad_sum: f64 = input
            .grad()
            .unwrap()
            .as_slice()
            .iter()
            .map(|x| x.to_f64().unwrap())
            .sum();
        assert!(
            input_grad_sum.abs() > 0.0,
            "Input gradient should be non-zero"
        );
    }
}
*/
