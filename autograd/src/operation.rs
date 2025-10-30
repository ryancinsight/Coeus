//! Operations that participate in automatic differentiation
//!
//! **LEGACY CODE**: This operation-based autograd system is being replaced with
//! PyTorch-compatible automatic graph construction. See Sprint MS-6 for details.
//!
//! The current Operation enum stores full tensor data in each operation, leading to
//! excessive memory usage (100MB+ per Conv2D operation). The new Function-based
//! approach will provide automatic graph construction with O(1) memory per operation.

use coeus_backend::CpuBackend;
use coeus_dtype::traits::FloatExt;
use coeus_dtype::DataType;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;
use smallvec::{smallvec, SmallVec};

use crate::error::Result;
use crate::variable::Variable;

/// An operation in the computation graph
///
/// **PyTorch Equivalent**: `Function` objects (e.g., `MulBackward0`, `SumBackward0`)
/// Operations represent differentiable computations and know how to compute
/// gradients with respect to their inputs given gradients with respect to their outputs.
///
/// # Note on Circular References
/// This enum stores `Variable<T>` (which is `Arc<VariableInner<T>>`) creating a circular reference:
/// Variable → Operation → Variable. This is intentional and matches `PyTorch`'s design.
/// The circular reference is broken when variables go out of scope in user code.
///
/// # Current Architecture Issues
/// - **Operation-based**: Each operation contains full input/output data
/// - **Memory intensive**: Stores entire tensors in each operation
/// - **Not PyTorch-compatible**: Missing dynamic graph construction
///
/// # Required Architecture Changes
/// - **Node-based**: Operations become lightweight function pointers
/// - **Dynamic construction**: Graph built during forward pass
/// - **Lazy evaluation**: Gradients computed only when needed
#[derive(Debug)]
pub enum Operation<T: DataType> {
    /// Addition operation: lhs + rhs
    Add {
        /// Left-hand side operand
        lhs: Variable<T>,
        /// Right-hand side operand
        rhs: Variable<T>,
    },

    /// Multiplication operation: lhs * rhs
    Mul {
        /// Left-hand side operand
        lhs: Variable<T>,
        /// Right-hand side operand
        rhs: Variable<T>,
    },

    /// Subtraction operation: lhs - rhs
    Sub {
        /// Left-hand side operand
        lhs: Variable<T>,
        /// Right-hand side operand
        rhs: Variable<T>,
    },

    /// Division operation: lhs / rhs
    Div {
        /// Left-hand side operand
        lhs: Variable<T>,
        /// Right-hand side operand
        rhs: Variable<T>,
    },

    /// Power operation: lhs ^ rhs (lhs raised to the power of rhs)
    Pow {
        /// Base operand
        base: Variable<T>,
        /// Exponent operand
        exponent: Variable<T>,
    },

    /// Exponential operation: e^input
    Exp {
        /// Input operand
        input: Variable<T>,
    },

    /// Natural logarithm operation: ln(input)
    Log {
        /// Input operand
        input: Variable<T>,
    },

    /// Sine operation: sin(input)
    Sin {
        /// Input operand
        input: Variable<T>,
    },

    /// Cosine operation: cos(input)
    Cos {
        /// Input operand
        input: Variable<T>,
    },

    /// Sum reduction operation: sum(input)
    Sum {
        /// Input operand
        input: Variable<T>,
        /// Original input shape (for gradient broadcasting)
        input_shape: Vec<usize>,
    },

    /// Mean reduction operation: mean(input)
    Mean {
        /// Input operand
        input: Variable<T>,
        /// Original input shape (for gradient broadcasting)
        input_shape: Vec<usize>,
    },

    /// Matrix multiplication operation: lhs @ rhs
    Matmul {
        /// Left-hand side operand
        lhs: Variable<T>,
        /// Right-hand side operand
        rhs: Variable<T>,
    },

    /// 2D convolution operation: conv2d(input, weight) + bias
    Conv2D {
        /// Input tensor
        input: Variable<T>,
        /// Convolution weight
        weight: Variable<T>,
        /// Convolution bias (optional)
        bias: Option<Variable<T>>,
        /// Stride in height dimension
        stride_h: usize,
        /// Stride in width dimension
        stride_w: usize,
        /// Padding in height dimension
        padding_h: usize,
        /// Padding in width dimension
        padding_w: usize,
    },

    /// `ReLU` activation: max(0, x)
    ReLU {
        /// Input operand
        input: Variable<T>,
    },

    /// Sigmoid activation: 1 / (1 + e^(-x))
    Sigmoid {
        /// Input operand
        input: Variable<T>,
    },

    /// Tanh activation: (e^x - e^(-x)) / (e^x + e^(-x))
    Tanh {
        /// Input operand
        input: Variable<T>,
    },

    /// GELU activation: x * Φ(x) where Φ is Gaussian CDF
    GELU {
        /// Input operand
        input: Variable<T>,
    },

    /// RNN cell operation: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
    RNNCell {
        /// Input tensor x_t [batch, input_size]
        input: Variable<T>,
        /// Previous hidden state h_{t-1} [batch, hidden_size]
        hidden: Variable<T>,
        /// Input-to-hidden weight W_ih [input_size, hidden_size]
        weight_ih: Variable<T>,
        /// Hidden-to-hidden weight W_hh [hidden_size, hidden_size]
        weight_hh: Variable<T>,
        /// Bias b [hidden_size]
        bias: Variable<T>,
    },

    /// LSTM cell operation: single time step with gates
    /// c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    /// h_t = o_t ⊙ tanh(c_t)
    LSTMCell {
        /// Input tensor x_t [batch, input_size]
        input: Variable<T>,
        /// Previous hidden state h_{t-1} [batch, hidden_size]
        hidden: Variable<T>,
        /// Previous cell state c_{t-1} [batch, hidden_size]
        cell: Variable<T>,
        /// Input-to-hidden weight W_ih [input_size, 4*hidden_size] (i,f,g,o gates)
        weight_ih: Variable<T>,
        /// Hidden-to-hidden weight W_hh [hidden_size, 4*hidden_size]
        weight_hh: Variable<T>,
        /// Input-to-hidden bias b_ih [4*hidden_size]
        bias_ih: Variable<T>,
        /// Hidden-to-hidden bias b_hh [4*hidden_size]
        bias_hh: Variable<T>,
    },

    /// GRU cell operation: single time step with reset and update gates
    /// h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ tanh(W_h @ (r_t ⊙ h_{t-1}) + W_x @ x_t + b_h + b_x)
    GRUCell {
        /// Input tensor x_t [batch, input_size]
        input: Variable<T>,
        /// Previous hidden state h_{t-1} [batch, hidden_size]
        hidden: Variable<T>,
        /// Input-to-hidden weight W_ih [input_size, 3*hidden_size] (r,z,n gates)
        weight_ih: Variable<T>,
        /// Hidden-to-hidden weight W_hh [hidden_size, 3*hidden_size]
        weight_hh: Variable<T>,
        /// Bias b_ih [3*hidden_size]
        bias_ih: Variable<T>,
        /// Bias b_hh [3*hidden_size]
        bias_hh: Variable<T>,
    },

    /// RNN sequence operation: processes entire input sequence
    RNN {
        /// Input sequence [seq_len, batch, input_size]
        input: Variable<T>,
        /// Initial hidden state [num_layers * num_directions, batch, hidden_size]
        hidden: Variable<T>,
        /// Input-to-hidden weights [num_layers * num_directions, input_size, hidden_size]
        weight_ih: Vec<Variable<T>>,
        /// Hidden-to-hidden weights [num_layers * num_directions, hidden_size, hidden_size]
        weight_hh: Vec<Variable<T>>,
        /// Biases [num_layers * num_directions, hidden_size]
        bias: Vec<Variable<T>>,
        /// Number of layers
        num_layers: usize,
        /// Bidirectional flag
        bidirectional: bool,
        /// Batch first flag
        batch_first: bool,
    },

    /// LSTM sequence operation: processes entire input sequence
    LSTM {
        /// Input sequence [seq_len, batch, input_size]
        input: Variable<T>,
        /// Initial hidden state [num_layers * num_directions, batch, hidden_size]
        hidden: Variable<T>,
        /// Initial cell state [num_layers * num_directions, batch, hidden_size]
        cell: Variable<T>,
        /// Input-to-hidden weights [num_layers * num_directions, input_size, hidden_size]
        weight_ih: Vec<Variable<T>>,
        /// Hidden-to-hidden weights [num_layers * num_directions, hidden_size, hidden_size]
        weight_hh: Vec<Variable<T>>,
        /// Biases [num_layers * num_directions, hidden_size]
        bias: Vec<Variable<T>>,
        /// Number of layers
        num_layers: usize,
        /// Bidirectional flag
        bidirectional: bool,
        /// Batch first flag
        batch_first: bool,
    },

    /// GRU sequence operation: processes entire input sequence
    GRU {
        /// Input sequence [seq_len, batch, input_size]
        input: Variable<T>,
        /// Initial hidden state [num_layers * num_directions, batch, hidden_size]
        hidden: Variable<T>,
        /// Input-to-hidden weights [num_layers * num_directions, input_size, hidden_size]
        weight_ih: Vec<Variable<T>>,
        /// Hidden-to-hidden weights [num_layers * num_directions, hidden_size, hidden_size]
        weight_hh: Vec<Variable<T>>,
        /// Biases [num_layers * num_directions, hidden_size]
        bias: Vec<Variable<T>>,
        /// Number of layers
        num_layers: usize,
        /// Bidirectional flag
        bidirectional: bool,
        /// Batch first flag
        batch_first: bool,
    },
}

impl<T: DataType + std::ops::Neg<Output = T> + FloatExt + PartialOrd + std::ops::AddAssign> Operation<T> {
    /// Compute backward gradients for this operation
    ///
    /// **PyTorch Equivalent**: `Function.backward()` method
    /// Each operation knows its own backward pass, similar to PyTorch's Function objects.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient with respect to this operation's output
    ///
    /// # Returns
    /// Vector of gradients with respect to each input variable
    pub fn backward(
        &self,
        grad_output: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Vec<Tensor<CpuBackend, DenseStorage<T>, T>>> {
        match self {
            Operation::Conv2D {
                input,
                weight,
                bias,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
            } => {
                Self::conv2d_backward(
                    input.data(),
                    weight.data(),
                    bias.as_ref().map(|b| b.data()),
                    grad_output,
                    *stride_h,
                    *stride_w,
                    *padding_h,
                    *padding_w,
                )
            }
            // Stub implementation for other operations
            _ => Err(crate::error::AutogradError::InvalidOperation {
                operation: "Backward pass not yet implemented for this operation".to_string(),
            }),
        }
    }

    /// Compute Conv2D backward pass gradients
    ///
    /// For a Conv2D operation: output = conv2d(input, weight) + bias
    ///
    /// The gradients are:
    /// - ∇input = conv2d_transpose(∇output, weight)
    /// - ∇weight = conv2d(∇output_rotated_180, input_rotated_180, groups=1)
    /// - ∇bias = sum(∇output, axis=[0,2,3]) if bias exists
    fn conv2d_backward(
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
        weight: &Tensor<CpuBackend, DenseStorage<T>, T>,
        bias: Option<&Tensor<CpuBackend, DenseStorage<T>, T>>,
        grad_output: &Tensor<CpuBackend, DenseStorage<T>, T>,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<Vec<Tensor<CpuBackend, DenseStorage<T>, T>>> {
        let input_shape = input.shape().dims();
        let weight_shape = weight.shape().dims();
        let grad_output_shape = grad_output.shape().dims();

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let out_channels = weight_shape[0];
        let kernel_height = weight_shape[2];
        let kernel_width = weight_shape[3];

        // Compute input gradient using conv2d_transpose
        let grad_input = Self::conv2d_transpose_backward_input(
            grad_output,
            weight,
            input_shape,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        )?;

        // Compute weight gradient
        let grad_weight = Self::conv2d_backward_weight(
            input,
            grad_output,
            weight_shape,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        )?;

        // Compute bias gradient if bias exists
        let grad_bias = if bias.is_some() {
            Some(Self::conv2d_backward_bias(grad_output)?)
        } else {
            None
        };

        // Return gradients in order: input, weight, bias
        let mut gradients = vec![grad_input, grad_weight];
        if let Some(grad_bias) = grad_bias {
            gradients.push(grad_bias);
        }

        Ok(gradients)
    }

    /// Compute input gradient for Conv2D backward pass using transposed convolution
    fn conv2d_transpose_backward_input(
        grad_output: &Tensor<CpuBackend, DenseStorage<T>, T>,
        weight: &Tensor<CpuBackend, DenseStorage<T>, T>,
        input_shape: &[usize],
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let grad_output_shape = grad_output.shape().dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let out_channels = grad_output_shape[1];
        let grad_output_height = grad_output_shape[2];
        let grad_output_width = grad_output_shape[3];

        // Initialize gradient input tensor
        let mut grad_input_data = vec![T::zero(); batch_size * in_channels * input_height * input_width];

        let grad_output_data = grad_output.as_slice();
        let weight_data = weight.as_slice();

        // Perform transposed convolution to compute input gradients
        for b in 0..batch_size {
            for ic in 0..in_channels {
                for ih in 0..input_height {
                    for iw in 0..input_width {
                        let mut sum = T::zero();

                        // Convolve grad_output with weight (transposed)
                        for oc in 0..out_channels {
                            for kh in 0..weight.shape().dims()[2] {
                                for kw in 0..weight.shape().dims()[3] {
                                    // Compute corresponding output position
                                    let oh_start = ih as isize - kh as isize + padding_h as isize;
                                    let ow_start = iw as isize - kw as isize + padding_w as isize;

                                    if oh_start >= 0 && ow_start >= 0 {
                                        let oh = oh_start as usize / stride_h;
                                        let ow = ow_start as usize / stride_w;

                                        // Check bounds and stride alignment
                                        if oh_start as usize % stride_h == 0
                                            && ow_start as usize % stride_w == 0
                                            && oh < grad_output_height
                                            && ow < grad_output_width
                                        {
                                            // Weight index (note: weight is [out_channels, in_channels, kh, kw])
                                            let weight_idx = ((oc * in_channels + ic) * weight.shape().dims()[2] + kh) * weight.shape().dims()[3] + kw;

                                            // Grad output index
                                            let grad_idx = ((b * out_channels + oc) * grad_output_height + oh) * grad_output_width + ow;

                                            sum = sum + grad_output_data[grad_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }
                        }

                        // Store gradient
                        let input_idx = ((b * in_channels + ic) * input_height + ih) * input_width + iw;
                        grad_input_data[input_idx] = sum;
                    }
                }
            }
        }

        Tensor::from_vec(grad_input_data, input_shape)
    }

    /// Compute weight gradient for Conv2D backward pass
    fn conv2d_backward_weight(
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
        grad_output: &Tensor<CpuBackend, DenseStorage<T>, T>,
        weight_shape: &[usize],
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        let grad_output_shape = grad_output.shape().dims();

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let out_channels = weight_shape[0];
        let kernel_height = weight_shape[2];
        let kernel_width = weight_shape[3];

        let grad_output_height = grad_output_shape[2];
        let grad_output_width = grad_output_shape[3];

        // Initialize weight gradient tensor
        let mut grad_weight_data = vec![T::zero(); out_channels * in_channels * kernel_height * kernel_width];

        let input_data = input.as_slice();
        let grad_output_data = grad_output.as_slice();

        // Compute weight gradients by correlating input with grad_output
        for oc in 0..out_channels {
            for ic in 0..in_channels {
                for kh in 0..kernel_height {
                    for kw in 0..kernel_width {
                        let mut sum = T::zero();

                        // Correlate input with grad_output at this kernel position
                        for b in 0..batch_size {
                            for oh in 0..grad_output_height {
                                for ow in 0..grad_output_width {
                                    // Corresponding input position
                                    let ih = oh * stride_h + kh;
                                    let iw = ow * stride_w + kw;

                                    // Apply padding
                                    if ih >= padding_h && iw >= padding_w {
                                        let ih_unpadded = ih - padding_h;
                                        let iw_unpadded = iw - padding_w;

                                        if ih_unpadded < input_height && iw_unpadded < input_width {
                                            // Input index
                                            let input_idx = ((b * in_channels + ic) * input_height + ih_unpadded) * input_width + iw_unpadded;

                                            // Grad output index
                                            let grad_idx = ((b * out_channels + oc) * grad_output_height + oh) * grad_output_width + ow;

                                            sum = sum + input_data[input_idx] * grad_output_data[grad_idx];
                                        }
                                    }
                                }
                            }
                        }

                        // Store weight gradient
                        let weight_idx = ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                        grad_weight_data[weight_idx] = sum;
                    }
                }
            }
        }

        Tensor::from_vec(grad_weight_data, weight_shape)
    }

    /// Compute bias gradient for Conv2D backward pass
    fn conv2d_backward_bias(
        grad_output: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let grad_output_shape = grad_output.shape().dims();
        let batch_size = grad_output_shape[0];
        let out_channels = grad_output_shape[1];
        let grad_output_height = grad_output_shape[2];
        let grad_output_width = grad_output_shape[3];

        let grad_output_data = grad_output.as_slice();
        let mut grad_bias_data = vec![T::zero(); out_channels];

        // Sum gradients over batch, height, and width dimensions
        for oc in 0..out_channels {
            let mut sum = T::zero();
            for b in 0..batch_size {
                for h in 0..grad_output_height {
                    for w in 0..grad_output_width {
                        let idx = ((b * out_channels + oc) * grad_output_height + h) * grad_output_width + w;
                        sum = sum + grad_output_data[idx];
                    }
                }
            }
            grad_bias_data[oc] = sum;
        }

        Tensor::from_vec(grad_bias_data, &[out_channels])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx;
    use coeus_dtype::float::Float32;

    #[test]
    fn test_add_backward() {
        let t1 = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();
        let v1 = Variable::new(t1);

        let t2 = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(3.0), Float32::new(4.0)],
            &[2],
        )
        .unwrap();
        let v2 = Variable::new(t2);

        let result = &v1 + &v2;
        let loss = result.sum_all();
        loss.backward().unwrap();

        // Check gradients
        assert_eq!(v1.grad().unwrap().as_slice(), &[Float32::new(1.0), Float32::new(1.0)]);
        assert_eq!(v2.grad().unwrap().as_slice(), &[Float32::new(1.0), Float32::new(1.0)]);
    }
}
