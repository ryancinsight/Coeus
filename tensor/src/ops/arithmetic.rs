//! Element-wise arithmetic operations

use crate::{Dtype, FloatDtype, Result, Tensor, TensorError};
use num_traits::{Float, Num};
use coeus_backend::Backend;
use coeus_storage::TensorStorage;
use super::matrix::Broadcast;
use crate::core::tensor::Operation;

/// Element-wise addition with broadcasting support
pub fn add<T: Dtype + std::ops::Add<Output = T> + Clone + Copy, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(a: &Tensor<T, B, S>, b: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    if a.shape() == b.shape() {
        // Direct computation for matching shapes
        let data = a
            .data()
            .iter()
            .zip(b.data())
            .map(|(x, y)| *x + *y)
            .collect();

        let mut result = Tensor::from_vec(a.backend().clone(), data, a.shape().to_vec()).unwrap();

        if a.requires_grad() || b.requires_grad() {
            result.set_requires_grad(true);

            // Set up autograd graph
            use crate::core::tensor::{with_autograd_context, Operation};
            with_autograd_context(|context| {
                let a_node = if let Some(node) = a.node {
                    node
                } else {
                    context.create_leaf_node()
                };
                let b_node = if let Some(node) = b.node {
                    node
                } else {
                    context.create_leaf_node()
                };

                let input_data = vec![
                    a.data().iter().map(|&x| <T as Dtype>::to_f64(&x).unwrap_or(0.0)).collect::<Vec<f64>>(),
                    b.data().iter().map(|&x| <T as Dtype>::to_f64(&x).unwrap_or(0.0)).collect::<Vec<f64>>(),
                ];

                let result_node = context.create_node_with_data(Operation::Add, vec![a_node, b_node], input_data);
                result.node = Some(result_node);
            });
        }

        Ok(result)
    } else {
        // Broadcasting case
        broadcast_binary_op(a, b, |x, y| x + y, Operation::Add)
    }
}

/// Element-wise multiplication with broadcasting support
pub fn mul<T: Dtype + Float + Num + Clone, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(a: &Tensor<T, B, S>, b: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    if a.shape() == b.shape() {
        // Direct computation for matching shapes
        let data = a
            .data()
            .iter()
            .zip(b.data())
            .map(|(x, y)| *x * *y)
            .collect();

        let mut result = Tensor::from_vec(a.backend().clone(), data, a.shape().to_vec()).unwrap();

        if a.requires_grad() || b.requires_grad() {
            result.set_requires_grad(true);

            // Set up autograd graph
            use crate::core::tensor::{with_autograd_context, Operation};
            with_autograd_context(|context| {
                let a_node = if let Some(node) = a.node {
                    node
                } else {
                    context.create_leaf_node()
                };
                let b_node = if let Some(node) = b.node {
                    node
                } else {
                    context.create_leaf_node()
                };

                let input_data = vec![
                    a.data().iter().map(|&x| <T as Dtype>::to_f64(&x).unwrap_or(0.0)).collect::<Vec<f64>>(),
                    b.data().iter().map(|&x| <T as Dtype>::to_f64(&x).unwrap_or(0.0)).collect::<Vec<f64>>(),
                ];

                let result_node = context.create_node_with_data(Operation::Mul, vec![a_node, b_node], input_data);
                result.node = Some(result_node);
            });
        }

        Ok(result)
    } else {
        // Broadcasting case
        broadcast_binary_op(a, b, |x, y| x * y, Operation::Mul)
    }
}

/// Element-wise division with broadcasting support
pub fn div<T: Dtype + Float + Num + Clone, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(a: &Tensor<T, B, S>, b: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    if a.shape() == b.shape() {
        // Direct computation for matching shapes
        let data = a
            .data()
            .iter()
            .zip(b.data())
            .map(|(x, y)| *x / *y)
            .collect();

        let mut result = Tensor::from_vec(a.backend().clone(), data, a.shape().to_vec()).unwrap();

        if a.requires_grad() || b.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    } else {
        // Broadcasting case
        broadcast_binary_op(a, b, |x, y| x / y, Operation::Div)
    }
}

/// Element-wise subtraction with broadcasting support
pub fn sub<T: Dtype + Float + Num + Clone, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(a: &Tensor<T, B, S>, b: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    if a.shape() == b.shape() {
        // Direct computation for matching shapes
        let data = a
            .data()
            .iter()
            .zip(b.data())
            .map(|(x, y)| *x - *y)
            .collect();

        let mut result = Tensor::from_vec(a.backend().clone(), data, a.shape().to_vec()).unwrap();

        if a.requires_grad() || b.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    } else {
        // Broadcasting case
        broadcast_binary_op(a, b, |x, y| x - y, Operation::Sub)
    }
}

/// Element-wise negation
pub fn neg<T: Dtype + Float + Num + Clone, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    let data = tensor
        .data()
        .iter()
        .map(|x| -*x)
        .collect();

    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);
    }

    Ok(result)
}

/// Generic broadcasting for binary operations
fn broadcast_binary_op<T: Dtype + Clone, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(
    a: &Tensor<T, B, S>,
    b: &Tensor<T, B, S>,
    op: impl Fn(T, T) -> T,
    operation: Operation,
) -> Result<Tensor<T, B, S>> {
    if !Broadcast::can_broadcast(a.shape(), b.shape()) {
        return Err(TensorError::BroadcastingError {
            shape1: a.shape().to_vec(),
            shape2: b.shape().to_vec(),
        });
    }

    let broadcast_shape = Broadcast::broadcast_shape(a.shape(), b.shape());
    let a_padded = pad_tensor_to_shape(a, &broadcast_shape);
    let b_padded = pad_tensor_to_shape(b, &broadcast_shape);

    let data = a_padded
        .data()
        .iter()
        .zip(b_padded.data())
        .map(|(x, y)| op(*x, *y))
        .collect();

    let mut result = Tensor::from_vec(a.backend().clone(), data, broadcast_shape).unwrap();

    if a.requires_grad() || b.requires_grad() {
        result.set_requires_grad(true);

        // Set up autograd graph for broadcasting case
        use crate::core::tensor::with_autograd_context;
        with_autograd_context(|context| {
            let a_node = if let Some(node) = a.node {
                node
            } else {
                context.create_leaf_node()
            };
            let b_node = if let Some(node) = b.node {
                node
            } else {
                context.create_leaf_node()
            };

            let input_data = vec![
                a.data().iter().map(|&x| <T as Dtype>::to_f64(&x).unwrap_or(0.0)).collect::<Vec<f64>>(),
                b.data().iter().map(|&x| <T as Dtype>::to_f64(&x).unwrap_or(0.0)).collect::<Vec<f64>>(),
            ];

            let result_node = context.create_node_with_data(operation, vec![a_node, b_node], input_data);
            result.node = Some(result_node);
        });
    }

    Ok(result)
}

/// Pad tensor to target shape for broadcasting
fn pad_tensor_to_shape<T: Dtype + Clone, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>, target_shape: &[usize]) -> Tensor<T, B, S> {
    let num_elements = tensor.numel();
    let padding_factor = target_shape.iter().product::<usize>() / num_elements;
    let mut padded_data = Vec::with_capacity(target_shape.iter().product());

    for i in 0..num_elements {
        for _ in 0..padding_factor {
            padded_data.push(tensor.data()[i]);
        }
    }

    Tensor::from_vec(tensor.backend().clone(), padded_data, target_shape.to_vec()).unwrap()
}

/// Element-wise maximum
pub fn maximum<T: Dtype + Float + Num + Clone, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(a: &Tensor<T, B, S>, b: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }

    let data = a
        .data()
        .iter()
        .zip(b.data())
        .map(|(x, y)| if *x > *y { *x } else { *y })
        .collect();

    let mut result = Tensor::from_vec(a.backend().clone(), data, a.shape().to_vec()).unwrap();

    if a.requires_grad() || b.requires_grad() {
        result.set_requires_grad(true);
    }

    Ok(result)
}

/// Element-wise minimum
pub fn minimum<T: Dtype + Float + Num + Clone, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(a: &Tensor<T, B, S>, b: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }

    let data = a
        .data()
        .iter()
        .zip(b.data())
        .map(|(x, y)| if *x < *y { *x } else { *y })
        .collect();

    let mut result = Tensor::from_vec(a.backend().clone(), data, a.shape().to_vec()).unwrap();

    if a.requires_grad() || b.requires_grad() {
        result.set_requires_grad(true);
    }

    Ok(result)
}

/// Element-wise power
pub fn pow<T: Dtype + Float + Num + Clone, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(base: &Tensor<T, B, S>, exponent: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    if base.shape() != exponent.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: base.shape().to_vec(),
            actual: exponent.shape().to_vec(),
        });
    }

    let data = base
        .data()
        .iter()
        .zip(exponent.data())
        .map(|(b, e)| {
            if *b < T::zero() && T::from(2.0).is_some_and(|two| *e % two != T::zero()) {
                // Negative base with non-integer exponent -> complex, handle as NaN
                T::nan()
            } else if *b == T::zero() && *e < T::zero() {
                // 0^negative -> infinity
                T::infinity()
            } else {
                b.powf(*e)
            }
        })
        .collect();

    let mut result = Tensor::from_vec(base.backend().clone(), data, base.shape().to_vec()).unwrap();

    if base.requires_grad() || exponent.requires_grad() {
        result.set_requires_grad(true);
        // Backward: ∂(b^e)/∂b = e * b^(e-1), ∂(b^e)/∂e = b^e * ln(b)
        // Edge cases: handle NaN/inf propagation
    }

    Ok(result)
}

/// Element-wise absolute value
pub fn abs<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Tensor<T, B, S> {
    let data = tensor.data().iter().map(|x| x.abs()).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        // Backward: ∂|x|/∂x = sign(x) = x/|x| for x≠0, undefined at 0
    }

    result
}

/// Element-wise exponential
pub fn exp<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    let data = tensor.data().iter().map(|x| x.exp()).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);

        // Create computational graph node
        use crate::core::tensor::{with_autograd_context, Operation};
        with_autograd_context(|context| {
            let input_node = if let Some(node) = tensor.node {
                node
            } else {
                context.create_leaf_node()
            };

            // Store input tensor data for gradient computation
            let input_data_f64: Vec<f64> = tensor.data.iter().map(|&x| crate::Dtype::to_f64(&x)).collect::<Option<Vec<f64>>>().unwrap_or_default();
            let exp_node = context.create_node_with_data(Operation::Exp, vec![input_node], vec![input_data_f64]);
            result.node = Some(exp_node);
        });
    }

    Ok(result)
}

/// Element-wise natural logarithm
pub fn log<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    use crate::core::tensor::{with_autograd_context, Operation};

    let data = tensor.data().iter().map(|x| x.ln()).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);

        with_autograd_context(|context| {
            let input_node = if let Some(node) = tensor.node {
                node
            } else {
                context.create_leaf_node()
            };

            // Store input data for gradient computation (∂log(x)/∂x = 1/x)
            let input_data_f64: Vec<f64> = tensor.data().iter()
                .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                .collect();

            let result_node = context.create_node_with_data(Operation::Log, vec![input_node], vec![input_data_f64]);
            result.node = Some(result_node);
        });
    }

    Ok(result)
}

/// Element-wise sine
pub fn sin<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    use crate::core::tensor::{with_autograd_context, Operation};

    let data = tensor.data().iter().map(|x| x.sin()).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        with_autograd_context(|context| {
            let input_node = if let Some(node) = tensor.node {
                node
            } else {
                context.create_leaf_node()
            };

            // Store input tensor data for gradient computation (∂sin(x)/∂x = cos(x))
            let input_data_f64: Vec<f64> = tensor.data().iter().map(|&x| Dtype::to_f64(&x)).collect::<Option<Vec<f64>>>().unwrap_or_default();

            let sin_node = context.create_node_with_data(Operation::Sin, vec![input_node], vec![input_data_f64]);
            result.node = Some(sin_node);
        });
    }

    Ok(result)
}

/// Element-wise cosine
pub fn cos<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Result<Tensor<T, B, S>> {
    use crate::core::tensor::{with_autograd_context, Operation};

    let data = tensor.data().iter().map(|x| x.cos()).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec())?;

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        with_autograd_context(|context| {
            let input_node = if let Some(node) = tensor.node {
                node
            } else {
                context.create_leaf_node()
            };

            // Store input tensor data for gradient computation (∂cos(x)/∂x = -sin(x))
            let input_data_f64: Vec<f64> = tensor.data().iter().map(|&x| Dtype::to_f64(&x)).collect::<Option<Vec<f64>>>().unwrap_or_default();

            let cos_node = context.create_node_with_data(Operation::Cos, vec![input_node], vec![input_data_f64]);
            result.node = Some(cos_node);
        });
    }

    Ok(result)
}

/// Element-wise inverse cosine
pub fn acos<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Tensor<T, B, S> {
    let data = tensor.data().iter().map(|x| x.acos()).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        // Backward: ∂acos(x)/∂x = -1/√(1-x²)
    }

    result
}

/// Element-wise inverse tangent
pub fn atan<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Tensor<T, B, S> {
    let data = tensor.data().iter().map(|x| x.atan()).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        // Backward: ∂atan(x)/∂x = 1/(1+x²)
    }

    result
}

/// Element-wise error function
pub fn erf<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Tensor<T, B, S> {
    let data = tensor.data().iter().map(|&x| {
        let x_f64 = num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0);
        let erf_f64 = statrs::function::erf::erf(x_f64);
        num_traits::FromPrimitive::from_f64(erf_f64).unwrap_or(T::zero())
    }).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        // Backward: ∂erf(x)/∂x = (2/√π) * exp(-x²)
    }

    result
}

/// Element-wise base-2 exponential
pub fn exp2<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Tensor<T, B, S> {
    let data = tensor.data().iter().map(|x| x.exp2()).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        // Backward: ∂exp2(x)/∂x = exp2(x) * ln(2)
    }

    result
}

/// Element-wise base-10 logarithm
pub fn log10<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Tensor<T, B, S> {
    let data = tensor.data().iter().map(|x| x.log10()).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        // Backward: ∂log10(x)/∂x = 1/(x * ln(10))
    }

    result
}

/// Element-wise base-2 logarithm
pub fn log2<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Tensor<T, B, S> {
    let data = tensor.data().iter().map(|x| x.log2()).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        // Backward: ∂log2(x)/∂x = 1/(x * ln(2))
    }

    result
}

/// Element-wise reciprocal square root
pub fn rsqrt<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Tensor<T, B, S> {
    let data = tensor.data().iter().map(|x| T::one() / x.sqrt()).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        // Backward: ∂rsqrt(x)/∂x = -0.5 * rsqrt(x)^3
    }

    result
}

/// Element-wise square root
pub fn sqrt<T: FloatDtype, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(tensor: &Tensor<T, B, S>) -> Tensor<T, B, S> {
    use crate::core::tensor::{with_autograd_context, Operation};

    let data = tensor.data().iter().map(|x| x.sqrt()).collect();
    let mut result = Tensor::from_vec(tensor.backend().clone(), data, tensor.shape().to_vec()).unwrap();

    if tensor.requires_grad() {
        result.set_requires_grad(true);
        with_autograd_context(|context| {
            let input_node = if let Some(node) = tensor.node {
                node
            } else {
                context.create_leaf_node()
            };

            // Store input tensor data for gradient computation (∂sqrt(x)/∂x = 1/(2*sqrt(x)))
            let input_data_f64: Vec<f64> = tensor.data().iter().map(|&x| Dtype::to_f64(&x)).collect::<Option<Vec<f64>>>().unwrap_or_default();

            let sqrt_node = context.create_node_with_data(Operation::Sqrt, vec![input_node], vec![input_data_f64]);
            result.node = Some(sqrt_node);
        });
    }

    result
}

/// Element-wise power with scalar exponent
pub fn pow_scalar<T: Dtype + Float + Num + Clone, B: Backend<T> + Clone + Send + Sync, S: TensorStorage<T> + Clone + Send + Sync>(base: &Tensor<T, B, S>, exponent: T) -> Result<Tensor<T, B, S>> {
    let data = base
        .data()
        .iter()
        .map(|b| {
            if *b < T::zero() && T::from(2.0).is_some_and(|two| exponent % two != T::zero()) {
                // Negative base with non-integer exponent -> complex, handle as NaN
                T::nan()
            } else if *b == T::zero() && exponent < T::zero() {
                // 0^negative -> infinity
                T::infinity()
            } else {
                b.powf(exponent)
            }
        })
        .collect();

    let mut result = Tensor::from_vec(base.backend().clone(), data, base.shape().to_vec()).unwrap();

    if base.requires_grad() {
        result.set_requires_grad(true);

        // Create computational graph node
        use crate::core::tensor::{with_autograd_context, Operation};
        with_autograd_context(|context| {
            let input_node = if let Some(node) = base.node {
                node
            } else {
                context.create_leaf_node()
            };

            // Store base and exponent data for gradient computation
            let base_data_f64: Vec<f64> = base.data.iter().map(|&x| crate::Dtype::to_f64(&x)).collect::<Option<Vec<f64>>>().unwrap_or_default();
            let exp_data_f64: Vec<f64> = vec![crate::Dtype::to_f64(&exponent).unwrap_or(1.0); base_data_f64.len()]; // Repeat exponent for each element
            let pow_node = context.create_node_with_data(Operation::Pow, vec![input_node], vec![base_data_f64, exp_data_f64]);
            result.node = Some(pow_node);
        });
    }

    Ok(result)
}




