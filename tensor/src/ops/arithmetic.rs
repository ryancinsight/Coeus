//! Element-wise arithmetic operations
//!
//! This module provides basic arithmetic operations for tensors, including:
//! - Addition, subtraction, multiplication, division
//! - Broadcasting support
//! - Element-wise math functions (sin, cos, exp, log, etc.)

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};
use num_traits::{Float, Num};

/// Element-wise addition with broadcasting support
pub fn add<
    T: DataType + std::ops::Add<Output = T> + Clone + Copy,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T>,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    if a.shape() == b.shape() {
        // Direct computation for matching shapes
        let data = a
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(x, y)| *x + *y)
            .collect();

        let mut result = Tensor::from_vec(data, a.shape().dims())?;

        if a.requires_grad || b.requires_grad {
            result = result.requires_grad_(true);
            // Backward: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
        }

        Ok(result)
    } else {
        // Broadcasting case
        broadcast_binary_op(a, b, |x, y| x + y)
    }
}

/// Element-wise multiplication with broadcasting support
pub fn mul<
    T: DataType + Num + Clone,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T>,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    if a.shape() == b.shape() {
        // Direct computation for matching shapes
        let data = a
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(x, y)| *x * *y)
            .collect();

        let mut result = Tensor::from_vec(data, a.shape().dims())?;

        if a.requires_grad() || b.requires_grad() {
            result = result.requires_grad_(true);
            // Backward: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
        }

        Ok(result)
    } else {
        // Broadcasting case
        broadcast_binary_op(a, b, |x, y| x * y)
    }
}

/// Element-wise division with broadcasting support
pub fn div<
    T: DataType + Num + Clone,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T>,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    if a.shape() == b.shape() {
        // Direct computation for matching shapes
        let data = a
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(x, y)| *x / *y)
            .collect();

        let mut result = Tensor::from_vec(data, a.shape().dims())?;

        if a.requires_grad() || b.requires_grad() {
            result = result.requires_grad_(true);
        }

        Ok(result)
    } else {
        // Broadcasting case
        broadcast_binary_op(a, b, |x, y| x / y)
    }
}

/// Element-wise subtraction with broadcasting support
pub fn sub<
    T: DataType + Num + Clone,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    if a.shape() == b.shape() {
        // Direct computation for matching shapes
        let data = a
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(x, y)| *x - *y)
            .collect();

        let mut result = Tensor::from_vec(data, a.shape().dims())?;

        if a.requires_grad() || b.requires_grad() {
            result = result.requires_grad_(true);
        }

        Ok(result)
    } else {
        // Broadcasting case
        broadcast_binary_op(a, b, |x, y| x - y)
    }
}

/// Element-wise negation
pub fn neg<
    T: DataType + Num + Clone + std::ops::Neg<Output = T>,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| -*x).collect();

    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad {
        result = result.requires_grad_(true);
    }

    Ok(result)
}

/// Generic broadcasting for binary operations
fn broadcast_binary_op<
    T: DataType + Clone,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
    op: impl Fn(T, T) -> T,
) -> Result<Tensor<B, S, T>> {
    // NumPy-style broadcasting: handle tensor operations with different but compatible shapes
    let result_shape = if a.shape().dims() == b.shape().dims() {
        a.shape().dims().to_vec()
    } else {
        // NumPy-style broadcasting logic
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        // Pad the shorter shape with leading dimensions of size 1
        let max_len = a_dims.len().max(b_dims.len());
        let mut a_padded = vec![1; max_len - a_dims.len()];
        a_padded.extend_from_slice(a_dims);
        let mut b_padded = vec![1; max_len - b_dims.len()];
        b_padded.extend_from_slice(b_dims);

        // Check if broadcasting is possible
        let mut result_dims: Vec<usize> = Vec::new();
        for (a_dim, b_dim) in a_padded.iter().zip(b_padded.iter()) {
            if *a_dim == *b_dim || *a_dim == 1 || *b_dim == 1 {
                result_dims.push(*a_dim.max(b_dim));
            } else {
                return Err(TensorError::BroadcastError {
                    lhs_shape: a_dims.to_vec(),
                    rhs_shape: b_dims.to_vec(),
                });
            }
        }
        result_dims
    };

    let broadcast_shape = &result_shape;
    let a_padded = pad_tensor_to_shape(a, broadcast_shape)?;
    let b_padded = pad_tensor_to_shape(b, broadcast_shape)?;

    let data = a_padded
        .as_slice()
        .iter()
        .zip(b_padded.as_slice())
        .map(|(x, y)| op(*x, *y))
        .collect();

    let mut result = Tensor::from_vec(data, broadcast_shape)?;

    if a.requires_grad || b.requires_grad {
        result = result.requires_grad_(true);
    }

    Ok(result)
}

/// Pad tensor to target shape for broadcasting
fn pad_tensor_to_shape<
    T: DataType + Clone,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    target_shape: &[usize],
) -> Result<Tensor<B, S, T>> {
    let source_shape = tensor.shape().dims();

    // Implement proper NumPy-style broadcasting
    let broadcasted_data = broadcast_tensor_data(tensor.as_slice(), source_shape, target_shape)?;

    Tensor::from_vec(broadcasted_data, target_shape)
}

/// Broadcast tensor data from source shape to target shape using NumPy-style broadcasting.
fn broadcast_tensor_data<T: DataType + Clone>(
    source_data: &[T],
    source_shape: &[usize],
    target_shape: &[usize],
) -> Result<Vec<T>> {
    if source_shape == target_shape {
        return Ok(source_data.to_vec());
    }

    // Align shapes from the right (NumPy broadcasting)
    let max_ndim = source_shape.len().max(target_shape.len());
    let mut source_padded = vec![1; max_ndim - source_shape.len()];
    source_padded.extend_from_slice(source_shape);
    let mut target_padded = vec![1; max_ndim - target_shape.len()];
    target_padded.extend_from_slice(target_shape);

    // Check broadcasting compatibility
    for (s, t) in source_padded.iter().zip(target_padded.iter()) {
        if *s != *t && *s != 1 && *t != 1 {
            return Err(TensorError::BroadcastError {
                lhs_shape: source_shape.to_vec(),
                rhs_shape: target_shape.to_vec(),
            });
        }
    }

    // Calculate strides for source and target
    let mut source_strides = vec![1; max_ndim];
    for i in (0..max_ndim - 1).rev() {
        source_strides[i] = source_strides[i + 1] * source_padded[i + 1];
    }

    // Generate broadcasted data
    let total_elements: usize = target_padded.iter().product();
    let mut result = Vec::with_capacity(total_elements);

    for target_idx in 0..total_elements {
        // Convert linear index to multi-dimensional coordinates
        let mut coords = vec![0; max_ndim];
        let mut temp_idx = target_idx;
        for i in (0..max_ndim).rev() {
            coords[i] = temp_idx % target_padded[i];
            temp_idx /= target_padded[i];
        }

        // Map to source coordinates (broadcasting)
        let mut source_idx = 0;
        for i in 0..max_ndim {
            let source_coord = if source_padded[i] == 1 { 0 } else { coords[i] };
            source_idx += source_coord * source_strides[i];
        }

        result.push(source_data[source_idx]);
    }

    Ok(result)
}

/// Element-wise maximum
pub fn maximum<
    T: DataType + Float + Num + Clone,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            actual: b.shape().dims().to_vec(),
            operation: "maximum",
        });
    }

    let data = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| if *x > *y { *x } else { *y })
        .collect();

    let mut result = Tensor::from_vec(data, a.shape().dims())?;

    if a.requires_grad() || b.requires_grad() {
        result = result.requires_grad_(true);
    }

    Ok(result)
}

/// Element-wise minimum
pub fn minimum<
    T: DataType + Float + Num + Clone,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            actual: b.shape().dims().to_vec(),
            operation: "minimum",
        });
    }

    let data = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| if *x < *y { *x } else { *y })
        .collect();

    let mut result = Tensor::from_vec(data, a.shape().dims())?;

    if a.requires_grad() || b.requires_grad() {
        result = result.requires_grad_(true);
    }

    Ok(result)
}

/// Element-wise power
pub fn pow<
    T: DataType + Float + Num + Clone,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    base: &Tensor<B, S, T>,
    exponent: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    if base.shape() != exponent.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: base.shape().dims().to_vec(),
            actual: exponent.shape().dims().to_vec(),
            operation: "pow",
        });
    }

    let data = base
        .as_slice()
        .iter()
        .zip(exponent.as_slice())
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

    let mut result = Tensor::from_vec(data, base.shape().dims())?;

    if base.requires_grad() || exponent.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂(b^e)/∂b = e * b^(e-1), ∂(b^e)/∂e = b^e * ln(b)
        // Edge cases: handle NaN/inf propagation
    }

    Ok(result)
}

/// Element-wise absolute value
pub fn abs<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| x.abs()).collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂|x|/∂x = sign(x) = x/|x| for x≠0, undefined at 0
    }

    Ok(result)
}

/// Element-wise exponential
pub fn exp<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| x.exp()).collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad {
        result = result.requires_grad_(true);
        // Backward: ∂exp(x)/∂x = exp(x)
    }

    Ok(result)
}

/// Element-wise natural logarithm
pub fn log<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| x.ln()).collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad {
        result = result.requires_grad_(true);
        // Backward: ∂log(x)/∂x = 1/x
    }

    Ok(result)
}

/// Element-wise sine
pub fn sin<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| x.sin()).collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂sin(x)/∂x = cos(x)
    }

    Ok(result)
}

/// Element-wise cosine
pub fn cos<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| x.cos()).collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂cos(x)/∂x = -sin(x)
    }

    Ok(result)
}

/// Element-wise inverse cosine
pub fn acos<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| x.acos()).collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂acos(x)/∂x = -1/√(1-x²)
    }

    Ok(result)
}

/// Element-wise inverse tangent
pub fn atan<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| x.atan()).collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂atan(x)/∂x = 1/(1+x²)
    }

    Ok(result)
}

/// Element-wise error function
pub fn erf<
    T: DataType + num_traits::FromPrimitive,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor
        .as_slice()
        .iter()
        .map(|&x| {
            let x_f64 = num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0);
            let erf_f64 = statrs::function::erf::erf(x_f64);
            num_traits::FromPrimitive::from_f64(erf_f64).unwrap_or(T::zero())
        })
        .collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂erf(x)/∂x = (2/√π) * exp(-x²)
    }

    Ok(result)
}

/// Element-wise base-2 exponential
pub fn exp2<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| x.exp2()).collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂exp2(x)/∂x = exp2(x) * ln(2)
    }

    Ok(result)
}

/// Element-wise base-10 logarithm
pub fn log10<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| x.log10()).collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂log10(x)/∂x = 1/(x * ln(10))
    }

    Ok(result)
}

/// Element-wise base-2 logarithm
pub fn log2<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| x.log2()).collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂log2(x)/∂x = 1/(x * ln(2))
    }

    Ok(result)
}

/// Element-wise reciprocal square root
pub fn rsqrt<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor
        .as_slice()
        .iter()
        .map(|x| T::one() / x.sqrt())
        .collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad {
        result = result.requires_grad_(true);
        // Backward: ∂rsqrt(x)/∂x = -0.5 * rsqrt(x)^3
    }

    Ok(result)
}

/// Element-wise square root
pub fn sqrt<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| x.sqrt()).collect();
    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad {
        result = result.requires_grad_(true);
        // Backward: ∂sqrt(x)/∂x = 1/(2*sqrt(x))
    }

    Ok(result)
}

/// Scalar addition: tensor + scalar
pub fn scalar_add<
    T: DataType + std::ops::Add<Output = T> + Clone + Copy,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    scalar: T,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| *x + scalar).collect();

    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂(x+c)/∂x = 1
    }

    Ok(result)
}

/// Scalar multiplication: tensor * scalar
pub fn scalar_mul<
    T: DataType + std::ops::Mul<Output = T> + Clone + Copy,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    scalar: T,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| *x * scalar).collect();

    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂(x*c)/∂x = c
    }

    Ok(result)
}

/// Element-wise division by scalar
pub fn scalar_div<
    T: DataType + std::ops::Div<Output = T> + Clone + Copy,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    scalar: T,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| *x / scalar).collect();

    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂(x/c)/∂x = 1/c
    }

    Ok(result)
}

/// Element-wise subtraction of scalar
pub fn scalar_sub<
    T: DataType + std::ops::Sub<Output = T> + Clone + Copy,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    scalar: T,
) -> Result<Tensor<B, S, T>> {
    let data = tensor.as_slice().iter().map(|x| *x - scalar).collect();

    let mut result = Tensor::from_vec(data, tensor.shape().dims())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
        // Backward: ∂(x-c)/∂x = 1
    }

    Ok(result)
}

/// Element-wise power with scalar exponent
pub fn pow_scalar<
    T: DataType + Float + Num + Clone,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    base: &Tensor<B, S, T>,
    exponent: T,
) -> Result<Tensor<B, S, T>> {
    let data = base
        .as_slice()
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

    let mut result = Tensor::from_vec(data, base.shape().dims())?;

    if base.requires_grad {
        result = result.requires_grad_(true);
        // Gradient: ∂(x^y)/∂x = y * x^(y-1)
    }

    Ok(result)
}

/// Broadcasts a tensor to a target shape.
///
/// # Arguments
/// * `tensor` - The tensor to broadcast
/// * `target_shape` - The desired shape to broadcast to
///
/// # Returns
/// A new tensor with the broadcast shape
pub fn broadcast_to<
    T: DataType + Clone + Copy,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    target_shape: &[usize],
) -> Result<Tensor<B, S, T>> {
    let source_shape = tensor.shape().dims();

    // Check if broadcasting is possible
    if source_shape.len() > target_shape.len() {
        return Err(TensorError::ShapeMismatch {
            expected: target_shape.to_vec(),
            actual: source_shape.to_vec(),
            operation: "broadcast_to",
        });
    }

    // Pad source shape with leading dimensions of size 1 if necessary
    let mut padded_source = vec![1; target_shape.len() - source_shape.len()];
    padded_source.extend_from_slice(source_shape);

    // Check broadcasting rules
    for (source_dim, target_dim) in padded_source.iter().zip(target_shape.iter()) {
        if *source_dim != 1 && *source_dim != *target_dim {
            return Err(TensorError::ShapeMismatch {
                expected: target_shape.to_vec(),
                actual: source_shape.to_vec(),
                operation: "broadcast_to",
            });
        }
    }

    // Use proper NumPy-style broadcasting
    let broadcasted_data = broadcast_tensor_data(tensor.as_slice(), source_shape, target_shape)?;

    Tensor::from_vec(broadcasted_data, target_shape)
}

impl<B, S, T> Tensor<B, S, T>
where
    T: DataType + Num + Clone,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T>,
{
    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> Result<Tensor<B, S, T>> {
        mul(self, other)
    }
    
    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Result<Tensor<B, S, T>>
    where T: std::ops::Add<Output = T> + Copy {
        add(self, other)
    }
}

impl<B, S, T> Tensor<B, S, T>
where
    T: DataType + Num + Clone + std::ops::Neg<Output = T>,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
{
    /// Element-wise negation
    pub fn neg(&self) -> Result<Tensor<B, S, T>> {
        neg(self)
    }
}
