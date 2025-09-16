//! Element-wise arithmetic operations

use crate::{FloatDtype, Result, Tensor, TensorError};

/// Element-wise maximum
pub fn maximum<T: FloatDtype>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
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

    let mut result = Tensor::from_vec(data, a.shape().to_vec());

    if a.requires_grad() || b.requires_grad() {
        result.set_requires_grad(true);
        // Note: Graph integration is handled by tensor methods, not free functions
    }

    Ok(result)
}

/// Element-wise minimum
pub fn minimum<T: FloatDtype>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
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

    let mut result = Tensor::from_vec(data, a.shape().to_vec());

    if a.requires_grad() || b.requires_grad() {
        result.set_requires_grad(true);
        // Note: Graph integration is handled by tensor methods, not free functions
    }

    Ok(result)
}

/// Element-wise power
pub fn pow<T: FloatDtype>(base: &Tensor<T>, exponent: &Tensor<T>) -> Result<Tensor<T>> {
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

    let mut result = Tensor::from_vec(data, base.shape().to_vec());

    if base.requires_grad() || exponent.requires_grad() {
        result.set_requires_grad(true);
        // Backward: ∂(b^e)/∂b = e * b^(e-1), ∂(b^e)/∂e = b^e * ln(b)
        // Edge cases: handle NaN/inf propagation
    }

    Ok(result)
}
