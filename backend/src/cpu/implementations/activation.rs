use crate::Result;
use dtype::{num_traits, DataType};
use storage::{DenseStorage, Storage};
use crate::cpu::activation;

pub fn relu_dense<T: DataType + PartialOrd + Default>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];

    activation::relu::relu_primitive(input_slice, &mut result)?;

    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn neg_strided<T: DataType + core::ops::Neg<Output = T>>(
    input: &storage::StridedStorage<T>,
) -> Result<storage::StridedStorage<T>> {
    let mut result_data = vec![T::default(); input.shape().size()];
    activation::math_ops::neg_strided_primitive(
        input.as_slice(),
        input.shape().dims(),
        input.strides(),
        input.offset(),
        &mut result_data,
    )?;

    storage::StridedStorage::new(result_data, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn neg_csr<T: DataType + core::ops::Neg<Output = T>>(
    input: &storage::CsrStorage<T>,
) -> Result<storage::CsrStorage<T>> {
    let (data, indices, indptr) = activation::math_ops::neg_csr_primitive(
        input.data(),
        input.indices(),
        input.indptr(),
        input.shape().dims(),
    )?;

    storage::CsrStorage::new(data, indices, indptr, input.shape().dims())
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn relu_strided<T: DataType + PartialOrd + Default>(
    input: &storage::StridedStorage<T>,
) -> Result<storage::StridedStorage<T>> {
    let mut result_data = vec![T::default(); input.shape().size()];
    activation::relu::relu_strided_primitive(
        input.as_slice(),
        input.shape().dims(),
        input.strides(),
        input.offset(),
        &mut result_data,
    )?;

    storage::StridedStorage::new(result_data, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn relu_csr<T: DataType + PartialOrd + Default>(
    input: &storage::CsrStorage<T>,
) -> Result<storage::CsrStorage<T>> {
    let (data, indices, indptr) = activation::relu::relu_csr_primitive(
        input.data(),
        input.indices(),
        input.indptr(),
        input.shape().dims(),
    )?;

    storage::CsrStorage::new(data, indices, indptr, input.shape().dims())
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn sigmoid_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];

    activation::sigmoid::sigmoid_primitive(input_slice, &mut result)?;

    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn sigmoid_strided<T: DataType + num_traits::Float>(
    input: &storage::StridedStorage<T>,
) -> Result<storage::StridedStorage<T>> {
    let mut result_data = vec![T::default(); input.shape().size()];
    activation::sigmoid::sigmoid_strided_primitive(
        input.as_slice(),
        input.shape().dims(),
        input.strides(),
        input.offset(),
        &mut result_data,
    )?;

    storage::StridedStorage::new(result_data, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn gelu_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let sqrt_2_over_pi = num_traits::NumCast::from(0.7978845608028654).unwrap_or(T::one());
    let coeff = num_traits::NumCast::from(0.044715).unwrap_or(T::zero());
    let half = num_traits::NumCast::from(0.5).unwrap_or(T::one());

    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];

    // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    for (i, &x) in input_slice.iter().enumerate() {
        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + coeff * x3);
        result[i] = half * x * (T::one() + inner.tanh());
    }

    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

// Math operations
pub fn exp_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];

    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.exp();
    }

    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn exp_strided<T: DataType + num_traits::Float>(
    input: &storage::StridedStorage<T>,
) -> Result<storage::StridedStorage<T>> {
    let mut result_data = vec![T::default(); input.shape().size()];
    activation::exp_strided_primitive(
        input.as_slice(),
        input.shape().dims(),
        input.strides(),
        input.offset(),
        &mut result_data,
    )?;

    storage::StridedStorage::new(result_data, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn log_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];

    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.ln();
    }

    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn log_strided<T: DataType + num_traits::Float>(
    input: &storage::StridedStorage<T>,
) -> Result<storage::StridedStorage<T>> {
    let mut result_data = vec![T::default(); input.shape().size()];
    activation::log_strided_primitive(
        input.as_slice(),
        input.shape().dims(),
        input.strides(),
        input.offset(),
        &mut result_data,
    )?;

    storage::StridedStorage::new(result_data, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn sin_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];

    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.sin();
    }

    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn cos_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];

    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.cos();
    }

    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn cos_strided<T: DataType + num_traits::Float>(
    input: &storage::StridedStorage<T>,
) -> Result<storage::StridedStorage<T>> {
    let mut result_data = vec![T::default(); input.shape().size()];
    activation::cos_strided_primitive(
        input.as_slice(),
        input.shape().dims(),
        input.strides(),
        input.offset(),
        &mut result_data,
    )?;

    storage::StridedStorage::new(result_data, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn tan_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.tan();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn tanh_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.tanh();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn tanh_strided<T: DataType + num_traits::Float>(
    input: &storage::StridedStorage<T>,
) -> Result<storage::StridedStorage<T>> {
    let mut result_data = vec![T::default(); input.shape().size()];
    activation::tanh::tanh_strided_primitive(
        input.as_slice(),
        input.shape().dims(),
        input.strides(),
        input.offset(),
        &mut result_data,
    )?;

    storage::StridedStorage::new(result_data, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn tanh_csr<T: DataType + num_traits::Float>(
    input: &storage::CsrStorage<T>,
) -> Result<storage::CsrStorage<T>> {
    let (data, indices, indptr) = activation::tanh::tanh_csr_primitive(
        input.data(),
        input.indices(),
        input.indptr(),
        input.shape().dims(),
    )?;

    storage::CsrStorage::new(data, indices, indptr, input.shape().dims())
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn asin_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.asin();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn acos_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.acos();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn atan_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.atan();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn sinh_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.sinh();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn cosh_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.cosh();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn sqrt_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.sqrt();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn sqrt_strided<T: DataType + num_traits::Float>(
    input: &storage::StridedStorage<T>,
) -> Result<storage::StridedStorage<T>> {
    let mut result_data = vec![T::default(); input.shape().size()];
    activation::sqrt_strided_primitive(
        input.as_slice(),
        input.shape().dims(),
        input.strides(),
        input.offset(),
        &mut result_data,
    )?;

    storage::StridedStorage::new(result_data, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn abs_dense<T: DataType + num_traits::Signed>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.abs();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn abs_strided<T: DataType + num_traits::Signed>(
    input: &storage::StridedStorage<T>,
) -> Result<storage::StridedStorage<T>> {
    let mut result_data = vec![T::default(); input.shape().size()];
    activation::abs_strided_primitive(
        input.as_slice(),
        input.shape().dims(),
        input.strides(),
        input.offset(),
        &mut result_data,
    )?;

    storage::StridedStorage::new(result_data, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn abs_csr<T: DataType + num_traits::Signed>(
    input: &storage::CsrStorage<T>,
) -> Result<storage::CsrStorage<T>> {
    let (data, indices, indptr) = activation::math_ops::abs_csr_primitive(
        input.data(),
        input.indices(),
        input.indptr(),
        input.shape().dims(),
    )?;

    storage::CsrStorage::new(data, indices, indptr, input.shape().dims())
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn floor_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.floor();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn ceil_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.ceil();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn round_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for i in 0..input_slice.len() {
        result[i] = input_slice[i].round();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

/// Approximation of erf using Abramowitz and Stegun formula
fn erf_approx<T: num_traits::Float>(x: T) -> T {
    let a1 = T::from(0.254829592).unwrap();
    let a2 = T::from(-0.284496736).unwrap();
    let a3 = T::from(1.421413741).unwrap();
    let a4 = T::from(-1.453152027).unwrap();
    let a5 = T::from(1.061405429).unwrap();
    let p = T::from(0.3275911).unwrap();

    let sign = if x < T::zero() {
        T::from(-1.0).unwrap()
    } else {
        T::one()
    };
    let x = x.abs();

    let t = T::one() / (T::one() + p * x);
    let y = T::one() - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

pub fn erf_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = erf_approx(x);
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn erfc_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        let x_f64 = num_traits::cast::<T, f64>(x).unwrap_or(0.0);
        let erfc_f64 = statrs::function::erf::erfc(x_f64);
        result[i] = num_traits::cast::<f64, T>(erfc_f64).unwrap_or(T::zero());
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn erfinv_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        let x_f64 = num_traits::cast::<T, f64>(x).unwrap_or(0.0);
        let erfinv_f64 = statrs::function::erf::erf_inv(x_f64);
        result[i] = num_traits::cast::<f64, T>(erfinv_f64).unwrap_or(T::zero());
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn rsqrt_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = T::one() / x.sqrt();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn log1p_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = (T::one() + x).ln();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn expm1_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = x.exp() - T::one();
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn reciprocal_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for (i, &x) in input_slice.iter().enumerate() {
        result[i] = T::one() / x;
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn atan2_dense<T: DataType + num_traits::Float>(
    y: &DenseStorage<T>,
    x: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    if y.shape() != x.shape() {
        return Err(crate::BackendError::InvalidInput(format!("Shape mismatch for atan2: {:?} vs {:?}", y.shape(), x.shape())));
    }
    let y_slice = y.as_slice();
    let x_slice = x.as_slice();
    let mut result = vec![T::default(); y_slice.len()];
    for i in 0..y_slice.len() {
        result[i] = y_slice[i].atan2(x_slice[i]);
    }
    DenseStorage::from_vec(result, y.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}
