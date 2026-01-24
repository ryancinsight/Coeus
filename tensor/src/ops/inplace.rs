//! In-place operations.

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use num_traits::{Float, Zero};

/// Fills the tensor with the specified value.
pub fn fill_<B, S, T>(tensor: &mut Tensor<B, S, T>, value: T) -> Result<()>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static,
    T: DataType + Copy + 'static,
{
    // Note: Assuming Storage has as_mut_slice or we convert to dense and back?
    // Since tensor.storage is pub(crate), we can access it.
    // If S provides mutable access, we use it.
    // Ideally S implements a trait for mutable access.
    // For now, we rely on StorageToDense/FromVec if mutation isn't direct,
    // OR we assume S is DenseStorage or similar.
    // But pycoeus used `as_mut_slice`.
    // Let's use `to_dense_generic` (mutable?)
    // Actually, inplace usually means NO generic conversion if possible.
    // BUT we don't have a `StorageMut` trait visible.
    // Let's check if we can simply iterate generic backend?
    // Backend operations usually handle this.
    // But `pycoeus` did manual iteration. 
    // We will implement `fill_` assuming `as_mut_slice` exists on Generic Storage? NO.
    // We will assume `to_dense` logic or similar.
    // WAIT. If I write `tensor.storage.as_mut_slice()`, S must implement it.
    // I can't guarantee S has it without a trait bound.
    // I will try to use `tensor.storage.as_mut_slice()` and let the compiler tell me the trait.
    // If that fails, I'll need to define a trait or find the existing one.
    // However, `pycoeus` calls `inner.as_mut_slice()`. `inner` is `Tensor`.
    // So `Tensor` MUST implement `as_mut_slice`. I will try invoking it.
    
    let slice = tensor.as_mut_slice();
    for x in slice {
        *x = value;
    }
    Ok(())
}

/// Fills the tensor with zeros.
pub fn zero_<B, S, T>(tensor: &mut Tensor<B, S, T>) -> Result<()>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static,
    T: DataType + Copy + Zero + 'static,
{
    fill_(tensor, T::zero())
}

/// In-place addition of a scalar.
pub fn add_<B, S, T>(tensor: &mut Tensor<B, S, T>, value: T) -> Result<()>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static,
    T: DataType + Copy + std::ops::Add<Output = T> + 'static,
{
    let slice = tensor.as_mut_slice();
    for x in slice {
        *x = *x + value;
    }
    Ok(())
}

/// In-place subtraction of a scalar.
pub fn sub_<B, S, T>(tensor: &mut Tensor<B, S, T>, value: T) -> Result<()>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static,
    T: DataType + Copy + std::ops::Sub<Output = T> + 'static,
{
    let slice = tensor.as_mut_slice();
    for x in slice {
        *x = *x - value;
    }
    Ok(())
}

/// In-place multiplication by a scalar.
pub fn mul_<B, S, T>(tensor: &mut Tensor<B, S, T>, value: T) -> Result<()>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static,
    T: DataType + Copy + std::ops::Mul<Output = T> + 'static,
{
    let slice = tensor.as_mut_slice();
    for x in slice {
        *x = *x * value;
    }
    Ok(())
}

/// In-place division by a scalar.
pub fn div_<B, S, T>(tensor: &mut Tensor<B, S, T>, value: T) -> Result<()>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static,
    T: DataType + Copy + std::ops::Div<Output = T> + PartialEq + Zero + 'static,
{
    if value == T::zero() {
        return Err(crate::TensorError::InvalidOperation {
            operation: "div_",
            dtype: T::dtype(),
            reason: "Division by zero",
        });
    }
    let slice = tensor.as_mut_slice();
    for x in slice {
        *x = *x / value;
    }
    Ok(())
}

/// In-place absolute value.
pub fn abs_<
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static,
    T: DataType + Copy + PartialOrd + Zero + std::ops::Sub<Output = T> + 'static,
>(tensor: &mut Tensor<B, S, T>) -> Result<()> {
    let slice = tensor.as_mut_slice();
    let zero = T::zero();
    for x in slice {
        if *x < zero {
            *x = zero - *x;
        }
    }
    Ok(())
}


// Unary in-place math operations requiring Float

macro_rules! impl_unary_inplace {
    ($name:ident, $func:ident) => {
        /// In-place operation.
        pub fn $name<
            B: Backend<Data = T> + Clone + Default + 'static,
            S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static,
            T: DataType + Float + 'static,
        >(tensor: &mut Tensor<B, S, T>) -> Result<()> {
            let slice = tensor.as_mut_slice();
            for x in slice {
                *x = x.$func();
            }
            Ok(())
        }
    };
}

impl_unary_inplace!(sin_, sin);
impl_unary_inplace!(cos_, cos);
impl_unary_inplace!(tan_, tan);
impl_unary_inplace!(asin_, asin);
impl_unary_inplace!(acos_, acos);
impl_unary_inplace!(atan_, atan);
impl_unary_inplace!(sinh_, sinh);
impl_unary_inplace!(cosh_, cosh);
impl_unary_inplace!(tanh_, tanh);
impl_unary_inplace!(exp_, exp);
impl_unary_inplace!(log_, ln); // log is ln in rust traits usually
impl_unary_inplace!(sqrt_, sqrt);
impl_unary_inplace!(abs_float_, abs); 
impl_unary_inplace!(ceil_, ceil);
impl_unary_inplace!(floor_, floor);
impl_unary_inplace!(round_, round);
impl_unary_inplace!(neg_, neg); // neg() -> -self

// Special handling for sigmoid, etc methods not in Float trait directly?
// Actually sigmoid is not in Num::Float.
// We can use closure approach if needed.
