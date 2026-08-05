//! Provider-owned scalar exponentiation.

use crate::backend_ops::{ElementwiseOps, ScalarPowerOps};
use coeus_core::Float;
use coeus_tensor::Tensor;

/// Apply the native-precision scalar power `input.powf(exponent)`.
///
/// The backend captures the exponent as a scalar kernel parameter and keeps
/// input/output storage on the selected provider.
#[inline]
pub fn pow_scalar<T: Float, B: ElementwiseOps<T> + ScalarPowerOps<T>>(
    input: &Tensor<T, B>,
    exponent: T,
    backend: &B,
) -> Tensor<T, B> {
    let mut output = Tensor::alloc_on(input.shape_cloned(), backend);
    let (storage, layout) = output.storage_mut_and_layout();
    backend
        .elementwise_pow_scalar(input.storage(), input.layout(), exponent, storage, layout)
        .expect("scalar power provider dispatch");
    output
}
