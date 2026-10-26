//! Real digamma through an explicit backend capability.
use crate::DigammaOps;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// Element-wise real digamma, the derivative of `ln|Gamma(input)|`.
///
/// Scalar numerical contracts belong to Eunomia; backend kernels are supplied
/// by Leto (CPU) or the corresponding device provider. Unsupported backends
/// cannot satisfy the capability bound.
#[inline]
pub fn digamma<T: Float, B: DigammaOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    // SAFETY: DigammaOps initializes every output element before success.
    let mut output = unsafe { Tensor::alloc_on(input.shape_cloned(), backend) }?;
    let (storage, layout) = output.storage_mut_and_layout()?;
    backend.elementwise_digamma(input.storage(), input.layout(), storage, layout)?;
    Ok(output)
}
