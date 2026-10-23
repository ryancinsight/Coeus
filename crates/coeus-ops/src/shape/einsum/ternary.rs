//! Three-operand einsum chains, decomposed into two two-operand contractions.

use super::dispatch::einsum;
use crate::backend_ops::BackendOps;
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Evaluate a 3-operand einsum by pairwise contraction.
///
/// Pattern `"abc,bcd,def->..."` is decomposed into two 2-operand einsums.
/// Only patterns expressible as two sequential 2-operand contractions are
/// supported.  The intermediate subscript is inferred automatically.
///
/// Currently supported:
/// - `"ij,jk,kl->il"` — two sequential matmuls (3-layer linear chain).
/// - `"bij,bjk,bkl->bil"` — batched 3-layer linear chain.
///
/// # Errors
///
/// Returns [`BackendError::EinsumUnsupportedPattern`] for any other
/// subscript, or the inner two-operand contraction's error.
#[inline]
pub fn einsum3<T: Scalar, B: BackendOps<T> + Default>(
    subscript: &str,
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    c: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let sub = subscript.trim();
    match sub {
        // "ij,jk,kl->il" — triple matmul chain
        "ij,jk,kl->il" => {
            let ab = einsum("ij,jk->ik", &[a, b], backend)?;
            einsum("ij,jk->ik", &[&ab, c], backend)
        }
        // "bij,bjk,bkl->bil" — batched triple matmul chain
        "bij,bjk,bkl->bil" => {
            let ab = einsum("bij,bjk->bik", &[a, b], backend)?;
            einsum("bij,bjk->bik", &[&ab, c], backend)
        }
        _ => Err(B::Error::from(BackendError::EinsumUnsupportedPattern {
            subscript: subscript.to_string(),
            operand_count: 3,
        })),
    }
}
