//! Operand-count dispatch for [`einsum`].
//!
//! The public entry point parses the subscript once, rejects a count the
//! caller cannot satisfy, and hands the validated operands to the leaf that
//! owns that arity. Keeping recognition tables per arity means a pattern added
//! to one leaf cannot silently change another's matching order.

use super::subscript::{check_operand_count, parse_subscript, unsupported};
use super::{binary, single};
use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Einstein summation over one or two tensor operands.
///
/// Supported patterns (whitespace around labels is ignored):
/// - `"ij,jk->ik"` — 2-D matrix multiply
/// - `"bij,bjk->bik"` — batched 3-D matrix multiply
/// - `"ij->ji"` — 2-D transpose
/// - `"i,i->"` — dot product (inner product)
/// - `"i,j->ij"` — outer product
/// - `"ii->"` — trace (sum of diagonal)
/// - `"ij,j->i"` — matrix-vector multiply
/// - `"bik,bk->bi"` — batched matrix-vector multiply
///
/// # Errors
///
/// Returns a [`BackendError`](coeus_core::BackendError) (as `B::Error`) when the
/// subscript names a different operand count than provided
/// ([`EinsumOperandCount`](coeus_core::BackendError::EinsumOperandCount)),
/// an operand's rank disagrees with its pattern
/// ([`EinsumRank`](coeus_core::BackendError::EinsumRank)), a contraction
/// dimension disagrees ([`EinsumDim`](coeus_core::BackendError::EinsumDim)), or
/// the subscript names no supported pattern
/// ([`EinsumUnsupportedPattern`](coeus_core::BackendError::EinsumUnsupportedPattern)).
/// Caller-supplied subscripts are untrusted input: rejection is typed, never
/// a panic.
#[inline]
pub fn einsum<T: Scalar, B: BackendOps<T> + Default>(
    subscript: &str,
    operands: &[&Tensor<T, B>],
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let (lhs_parts, rhs) = parse_subscript(subscript);
    check_operand_count::<T, B>(subscript, lhs_parts.len(), operands.len())?;
    match operands.len() {
        1 => single::eval(subscript, lhs_parts[0], rhs, operands, backend),
        2 => binary::eval(subscript, &lhs_parts, rhs, operands, backend),
        count => Err(unsupported::<T, B>(subscript, count)),
    }
}
