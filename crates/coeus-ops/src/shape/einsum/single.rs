//! Single-operand einsum patterns (transpose and diagonal trace).
//!
//! Every arm returns early; falling through is the typed rejection rather than
//! a silent no-op, so an unrecognised subscript can never read as success.

use super::subscript::{check_rank, unsupported};
use crate::backend_ops::BackendOps;
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Evaluate the one-operand pattern `lhs -> rhs` over `operands[0]`.
///
/// # Errors
///
/// Returns the caller's backend error for a rank rejection
/// ([`BackendError::EinsumRank`]) or an unsupported subscript
/// ([`BackendError::EinsumUnsupportedPattern`]); a recognised pattern returns
/// the contracted tensor.
#[inline]
pub(super) fn eval<T: Scalar, B: BackendOps<T> + Default>(
    subscript: &str,
    lhs: &str,
    rhs: &str,
    operands: &[&Tensor<T, B>],
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let a = operands[0];

    // "ij->ji" — 2-D transpose
    if lhs == "ij" && rhs == "ji" {
        check_rank::<T, B>(subscript, 0, a.ndim(), 2)?;
        return a
            .to_contiguous()?
            .permute(&[1, 0])
            .to_contiguous_on(backend);
    }

    // "...ij->...ji" / generic last-two-dims swap (e.g. "bij->bji")
    if a.ndim() >= 2 {
        let chars: Vec<char> = lhs.chars().collect();
        let rhs_chars: Vec<char> = rhs.chars().collect();
        // Fewer than two labels cannot name a last-two-dims swap (and
        // `n - 2` would underflow); fall through to the typed rejection.
        if chars.len() == rhs_chars.len() && chars.len() >= 2 {
            // Check if rhs is lhs with last two indices swapped
            let n = chars.len();
            let mut expected_rhs = chars.clone();
            expected_rhs.swap(n - 2, n - 1);
            if rhs_chars == expected_rhs {
                // The label count must match the tensor rank (an ellipsis
                // covers leading batch dims); otherwise "bij->bji" on a 2-D
                // tensor would silently permute the wrong axes.
                let explicit = lhs.strip_prefix("...").unwrap_or(lhs);
                let expected = explicit.chars().count();
                if (!lhs.starts_with("...") && a.ndim() != expected) || a.ndim() < expected {
                    return Err(B::Error::from(BackendError::EinsumRank {
                        subscript: subscript.to_string(),
                        operand: 0,
                        expected,
                        actual: a.ndim(),
                    }));
                }
                let mut perm: Vec<usize> = (0..a.ndim()).collect();
                perm.swap(a.ndim() - 2, a.ndim() - 1);
                return a.to_contiguous()?.permute(&perm).to_contiguous_on(backend);
            }
        }
    }

    // "ii->" — trace (sum of diagonal)
    if lhs == "ii" && rhs.is_empty() {
        check_rank::<T, B>(subscript, 0, a.ndim(), 2)?;
        let n = a.shape()[0].min(a.shape()[1]);
        let a_cont = a.to_contiguous()?;
        let a_s = a_cont.as_slice();
        let stride = a.shape()[1];
        let trace = (0..n)
            .map(|i| a_s[i * stride + i])
            .fold(T::ZERO, |acc, x| acc + x);
        return Tensor::from_slice(vec![1], &[trace]);
    }

    Err(unsupported::<T, B>(subscript, 1))
}
