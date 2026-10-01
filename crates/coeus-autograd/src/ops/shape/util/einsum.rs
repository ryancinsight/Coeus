// ── Tracked einsum ──
//
// Backward is derived analytically per pattern by delegating to the existing
// autograd op for the equivalent dispatched forward operation.
// For patterns that dispatch to matmul: matmul backward is handled by the
// matmul autograd node. We therefore compose einsum forward + autograd-tracked
// helpers rather than writing custom BackwardNode implementations.

use crate::var::Var;
use coeus_core::Scalar;

/// Why a tracked [`fn@einsum`]/[`fn@einsum3`] call was rejected.
///
/// Caller-supplied subscript strings are untrusted input (numpy/torch-style
/// einsum notation typed by the calling code, including across the Python
/// binding boundary): an unrecognized pattern is a typed error, never a
/// panic.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EinsumFailure<E> {
    /// The subscript names a pattern this tracked implementation does not
    /// recognize for the given operand count.
    UnsupportedPattern {
        /// The rejected subscript string.
        subscript: String,
        /// Number of operands the subscript was evaluated against.
        operand_count: usize,
    },
    /// The selected backend could not execute a tensor operation.
    Backend(E),
}

impl<E: core::fmt::Display> core::fmt::Display for EinsumFailure<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedPattern {
                subscript,
                operand_count,
            } => write!(
                f,
                "einsum: unsupported {operand_count}-operand pattern '{subscript}'"
            ),
            Self::Backend(error) => write!(f, "einsum backend operation failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for EinsumFailure<E> {}

impl<E> From<E> for EinsumFailure<E> {
    fn from(error: E) -> Self {
        Self::Backend(error)
    }
}

/// Tracked Einstein summation for common ML patterns.
///
/// Supported patterns:
/// - `"ij,jk->ik"` — matrix multiply (tracked via `crate::matmul`)
/// - `"bij,bjk->bik"` — batched matmul (tracked via `crate::matmul`)
/// - `"ij->ji"` — 2-D transpose (tracked via `crate::permute`)
/// - `"i,i->"` — dot product (tracked via element-wise mul + sum)
/// - `"i,j->ij"` — outer product (tracked via unsqueeze + broadcast + mul)
/// - `"ij,j->i"` — matrix-vector multiply (tracked via matmul + squeeze)
///
/// Gradients flow through the delegated tracked operations automatically.
///
/// # Errors
///
/// Returns [`EinsumFailure::UnsupportedPattern`] for a single-operand
/// subscript this implementation does not recognize.
#[inline]
pub fn einsum<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    subscript: &str,
    operands: &[&Var<T, B>],
) -> Result<Var<T, B>, EinsumFailure<B::Error>>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let subscript = subscript.trim();
    let (lhs_raw, rhs) = if let Some(pos) = subscript.find("->") {
        (&subscript[..pos], &subscript[pos + 2..])
    } else {
        (subscript, "")
    };
    let lhs_parts: Vec<&str> = lhs_raw.split(',').map(str::trim).collect();

    assert_eq!(
        lhs_parts.len(),
        operands.len(),
        "einsum: subscript has {} operand(s) but {} Var(s) provided",
        lhs_parts.len(),
        operands.len()
    );

    let rhs = rhs.trim();

    // ── Single-operand ────────────────────────────────────────────────────
    if operands.len() == 1 {
        let a = operands[0];
        let lhs = lhs_parts[0];

        // "ij->ji" — 2-D transpose
        if lhs == "ij" && rhs == "ji" {
            assert_eq!(a.tensor.ndim(), 2, "einsum ij->ji: requires 2-D input");
            return crate::ops::permute(a, &[1, 0]).map_err(Into::into);
        }

        // generic last-two-dims swap (e.g. "bij->bji")
        if a.tensor.ndim() >= 2 {
            let chars: Vec<char> = lhs.chars().collect();
            let rhs_chars: Vec<char> = rhs.chars().collect();
            if chars.len() == rhs_chars.len() {
                let n = chars.len();
                let mut expected_rhs = chars.clone();
                expected_rhs.swap(n - 2, n - 1);
                if rhs_chars == expected_rhs {
                    let mut perm: Vec<usize> = (0..a.tensor.ndim()).collect();
                    perm.swap(a.tensor.ndim() - 2, a.tensor.ndim() - 1);
                    return crate::ops::permute(a, &perm).map_err(Into::into);
                }
            }
        }

        // "ii->" — trace (non-differentiable w.r.t. off-diagonal; forward only)
        if lhs == "ii" && rhs.is_empty() {
            assert_eq!(a.tensor.ndim(), 2, "einsum ii->: requires 2-D input");
            let backend = B::default();
            let t = coeus_ops::einsum("ii->", &[&a.tensor], &backend)?;
            return Var::new(t, false).map_err(Into::into);
        }

        return Err(EinsumFailure::UnsupportedPattern {
            subscript: subscript.to_string(),
            operand_count: 1,
        });
    }

    // ── Two-operand ───────────────────────────────────────────────────────
    assert_eq!(operands.len(), 2);
    let a = operands[0];
    let b = operands[1];
    let a_lhs = lhs_parts[0];
    let b_lhs = lhs_parts[1];

    // "i,i->" — dot product (element-wise mul then sum)
    if a_lhs == "i" && b_lhs == "i" && rhs.is_empty() {
        let product = crate::ops::mul(a, b)?;
        return crate::ops::sum(&product).map_err(Into::into);
    }

    // "i,j->ij" — outer product via broadcast + mul
    if a_lhs == "i" && b_lhs == "j" && rhs == "ij" {
        let m = a.tensor.shape()[0];
        let n = b.tensor.shape()[0];
        let a_col = crate::ops::unsqueeze(a, 1)?; // [m, 1]
        let b_row = crate::ops::unsqueeze(b, 0)?; // [1, n]
        let a_bcast = crate::ops::broadcast_to(&a_col, vec![m, n])?;
        let b_bcast = crate::ops::broadcast_to(&b_row, vec![m, n])?;
        return crate::ops::mul(&a_bcast, &b_bcast).map_err(Into::into);
    }

    // "ij,jk->ik" — 2-D matrix multiply
    if a_lhs == "ij" && b_lhs == "jk" && rhs == "ik" {
        assert_eq!(a.tensor.ndim(), 2, "einsum ij,jk->ik: a must be 2-D");
        assert_eq!(b.tensor.ndim(), 2, "einsum ij,jk->ik: b must be 2-D");
        return crate::ops::matmul(a, b).map_err(Into::into);
    }

    // "bij,bjk->bik" — batched 3-D matrix multiply via per-batch slice + matmul + cat
    if a_lhs == "bij" && b_lhs == "bjk" && rhs == "bik" {
        assert_eq!(a.tensor.ndim(), 3, "einsum bij,bjk->bik: a must be 3-D");
        assert_eq!(b.tensor.ndim(), 3, "einsum bij,bjk->bik: b must be 3-D");
        let batch = a.tensor.shape()[0];
        let batch_results: Vec<Var<T, B>> = (0..batch)
            .map(|bi| {
                let a_i = crate::ops::slice(
                    a,
                    &(0..a.tensor.ndim())
                        .map(|d| {
                            if d == 0 {
                                (bi, bi + 1)
                            } else {
                                (0, a.tensor.shape()[d])
                            }
                        })
                        .collect::<Vec<_>>(),
                )?;
                let a_2d = crate::ops::squeeze(&a_i, Some(0))?;
                let b_i = crate::ops::slice(
                    b,
                    &(0..b.tensor.ndim())
                        .map(|d| {
                            if d == 0 {
                                (bi, bi + 1)
                            } else {
                                (0, b.tensor.shape()[d])
                            }
                        })
                        .collect::<Vec<_>>(),
                )?;
                let b_2d = crate::ops::squeeze(&b_i, Some(0))?;
                let mm = crate::ops::matmul(&a_2d, &b_2d)?;
                crate::ops::unsqueeze(&mm, 0)
            })
            .collect::<Result<_, _>>()?;
        let refs: Vec<&Var<T, B>> = batch_results.iter().collect();
        return crate::ops::cat(&refs, 0).map_err(Into::into);
    }

    // "ij,j->i" — matrix-vector multiply
    if a_lhs == "ij" && b_lhs == "j" && rhs == "i" {
        assert_eq!(a.tensor.ndim(), 2, "einsum ij,j->i: a must be 2-D");
        assert_eq!(b.tensor.ndim(), 1, "einsum ij,j->i: b must be 1-D");
        let k = b.tensor.shape()[0];
        let b_col = crate::ops::reshape(b, vec![k, 1])?;
        let mm = crate::ops::matmul(a, &b_col)?;
        return crate::ops::squeeze(&mm, Some(1)).map_err(Into::into);
    }

    // Fallback: run the non-tracked op and return non-differentiable result.
    let backend = B::default();
    let raw_operands: Vec<&coeus_tensor::Tensor<T, B>> =
        operands.iter().map(|v| &v.tensor).collect();
    let out = coeus_ops::einsum(subscript, &raw_operands, &backend)?;
    Var::new(out, false).map_err(Into::into)
}

/// Tracked 3-operand einsum via sequential pairwise contraction.
///
/// Supported patterns:
/// - `"ij,jk,kl->il"` — triple matmul chain
/// - `"bij,bjk,bkl->bil"` — batched triple matmul chain
///
/// # Errors
///
/// Returns [`EinsumFailure::UnsupportedPattern`] for any other subscript.
#[inline]
pub fn einsum3<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    subscript: &str,
    a: &Var<T, B>,
    b: &Var<T, B>,
    c: &Var<T, B>,
) -> Result<Var<T, B>, EinsumFailure<B::Error>>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let sub = subscript.trim();
    match sub {
        "ij,jk,kl->il" => {
            let ab = einsum("ij,jk->ik", &[a, b])?;
            einsum("ij,jk->ik", &[&ab, c])
        }
        "bij,bjk,bkl->bil" => {
            let ab = einsum("bij,bjk->bik", &[a, b])?;
            einsum("bij,bjk->bik", &[&ab, c])
        }
        _ => Err(EinsumFailure::UnsupportedPattern {
            subscript: subscript.to_string(),
            operand_count: 3,
        }),
    }
}
