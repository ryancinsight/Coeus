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
pub enum EinsumError {
    /// The subscript names a pattern this tracked implementation does not
    /// recognize for the given operand count.
    UnsupportedPattern {
        /// The rejected subscript string.
        subscript: String,
        /// Number of operands the subscript was evaluated against.
        operand_count: usize,
    },
    /// The subscript names a different operand count than provided.
    OperandCountMismatch {
        /// The rejected subscript string.
        subscript: String,
        /// Comma-separated groups in the subscript.
        subscript_operands: usize,
        /// Provided operand count.
        provided: usize,
    },
    /// An operand's rank disagrees with its subscript pattern.
    RankMismatch {
        /// The rejected subscript string.
        subscript: String,
        /// Operand index whose rank disagrees.
        operand: usize,
        /// Rank the pattern requires.
        expected_ndim: usize,
        /// Operand's actual rank.
        actual_ndim: usize,
    },
    /// A contraction dimension disagrees between operands.
    DimMismatch {
        /// The rejected subscript string.
        subscript: String,
        /// Operand index whose dimension disagrees.
        operand: usize,
        /// Required extent.
        expected: usize,
        /// Operand's actual extent.
        actual: usize,
    },
    /// The non-tracked fallback execution failed for this subscript.
    ForwardFailed {
        /// The rejected subscript string.
        subscript: String,
        /// Fallback-reported failure detail.
        reason: String,
    },
}

impl core::fmt::Display for EinsumError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedPattern {
                subscript,
                operand_count,
            } => write!(
                f,
                "einsum: unsupported {operand_count}-operand pattern '{subscript}'"
            ),
            Self::OperandCountMismatch {
                subscript,
                subscript_operands,
                provided,
            } => write!(
                f,
                "einsum '{subscript}' names {subscript_operands} operands but {provided} were provided"
            ),
            Self::RankMismatch {
                subscript,
                operand,
                expected_ndim,
                actual_ndim,
            } => write!(
                f,
                "einsum '{subscript}' operand {operand} must be {expected_ndim}-D, got {actual_ndim}-D"
            ),
            Self::DimMismatch {
                subscript,
                operand,
                expected,
                actual,
            } => write!(
                f,
                "einsum '{subscript}' dimension mismatch on operand {operand}: expected {expected}, got {actual}"
            ),
            Self::ForwardFailed { subscript, reason } => {
                write!(f, "einsum '{subscript}' failed: {reason}")
            }
        }
    }
}

impl std::error::Error for EinsumError {}

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
/// Returns [`EinsumError::OperandCountMismatch`] when the subscript names a
/// different operand count than provided, [`EinsumError::RankMismatch`]
/// when an operand's rank disagrees with its pattern,
/// [`EinsumError::DimMismatch`] when a contraction dimension disagrees,
/// [`EinsumError::UnsupportedPattern`] for a subscript this implementation
/// does not recognize, or [`EinsumError::ForwardFailed`] when the
/// non-tracked fallback execution fails. Caller-supplied subscripts are
/// untrusted input: rejection is typed, never a panic.
#[inline]
pub fn einsum<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    subscript: &str,
    operands: &[&Var<T, B>],
) -> Result<Var<T, B>, EinsumError>
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

    if lhs_parts.len() != operands.len() {
        return Err(EinsumError::OperandCountMismatch {
            subscript: subscript.to_string(),
            subscript_operands: lhs_parts.len(),
            provided: operands.len(),
        });
    }

    let rhs = rhs.trim();

    // ── Single-operand ────────────────────────────────────────────────────
    if operands.len() == 1 {
        let a = operands[0];
        let lhs = lhs_parts[0];

        // "ij->ji" — 2-D transpose
        if lhs == "ij" && rhs == "ji" {
            if a.tensor.ndim() != 2 {
                return Err(EinsumError::RankMismatch {
                    subscript: subscript.to_string(),
                    operand: 0,
                    expected_ndim: 2,
                    actual_ndim: a.tensor.ndim(),
                });
            }
            return Ok(crate::ops::permute(a, &[1, 0]));
        }

        // generic last-two-dims swap (e.g. "bij->bji")
        if a.tensor.ndim() >= 2 {
            let chars: Vec<char> = lhs.chars().collect();
            let rhs_chars: Vec<char> = rhs.chars().collect();
            // Fewer than two labels cannot name a last-two-dims swap (and
            // `n - 2` would underflow); fall through to the typed rejection.
            if chars.len() == rhs_chars.len() && chars.len() >= 2 {
                let n = chars.len();
                let mut expected_rhs = chars.clone();
                expected_rhs.swap(n - 2, n - 1);
                if rhs_chars == expected_rhs {
                    let mut perm: Vec<usize> = (0..a.tensor.ndim()).collect();
                    perm.swap(a.tensor.ndim() - 2, a.tensor.ndim() - 1);
                    return Ok(crate::ops::permute(a, &perm));
                }
            }
        }

        // "ii->" — trace (non-differentiable w.r.t. off-diagonal; forward only)
        if lhs == "ii" && rhs.is_empty() {
            if a.tensor.ndim() != 2 {
                return Err(EinsumError::RankMismatch {
                    subscript: subscript.to_string(),
                    operand: 0,
                    expected_ndim: 2,
                    actual_ndim: a.tensor.ndim(),
                });
            }
            let backend = B::default();
            let t = coeus_ops::einsum("ii->", &[&a.tensor], &backend).map_err(|error| {
                EinsumError::ForwardFailed {
                    subscript: subscript.to_string(),
                    reason: error.to_string(),
                }
            })?;
            return Ok(Var::new(t, false));
        }

        return Err(EinsumError::UnsupportedPattern {
            subscript: subscript.to_string(),
            operand_count: 1,
        });
    }

    // ── Two-operand ───────────────────────────────────────────────────────
    if operands.len() != 2 {
        return Err(EinsumError::UnsupportedPattern {
            subscript: subscript.to_string(),
            operand_count: operands.len(),
        });
    }
    let a = operands[0];
    let b = operands[1];
    let a_lhs = lhs_parts[0];
    let b_lhs = lhs_parts[1];

    // "i,i->" — dot product (element-wise mul then sum)
    if a_lhs == "i" && b_lhs == "i" && rhs.is_empty() {
        for (operand, var) in [(0, a), (1, b)] {
            if var.tensor.ndim() != 1 {
                return Err(EinsumError::RankMismatch {
                    subscript: subscript.to_string(),
                    operand,
                    expected_ndim: 1,
                    actual_ndim: var.tensor.ndim(),
                });
            }
        }
        // Same label, one extent: without this, `mul` broadcasts or panics
        // on unbroadcastable shapes instead of reporting the mismatch.
        if a.tensor.shape()[0] != b.tensor.shape()[0] {
            return Err(EinsumError::DimMismatch {
                subscript: subscript.to_string(),
                operand: 1,
                expected: a.tensor.shape()[0],
                actual: b.tensor.shape()[0],
            });
        }
        let product = crate::ops::mul(a, b);
        return Ok(crate::ops::sum(&product));
    }

    // "i,j->ij" — outer product via broadcast + mul
    if a_lhs == "i" && b_lhs == "j" && rhs == "ij" {
        for (operand, var) in [(0, a), (1, b)] {
            if var.tensor.ndim() != 1 {
                return Err(EinsumError::RankMismatch {
                    subscript: subscript.to_string(),
                    operand,
                    expected_ndim: 1,
                    actual_ndim: var.tensor.ndim(),
                });
            }
        }
        let m = a.tensor.shape()[0];
        let n = b.tensor.shape()[0];
        let a_col = crate::ops::unsqueeze(a, 1); // [m, 1]
        let b_row = crate::ops::unsqueeze(b, 0); // [1, n]
        let a_bcast = crate::ops::broadcast_to(&a_col, vec![m, n]);
        let b_bcast = crate::ops::broadcast_to(&b_row, vec![m, n]);
        return Ok(crate::ops::mul(&a_bcast, &b_bcast));
    }

    // "ij,jk->ik" — 2-D matrix multiply
    if a_lhs == "ij" && b_lhs == "jk" && rhs == "ik" {
        for (operand, var) in [(0, a), (1, b)] {
            if var.tensor.ndim() != 2 {
                return Err(EinsumError::RankMismatch {
                    subscript: subscript.to_string(),
                    operand,
                    expected_ndim: 2,
                    actual_ndim: var.tensor.ndim(),
                });
            }
        }
        return Ok(crate::ops::matmul(a, b));
    }

    // "bij,bjk->bik" — batched 3-D matrix multiply via per-batch slice + matmul + cat
    if a_lhs == "bij" && b_lhs == "bjk" && rhs == "bik" {
        for (operand, var) in [(0, a), (1, b)] {
            if var.tensor.ndim() != 3 {
                return Err(EinsumError::RankMismatch {
                    subscript: subscript.to_string(),
                    operand,
                    expected_ndim: 3,
                    actual_ndim: var.tensor.ndim(),
                });
            }
        }
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
                );
                let a_2d = crate::ops::squeeze(&a_i, Some(0));
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
                );
                let b_2d = crate::ops::squeeze(&b_i, Some(0));
                let mm = crate::ops::matmul(&a_2d, &b_2d);
                crate::ops::unsqueeze(&mm, 0)
            })
            .collect();
        let refs: Vec<&Var<T, B>> = batch_results.iter().collect();
        return Ok(crate::ops::cat(&refs, 0));
    }

    // "ij,j->i" — matrix-vector multiply
    if a_lhs == "ij" && b_lhs == "j" && rhs == "i" {
        for (operand, var, expected_ndim) in [(0, a, 2), (1, b, 1)] {
            if var.tensor.ndim() != expected_ndim {
                return Err(EinsumError::RankMismatch {
                    subscript: subscript.to_string(),
                    operand,
                    expected_ndim,
                    actual_ndim: var.tensor.ndim(),
                });
            }
        }
        let k = b.tensor.shape()[0];
        let b_col = crate::ops::reshape(b, vec![k, 1]);
        let mm = crate::ops::matmul(a, &b_col);
        return Ok(crate::ops::squeeze(&mm, Some(1)));
    }

    // Fallback: run the non-tracked op and return non-differentiable result.
    let backend = B::default();
    let raw_operands: Vec<&coeus_tensor::Tensor<T, B>> =
        operands.iter().map(|v| &v.tensor).collect();
    let out = coeus_ops::einsum(subscript, &raw_operands, &backend).map_err(|error| {
        EinsumError::ForwardFailed {
            subscript: subscript.to_string(),
            reason: error.to_string(),
        }
    })?;
    Ok(Var::new(out, false))
}

/// Tracked 3-operand einsum via sequential pairwise contraction.
///
/// Supported patterns:
/// - `"ij,jk,kl->il"` — triple matmul chain
/// - `"bij,bjk,bkl->bil"` — batched triple matmul chain
///
/// # Errors
///
/// Returns [`EinsumError::UnsupportedPattern`] for any other subscript, or
/// the inner two-operand contraction's error.
#[inline]
pub fn einsum3<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    subscript: &str,
    a: &Var<T, B>,
    b: &Var<T, B>,
    c: &Var<T, B>,
) -> Result<Var<T, B>, EinsumError>
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
        _ => Err(EinsumError::UnsupportedPattern {
            subscript: subscript.to_string(),
            operand_count: 3,
        }),
    }
}
