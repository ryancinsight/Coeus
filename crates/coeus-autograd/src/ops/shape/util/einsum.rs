// ── Tracked einsum ──
//
// Backward is derived analytically per pattern by delegating to the existing
// autograd op for the equivalent dispatched forward operation.
// For patterns that dispatch to matmul: matmul backward is handled by the
// matmul autograd node. We therefore compose einsum forward + autograd-tracked
// helpers rather than writing custom BackwardNode implementations.

use crate::var::Var;
use coeus_core::Scalar;
use std::sync::Arc;

/// Why a tracked [`fn@einsum`]/[`fn@einsum3`] call was rejected.
///
/// Caller-supplied subscript strings are untrusted input (numpy/torch-style
/// einsum notation typed by the calling code, including across the Python
/// binding boundary): an unrecognized pattern is a typed error, never a
/// panic.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum EinsumError {
    /// The subscript is not valid einsum syntax.
    #[error("einsum: malformed subscript '{subscript}'")]
    MalformedSubscript {
        /// The rejected subscript string.
        subscript: String,
    },
    /// The number of subscript operands differs from the supplied operands.
    #[error(
        "einsum: subscript '{subscript}' specifies {specified} operand(s), but {provided} were provided"
    )]
    OperandCountMismatch {
        /// The rejected subscript string.
        subscript: String,
        /// Number of operands named by the subscript.
        specified: usize,
        /// Number of operands supplied by the caller.
        provided: usize,
    },
    /// An operand rank does not match its subscript.
    #[error(
        "einsum: subscript '{subscript}' requires operand {operand} to have rank {expected}, got {actual}"
    )]
    RankMismatch {
        /// The rejected subscript string.
        subscript: String,
        /// Zero-based operand position.
        operand: usize,
        /// Rank required by the subscript.
        expected: usize,
        /// Rank supplied by the caller.
        actual: usize,
    },
    /// Two operands have incompatible contracted dimensions.
    #[error("einsum: subscript '{subscript}' cannot contract shapes {left:?} and {right:?}")]
    ShapeMismatch {
        /// The rejected subscript string.
        subscript: String,
        /// Shape of the left operand.
        left: Vec<usize>,
        /// Shape of the right operand.
        right: Vec<usize>,
    },
    /// The subscript names a pattern this tracked implementation does not
    /// recognize for the given operand count.
    #[error("einsum: unsupported {operand_count}-operand pattern '{subscript}'")]
    UnsupportedPattern {
        /// The rejected subscript string.
        subscript: String,
        /// Number of operands the subscript was evaluated against.
        operand_count: usize,
    },
    /// The non-tracked fallback rejected the operation.
    ///
    /// Type erasure is confined to this cold error path because the public
    /// error type spans every statically selected backend error type.
    #[error("einsum backend operation failed: {source}")]
    Backend {
        /// Provider error preserved as the source.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync>,
    },
}

fn valid_labels(labels: &str) -> bool {
    let labels = labels.strip_prefix("...").unwrap_or(labels);
    !labels.is_empty() && labels.chars().all(|label| label.is_ascii_alphabetic())
}

fn parse_subscript(subscript: &str) -> Result<(Vec<&str>, &str), EinsumError> {
    let subscript = subscript.trim();
    let (lhs, rhs) = subscript.split_once("->").unwrap_or((subscript, ""));
    let lhs_parts: Vec<&str> = lhs.split(',').map(str::trim).collect();
    let malformed = subscript.is_empty()
        || rhs.contains("->")
        || lhs_parts.iter().any(|labels| !valid_labels(labels))
        || (!rhs.trim().is_empty() && !valid_labels(rhs.trim()));
    if malformed {
        Err(EinsumError::MalformedSubscript {
            subscript: subscript.to_owned(),
        })
    } else {
        Ok((lhs_parts, rhs.trim()))
    }
}

fn validate_operands<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    subscript: &str,
    labels: &[&str],
    operands: &[&Var<T, B>],
) -> Result<(), EinsumError> {
    let mut extents = Vec::<(char, usize, usize)>::new();
    for (operand, (labels, var)) in labels.iter().zip(operands).enumerate() {
        let explicit = labels.strip_prefix("...").unwrap_or(labels);
        let expected = explicit.chars().count();
        let actual = var.tensor.ndim();
        if (!labels.starts_with("...") && actual != expected) || actual < expected {
            return Err(EinsumError::RankMismatch {
                subscript: subscript.to_owned(),
                operand,
                expected,
                actual,
            });
        }
        let offset = actual - expected;
        for (label, &extent) in explicit.chars().zip(&var.tensor.shape()[offset..]) {
            if let Some(&(_, previous, other)) = extents.iter().find(|(seen, ..)| *seen == label) {
                if previous != extent {
                    return Err(EinsumError::ShapeMismatch {
                        subscript: subscript.to_owned(),
                        left: operands[other].tensor.shape().to_vec(),
                        right: var.tensor.shape().to_vec(),
                    });
                }
            } else {
                extents.push((label, extent, operand));
            }
        }
    }
    Ok(())
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
/// Returns [`EinsumError`] when the subscript is malformed or unsupported,
/// operand counts, ranks, or shapes do not match, or the fallback backend
/// rejects the operation.
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
    let (lhs_parts, rhs) = parse_subscript(subscript)?;
    if lhs_parts.len() != operands.len() {
        return Err(EinsumError::OperandCountMismatch {
            subscript: subscript.to_owned(),
            specified: lhs_parts.len(),
            provided: operands.len(),
        });
    }
    validate_operands(subscript, &lhs_parts, operands)?;

    // ── Single-operand ────────────────────────────────────────────────────
    if operands.len() == 1 {
        let a = operands[0];
        let lhs = lhs_parts[0];

        // "ij->ji" — 2-D transpose
        if lhs == "ij" && rhs == "ji" {
            assert_eq!(a.tensor.ndim(), 2, "einsum ij->ji: requires 2-D input");
            return Ok(crate::ops::permute(a, &[1, 0]));
        }

        // generic last-two-dims swap (e.g. "bij->bji")
        if a.tensor.ndim() >= 2 {
            let chars: Vec<char> = lhs.chars().collect();
            let rhs_chars: Vec<char> = rhs.chars().collect();
            if chars.len() >= 2 && chars.len() == rhs_chars.len() {
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
            assert_eq!(a.tensor.ndim(), 2, "einsum ii->: requires 2-D input");
            let backend = B::default();
            let t = coeus_ops::einsum("ii->", &[&a.tensor], &backend).map_err(|source| {
                EinsumError::Backend {
                    source: Arc::new(source),
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
            subscript: subscript.to_owned(),
            operand_count: operands.len(),
        });
    }
    let a = operands[0];
    let b = operands[1];
    let a_lhs = lhs_parts[0];
    let b_lhs = lhs_parts[1];

    // "i,i->" — dot product (element-wise mul then sum)
    if a_lhs == "i" && b_lhs == "i" && rhs.is_empty() {
        let product = crate::ops::mul(a, b);
        return Ok(crate::ops::sum(&product));
    }

    // "i,j->ij" — outer product via broadcast + mul
    if a_lhs == "i" && b_lhs == "j" && rhs == "ij" {
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
        assert_eq!(a.tensor.ndim(), 2, "einsum ij,jk->ik: a must be 2-D");
        assert_eq!(b.tensor.ndim(), 2, "einsum ij,jk->ik: b must be 2-D");
        return Ok(crate::ops::matmul(a, b));
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
        assert_eq!(a.tensor.ndim(), 2, "einsum ij,j->i: a must be 2-D");
        assert_eq!(b.tensor.ndim(), 1, "einsum ij,j->i: b must be 1-D");
        let k = b.tensor.shape()[0];
        let b_col = crate::ops::reshape(b, vec![k, 1]);
        let mm = crate::ops::matmul(a, &b_col);
        return Ok(crate::ops::squeeze(&mm, Some(1)));
    }

    // Fallback: run the non-tracked op and return non-differentiable result.
    let backend = B::default();
    let raw_operands: Vec<&coeus_tensor::Tensor<T, B>> =
        operands.iter().map(|v| &v.tensor).collect();
    let out = coeus_ops::einsum(subscript, &raw_operands, &backend).map_err(|source| {
        EinsumError::Backend {
            source: Arc::new(source),
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
/// Returns [`EinsumError::UnsupportedPattern`] for any other subscript.
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


