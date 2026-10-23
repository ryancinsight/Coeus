//! Subscript parsing and the typed rejections every einsum arm shares.
//!
//! Caller-supplied subscripts are untrusted input: each helper below turns a
//! mismatch into the matching [`BackendError`] variant, so no pattern arm has
//! to reach for a panic or an `unwrap`.

use crate::backend_ops::BackendOps;
use coeus_core::{BackendError, Scalar};

/// Parse and strip whitespace from an einsum subscript string.
pub(super) fn parse_subscript(subscript: &str) -> (Vec<&str>, &str) {
    let subscript = subscript.trim();
    let (lhs, rhs) = if let Some(pos) = subscript.find("->") {
        (&subscript[..pos], &subscript[pos + 2..])
    } else {
        (subscript, "")
    };
    let operands: Vec<&str> = lhs.split(',').map(str::trim).collect();
    (operands, rhs.trim())
}

/// Reject a subscript naming a different operand count than the call supplied.
pub(super) fn check_operand_count<T: Scalar, B: BackendOps<T>>(
    subscript: &str,
    subscript_operands: usize,
    provided: usize,
) -> Result<(), B::Error> {
    if subscript_operands == provided {
        return Ok(());
    }
    Err(B::Error::from(BackendError::EinsumOperandCount {
        subscript: subscript.to_string(),
        subscript_operands,
        provided,
    }))
}

/// Reject an operand whose rank disagrees with the labels its pattern names.
pub(super) fn check_rank<T: Scalar, B: BackendOps<T>>(
    subscript: &str,
    operand: usize,
    actual: usize,
    expected: usize,
) -> Result<(), B::Error> {
    if actual == expected {
        return Ok(());
    }
    Err(B::Error::from(BackendError::EinsumRank {
        subscript: subscript.to_string(),
        operand,
        expected,
        actual,
    }))
}

/// Reject a contracted extent that disagrees between the two operands.
pub(super) fn check_dim<T: Scalar, B: BackendOps<T>>(
    subscript: &str,
    operand: usize,
    actual: usize,
    expected: usize,
) -> Result<(), B::Error> {
    if actual == expected {
        return Ok(());
    }
    Err(B::Error::from(BackendError::EinsumDim {
        subscript: subscript.to_string(),
        operand,
        expected,
        actual,
    }))
}

/// The typed rejection for a subscript this module does not recognise.
pub(super) fn unsupported<T: Scalar, B: BackendOps<T>>(
    subscript: &str,
    operand_count: usize,
) -> B::Error {
    B::Error::from(BackendError::EinsumUnsupportedPattern {
        subscript: subscript.to_string(),
        operand_count,
    })
}
