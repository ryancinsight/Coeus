//! Finite-difference checks for attention and the softmax family.
//!
//! `sdp_attention` is the highest-risk backward in the crate. It is fused: one
//! `scaled_dot_product_attention_backward` call produces `dQ`, `dK` and `dV`
//! from the saved attention weights, so the three rules
//!
//! ```text
//! dV = Aᵀ·dO
//! dA = dO·Vᵀ                     (then through the softmax Jacobian)
//! dS = A ⊙ (dA − rowsum(dA ⊙ A))
//! dQ = dS·K·scale,  dK = dSᵀ·Q·scale
//! ```
//!
//! share one implementation and one set of transposes. A swapped operand or a
//! dropped `scale` in any of them still produces correctly-shaped gradients
//! that descend, roughly, in the right direction. Nothing short of comparing
//! against the real forward distinguishes them, which is what these checks do —
//! and they differentiate Q, K and V together so a transposition that cancels
//! between two of the rules cannot hide.

use super::{tensor, weighted, weighting, Sampler, T64};
use coeus_autograd::{
    causal_softmax, gradcheck, log_softmax, masked_softmax, sdp_attention, softmin, CausalMask,
    NullMask,
};
use coeus_core::MoiraiBackend;

/// Batch, query/key length and head dimension of the attention fixtures.
///
/// Deliberately non-square: `SEQ != HEAD_DIM` and both differ from `BATCH`, so
/// a transposed operand produces a shape error or a wrong number rather than
/// silently working on a square fixture that hides the mistake.
const BATCH: usize = 2;
/// Sequence length; see [`BATCH`].
const SEQ: usize = 3;
/// Head dimension; see [`BATCH`].
const HEAD_DIM: usize = 4;

/// `1/sqrt(head_dim)`, the standard attention scale.
///
/// Passed explicitly rather than defaulted so that a backward which forgot to
/// apply it to `dQ` and `dK` disagrees with the finite difference by exactly
/// this factor.
const SCALE: f64 = 0.5;

#[test]
fn sdp_attention_backward_matches_finite_differences() {
    // Unmasked attention, all three operands differentiated together.
    let q = tensor(&[BATCH, SEQ, HEAD_DIM], 0.17);
    let k = tensor(&[BATCH, SEQ, HEAD_DIM], 0.43);
    let v = tensor(&[BATCH, SEQ, HEAD_DIM], 0.71);
    let w = weighting(&[BATCH, SEQ, HEAD_DIM]);

    gradcheck(&[q, k, v], |vars| {
        let (output, _weights) = sdp_attention::<f64, MoiraiBackend, NullMask>(
            &vars[0], &vars[1], &vars[2], None, SCALE,
        )
        .expect("attention operand shapes agree");
        weighted(&output, &w)
    })
    .expect("sdp_attention backward must match central differences");
}

#[test]
fn causal_sdp_attention_backward_matches_finite_differences() {
    // The causal mask is applied before the softmax, so the masked positions
    // must receive exactly zero gradient while the unmasked ones carry the full
    // rule. A backward that applied the mask after the softmax Jacobian — or
    // not at all on the reverse pass — leaks gradient into the upper triangle,
    // and the finite difference of the genuinely masked forward reports zero
    // there.
    let q = tensor(&[BATCH, SEQ, HEAD_DIM], 0.29);
    let k = tensor(&[BATCH, SEQ, HEAD_DIM], 0.61);
    let v = tensor(&[BATCH, SEQ, HEAD_DIM], 0.13);
    let w = weighting(&[BATCH, SEQ, HEAD_DIM]);

    gradcheck(&[q, k, v], |vars| {
        let (output, _weights) = sdp_attention::<f64, MoiraiBackend, CausalMask>(
            &vars[0], &vars[1], &vars[2], None, SCALE,
        )
        .expect("attention operand shapes agree");
        weighted(&output, &w)
    })
    .expect("causal sdp_attention backward must match central differences");
}

#[test]
fn log_softmax_backward_matches_finite_differences() {
    // d/dx log_softmax(x) = I - softmax(x)·1ᵀ, so the gradient is
    // dy - softmax(x)·Σdy. Unlike softmax's, this Jacobian's rows do *not* sum
    // to zero, so the common error of copying the softmax backward into
    // log_softmax produces a gradient that is wrong by softmax(x)·Σdy — an
    // O(1) discrepancy this check reports per element.
    let x = tensor(&[3, 5], 0.37);
    let w = weighting(&[3, 5]);

    gradcheck(&[x], |v| weighted(&log_softmax(&v[0], 1), &w))
        .expect("log_softmax backward must match central differences");
}

#[test]
fn softmin_backward_matches_finite_differences() {
    // softmin is softmax of the negation, so its gradient must carry the extra
    // -1 from the chain rule. A softmin that reused the softmax node without
    // negating differs from the finite difference by exactly a sign.
    let x = tensor(&[2, 4], 0.53);
    let w = weighting(&[2, 4]);

    gradcheck(&[x], |v| weighted(&softmin(&v[0], 1), &w))
        .expect("softmin backward must match central differences");
}

#[test]
fn causal_softmax_backward_matches_finite_differences() {
    // A square score matrix, so the lower-triangular mask is well defined. The
    // masked entries are excluded from the forward's normalisation, and their
    // gradient must be exactly zero rather than merely small.
    let x = tensor(&[4, 4], 0.23);
    let w = weighting(&[4, 4]);

    gradcheck(&[x], |v| weighted(&causal_softmax(&v[0], 1), &w))
        .expect("causal_softmax backward must match central differences");
}

#[test]
fn masked_softmax_backward_matches_finite_differences() {
    // An explicit irregular mask, rather than the triangular one: entries 1 and
    // 4 of the first row and entry 2 of the second are masked out. The pattern
    // is deliberately not a triangle, so a backward that hard-coded the causal
    // shape instead of reading the mask disagrees.
    let backend = MoiraiBackend::new();
    let x = tensor(&[2, 5], 0.79);
    let mask = T64::from_slice_on(
        [2, 5],
        &[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        &backend,
    );
    let w = weighting(&[2, 5]);

    gradcheck(&[x], |v| weighted(&masked_softmax(&v[0], &mask, 1), &w))
        .expect("masked_softmax backward must match central differences");
}

#[test]
fn sdp_attention_backward_matches_finite_differences_on_wide_values() {
    // A value tensor whose head dimension differs from the key's exercises the
    // dV = Aᵀ·dO rule on a non-square contraction, where a transposed operand
    // cannot be absorbed by symmetry. Q and K keep HEAD_DIM; V is sampled from a
    // separate phase so the three operands are not translates of each other.
    let q = Sampler::signed(0.11).tensor(&[BATCH, SEQ, HEAD_DIM]);
    let k = Sampler::signed(0.47).tensor(&[BATCH, SEQ, HEAD_DIM]);
    let v = Sampler::signed(0.89).tensor(&[BATCH, SEQ, HEAD_DIM]);
    let w = weighting(&[BATCH, SEQ, HEAD_DIM]);

    gradcheck(&[q, k, v], |vars| {
        let (output, _weights) = sdp_attention::<f64, MoiraiBackend, NullMask>(
            &vars[0], &vars[1], &vars[2], None, SCALE,
        )
        .expect("attention operand shapes agree");
        weighted(&output, &w)
    })
    .expect("sdp_attention backward must match central differences");
}
