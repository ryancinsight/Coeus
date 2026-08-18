//! Finite-difference checks for the shape and indexing backward passes.
//!
//! These ops are linear, which makes them look untestable — a linear map's
//! gradient is just its transpose, and the forward code and the backward code
//! usually sit next to each other. What they actually encode is index
//! arithmetic: a stride, an offset, a reversed axis, a wrapped shift. A
//! transposed permutation, an off-by-one pad offset or a shift applied in the
//! wrong direction produces a gradient of exactly the right shape delivered to
//! exactly the wrong elements, and no shape assertion catches it.
//!
//! Every check reduces through the non-uniform weighting from the parent
//! module, so the loss is sensitive to *which* element a gradient lands on.
//! Under a uniform `sum` most of these ops are loss-invariant and any
//! permutation of the gradient would pass.

use super::{tensor, weighted, weighting, Sampler, T64};
use coeus_autograd::{
    broadcast_to, cat, contiguous, diag, diagonal, diff, flip, gradcheck, index_select,
    masked_fill, pad, permute, reshape, roll, scatter_add, slice, split, squeeze, stack, tile,
    transpose, tril, triu, unsqueeze, where_cond, Var,
};
use coeus_core::MoiraiBackend;

/// Shape shared by most of the checks.
const SHAPE: [usize; 2] = [3, 4];

#[test]
fn reshape_backward_matches_finite_differences() {
    let x = tensor(&SHAPE, 0.11);
    let w = weighting(&[2, 6]);
    gradcheck(&[x], |v| weighted(&reshape(&v[0], [2, 6]), &w))
        .expect("reshape backward must match central differences");
}

#[test]
fn permute_backward_matches_finite_differences() {
    // The backward is the inverse permutation, not the same one; the two
    // coincide only for involutions, and [2,0,1] is deliberately not one.
    let x = tensor(&[2, 3, 4], 0.13);
    let w = weighting(&[4, 2, 3]);
    gradcheck(&[x], |v| weighted(&permute(&v[0], &[2, 0, 1]), &w))
        .expect("permute backward must match central differences");
}

#[test]
fn transpose_backward_matches_finite_differences() {
    let x = tensor(&SHAPE, 0.17);
    let w = weighting(&[4, 3]);
    gradcheck(&[x], |v| weighted(&transpose(&v[0], 0, 1), &w))
        .expect("transpose backward must match central differences");
}

#[test]
fn slice_backward_matches_finite_differences() {
    // Gradient must land at the slice's offset inside a zero-filled tensor; an
    // implementation that wrote it at the origin passes any shape check.
    let x = tensor(&SHAPE, 0.19);
    let w = weighting(&[2, 2]);
    gradcheck(&[x], |v| weighted(&slice(&v[0], &[(1, 3), (1, 3)]), &w))
        .expect("slice backward must match central differences");
}

#[test]
fn pad_backward_matches_finite_differences() {
    // The inverse of slice: the backward must drop the padded border and keep
    // the interior, offset by the same amount the forward inserted.
    let x = tensor(&SHAPE, 0.23);
    let w = weighting(&[5, 7]);
    gradcheck(&[x], |v| weighted(&pad(&v[0], &[(1, 1), (2, 1)], 0.0), &w))
        .expect("pad backward must match central differences");
}

#[test]
fn squeeze_and_unsqueeze_backward_match_finite_differences() {
    let x = tensor(&[3, 1, 4], 0.29);
    let w = weighting(&[3, 4]);
    gradcheck(std::slice::from_ref(&x), |v| {
        weighted(&squeeze(&v[0], Some(1)), &w)
    })
    .expect("squeeze backward must match central differences");

    let flat = tensor(&[3, 4], 0.31);
    let wide = weighting(&[3, 1, 4]);
    gradcheck(&[flat], |v| weighted(&unsqueeze(&v[0], 1), &wide))
        .expect("unsqueeze backward must match central differences");
}

#[test]
fn flip_backward_matches_finite_differences() {
    // Flip is an involution, so a backward that forgot to flip agrees with one
    // that flipped twice — only a non-uniform weighting separates them.
    let x = tensor(&SHAPE, 0.37);
    let w = weighting(&SHAPE);
    gradcheck(&[x], |v| weighted(&flip(&v[0], 1), &w))
        .expect("flip backward must match central differences");
}

#[test]
fn roll_backward_matches_finite_differences() {
    // The backward rolls by the negated shift. A sign error here is invisible
    // under a uniform reduction and invisible for a shift of half the extent;
    // the shift below is neither.
    let x = tensor(&SHAPE, 0.41);
    let w = weighting(&SHAPE);
    gradcheck(&[x], |v| weighted(&roll(&v[0], &[1], &[1]), &w))
        .expect("roll backward must match central differences");
}

#[test]
fn tile_backward_matches_finite_differences() {
    // Each source element appears in several output positions, so the backward
    // must *accumulate* across the repetitions rather than take the last one.
    let x = tensor(&[2, 3], 0.43);
    let w = weighting(&[4, 3]);
    gradcheck(&[x], |v| weighted(&tile(&v[0], &[2, 1]), &w))
        .expect("tile backward must match central differences");
}

#[test]
fn broadcast_to_backward_matches_finite_differences() {
    // The dual of tile: the backward sums over the broadcast axis. Summing over
    // the wrong axis, or not summing at all, is the usual failure.
    let x = tensor(&[1, 4], 0.47);
    let w = weighting(&[3, 4]);
    gradcheck(&[x], |v| weighted(&broadcast_to(&v[0], vec![3, 4]), &w))
        .expect("broadcast_to backward must match central differences");
}

#[test]
fn tril_and_triu_backward_match_finite_differences() {
    // A mask, so the backward is the same mask. The complementary triangle must
    // receive exactly zero; the two checks together confirm the diagonal offset
    // is applied consistently in both directions.
    let x = tensor(&[4, 4], 0.53);
    let w = weighting(&[4, 4]);
    gradcheck(std::slice::from_ref(&x), |v| weighted(&tril(&v[0], 0), &w))
        .expect("tril backward must match central differences");
    gradcheck(&[x], |v| weighted(&triu(&v[0], 1), &w))
        .expect("triu backward must match central differences");
}

#[test]
fn cat_backward_matches_finite_differences() {
    // The backward splits the output gradient back at the concatenation
    // boundary. The two operands have different extents along the joined axis,
    // so an off-by-one split boundary misroutes gradient rather than merely
    // reordering it.
    let a = tensor(&[2, 3], 0.59);
    let b = tensor(&[2, 5], 0.61);
    let w = weighting(&[2, 8]);
    gradcheck(&[a, b], |v| weighted(&cat(&[&v[0], &v[1]], 1), &w))
        .expect("cat backward must match central differences");
}

#[test]
fn stack_backward_matches_finite_differences() {
    let a = tensor(&[2, 3], 0.67);
    let b = tensor(&[2, 3], 0.71);
    let w = weighting(&[2, 2, 3]);
    gradcheck(&[a, b], |v| weighted(&stack(&[&v[0], &v[1]], 1), &w))
        .expect("stack backward must match central differences");
}

#[test]
fn split_backward_matches_finite_differences() {
    // Only the second chunk is reduced into the loss, so the first chunk's
    // gradient must be exactly zero and the second's must land at the right
    // offset — a split backward that wrote every chunk to the origin passes a
    // check that consumes all of them and fails this one.
    let x = tensor(&[2, 6], 0.73);
    let w = weighting(&[2, 3]);
    gradcheck(&[x], |v| weighted(&split(&v[0], 3, 1)[1], &w))
        .expect("split backward must match central differences");
}

#[test]
fn index_select_backward_matches_finite_differences() {
    // Row 1 is selected twice, so its gradient must accumulate; row 2 is never
    // selected and must stay at zero.
    let backend = MoiraiBackend::new();
    let x = tensor(&SHAPE, 0.79);
    let index = Var::new(T64::from_slice_on([3], &[0.0, 1.0, 1.0], &backend), false);
    let w = weighting(&[3, 4]);
    gradcheck(&[x], |v| weighted(&index_select(&v[0], 0, &index), &w))
        .expect("index_select backward must match central differences");
}

#[test]
fn scatter_add_backward_matches_finite_differences() {
    // Two gradients to check at once: the destination passes its gradient
    // through unchanged, while the source gathers from the scattered positions.
    // Both are differentiated so a rule swapped between them cannot hide.
    let backend = MoiraiBackend::new();
    let base = tensor(&SHAPE, 0.83);
    let src = tensor(&[3, 2], 0.89);
    let index = Var::new(
        T64::from_slice_on([3, 2], &[0.0, 2.0, 1.0, 1.0, 3.0, 0.0], &backend),
        false,
    );
    let w = weighting(&SHAPE);
    gradcheck(&[base, src], |v| {
        weighted(&scatter_add(&v[0], 1, &index, &v[1]), &w)
    })
    .expect("scatter_add backward must match central differences");
}

#[test]
fn masked_fill_backward_matches_finite_differences() {
    // Filled positions are replaced by a constant, so they must receive exactly
    // zero gradient while the rest pass through. The mask is irregular, so a
    // backward that inverted it fails rather than coincidentally agreeing.
    let backend = MoiraiBackend::new();
    let x = tensor(&[2, 4], 0.91);
    let mask = Var::new(
        T64::from_slice_on([2, 4], &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], &backend),
        false,
    );
    let w = weighting(&[2, 4]);
    gradcheck(&[x], |v| weighted(&masked_fill(&v[0], &mask, 0.0), &w))
        .expect("masked_fill backward must match central differences");
}

#[test]
fn where_cond_backward_matches_finite_differences() {
    // Each branch receives gradient only where it was selected. Differentiating
    // both branches together catches a backward that routed the whole gradient
    // to one of them.
    let backend = MoiraiBackend::new();
    let cond = Var::new(
        T64::from_slice_on([2, 4], &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], &backend),
        false,
    );
    let on_true = tensor(&[2, 4], 0.12);
    let on_false = tensor(&[2, 4], 0.34);
    let w = weighting(&[2, 4]);
    gradcheck(&[on_true, on_false], |v| {
        weighted(&where_cond(&cond, &v[0], &v[1]), &w)
    })
    .expect("where_cond backward must match central differences");
}

#[test]
fn diag_and_diagonal_backward_match_finite_differences() {
    // diag scatters a vector onto an offset diagonal; diagonal gathers one back.
    // The offset is non-zero in both, so an implementation that ignored `k`
    // writes to the main diagonal and disagrees.
    let vector = tensor(&[3], 0.14);
    let w = weighting(&[4, 4]);
    gradcheck(&[vector], |v| weighted(&diag(&v[0], 1), &w))
        .expect("diag backward must match central differences");

    let matrix = tensor(&[4, 4], 0.16);
    let diag_w = weighting(&[3]);
    gradcheck(&[matrix], |v| weighted(&diagonal(&v[0], 1), &diag_w))
        .expect("diagonal backward must match central differences");
}

#[test]
fn diff_backward_matches_finite_differences() {
    // The first difference is a linear map whose transpose is a negated,
    // shifted accumulation; the boundary elements are where a sign or offset
    // error shows.
    let x = tensor(&[2, 5], 0.18);
    let w = weighting(&[2, 4]);
    gradcheck(&[x], |v| weighted(&diff(&v[0], 1, 1), &w))
        .expect("diff backward must match central differences");
}

#[test]
fn contiguous_backward_matches_finite_differences() {
    // A materialising copy of a non-contiguous view: the gradient must be
    // scattered back through the original strides, not written densely.
    let x = tensor(&SHAPE, 0.22);
    let w = weighting(&[4, 3]);
    gradcheck(&[x], |v| weighted(&contiguous(&transpose(&v[0], 0, 1)), &w))
        .expect("contiguous backward must match central differences");
}

#[test]
fn composed_shape_chain_backward_matches_finite_differences() {
    // The individual checks above verify each backward in isolation. This one
    // composes four of them, so an index convention that each op applies
    // self-consistently but that disagrees between ops surfaces here and
    // nowhere else.
    let x = Sampler::signed(0.26).tensor(&[2, 6]);
    let w = weighting(&[3, 4]);
    gradcheck(&[x], |v| {
        let reshaped = reshape(&v[0], [3, 4]);
        let rolled = roll(&reshaped, &[2], &[1]);
        let flipped = flip(&rolled, 0);
        weighted(&contiguous(&flipped), &w)
    })
    .expect("composed shape chain backward must match central differences");
}
