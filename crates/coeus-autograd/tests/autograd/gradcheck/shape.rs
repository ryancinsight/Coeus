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

use super::{tensor, weighted, weighting, GradcheckScalar, Sampler};
use coeus_autograd::{
    broadcast_to, cat, contiguous, diag, diagonal, diff, flip, gradcheck, index_select,
    masked_fill, pad, permute, reshape, roll, scatter_add, slice, split, squeeze, stack, tile,
    transpose, tril, triu, unsqueeze, where_cond, Var,
};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

/// Shape shared by most of the checks.
const SHAPE: [usize; 2] = [3, 4];

fn reshape_case<T: GradcheckScalar>() {
    let x = tensor::<T>(&SHAPE, 0.11);
    let w = weighting::<T>(&[2, 6]);
    gradcheck(&[x], |v| {
        weighted(
            &reshape(&v[0], [2, 6]).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("reshape backward must match central differences");
}

#[test]
fn reshape_backward_matches_finite_differences() {
    reshape_case::<f64>();
    reshape_case::<f32>();
}

fn permute_case<T: GradcheckScalar>() {
    // The backward is the inverse permutation, not the same one; the two
    // coincide only for involutions, and [2,0,1] is deliberately not one.
    let x = tensor::<T>(&[2, 3, 4], 0.13);
    let w = weighting::<T>(&[4, 2, 3]);
    gradcheck(&[x], |v| {
        weighted(
            &permute(&v[0], &[2, 0, 1]).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("permute backward must match central differences");
}

#[test]
fn permute_backward_matches_finite_differences() {
    permute_case::<f64>();
    permute_case::<f32>();
}

fn transpose_case<T: GradcheckScalar>() {
    let x = tensor::<T>(&SHAPE, 0.17);
    let w = weighting::<T>(&[4, 3]);
    gradcheck(&[x], |v| {
        weighted(
            &transpose(&v[0], 0, 1).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("transpose backward must match central differences");
}

#[test]
fn transpose_backward_matches_finite_differences() {
    transpose_case::<f64>();
    transpose_case::<f32>();
}

fn slice_case<T: GradcheckScalar>() {
    // Gradient must land at the slice's offset inside a zero-filled tensor; an
    // implementation that wrote it at the origin passes any shape check.
    let x = tensor::<T>(&SHAPE, 0.19);
    let w = weighting::<T>(&[2, 2]);
    gradcheck(&[x], |v| {
        weighted(
            &slice(&v[0], &[(1, 3), (1, 3)]).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("slice backward must match central differences");
}

#[test]
fn slice_backward_matches_finite_differences() {
    slice_case::<f64>();
    slice_case::<f32>();
}

fn pad_case<T: GradcheckScalar>() {
    // The inverse of slice: the backward must drop the padded border and keep
    // the interior, offset by the same amount the forward inserted.
    let x = tensor::<T>(&SHAPE, 0.23);
    let w = weighting::<T>(&[5, 7]);
    gradcheck(&[x], |v| {
        weighted(
            &pad(
                &v[0],
                &[(1, 1), (2, 1)],
                <T as coeus_core::Scalar>::from_f64(0.0),
            )
            .expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("pad backward must match central differences");
}

#[test]
fn pad_backward_matches_finite_differences() {
    pad_case::<f64>();
    pad_case::<f32>();
}

fn squeeze_and_unsqueeze_case<T: GradcheckScalar>() {
    let x = tensor::<T>(&[3, 1, 4], 0.29);
    let w = weighting::<T>(&[3, 4]);
    gradcheck(std::slice::from_ref(&x), |v| {
        weighted(
            &squeeze(&v[0], Some(1)).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("squeeze backward must match central differences");

    let flat = tensor::<T>(&[3, 4], 0.31);
    let wide = weighting::<T>(&[3, 1, 4]);
    gradcheck(&[flat], |v| {
        weighted(
            &unsqueeze(&v[0], 1).expect("invariant: test operation succeeds"),
            &wide,
        )
    })
    .expect("unsqueeze backward must match central differences");
}

#[test]
fn squeeze_and_unsqueeze_backward_match_finite_differences() {
    squeeze_and_unsqueeze_case::<f64>();
    squeeze_and_unsqueeze_case::<f32>();
}

fn flip_case<T: GradcheckScalar>() {
    // Flip is an involution, so a backward that forgot to flip agrees with one
    // that flipped twice — only a non-uniform weighting separates them.
    let x = tensor::<T>(&SHAPE, 0.37);
    let w = weighting::<T>(&SHAPE);
    gradcheck(&[x], |v| {
        weighted(
            &flip(&v[0], 1).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("flip backward must match central differences");
}

#[test]
fn flip_backward_matches_finite_differences() {
    flip_case::<f64>();
    flip_case::<f32>();
}

fn roll_case<T: GradcheckScalar>() {
    // The backward rolls by the negated shift. A sign error here is invisible
    // under a uniform reduction and invisible for a shift of half the extent;
    // the shift below is neither.
    let x = tensor::<T>(&SHAPE, 0.41);
    let w = weighting::<T>(&SHAPE);
    gradcheck(&[x], |v| {
        weighted(
            &roll(&v[0], &[1], &[1]).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("roll backward must match central differences");
}

#[test]
fn roll_backward_matches_finite_differences() {
    roll_case::<f64>();
    roll_case::<f32>();
}

fn tile_case<T: GradcheckScalar>() {
    // Each source element appears in several output positions, so the backward
    // must *accumulate* across the repetitions rather than take the last one.
    let x = tensor::<T>(&[2, 3], 0.43);
    let w = weighting::<T>(&[4, 3]);
    gradcheck(&[x], |v| {
        weighted(
            &tile(&v[0], &[2, 1]).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("tile backward must match central differences");
}

#[test]
fn tile_backward_matches_finite_differences() {
    tile_case::<f64>();
    tile_case::<f32>();
}

fn broadcast_to_case<T: GradcheckScalar>() {
    // The dual of tile: the backward sums over the broadcast axis. Summing over
    // the wrong axis, or not summing at all, is the usual failure.
    let x = tensor::<T>(&[1, 4], 0.47);
    let w = weighting::<T>(&[3, 4]);
    gradcheck(&[x], |v| {
        weighted(
            &broadcast_to(&v[0], vec![3, 4]).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("broadcast_to backward must match central differences");
}

#[test]
fn broadcast_to_backward_matches_finite_differences() {
    broadcast_to_case::<f64>();
    broadcast_to_case::<f32>();
}

fn tril_and_triu_case<T: GradcheckScalar>() {
    // A mask, so the backward is the same mask. The complementary triangle must
    // receive exactly zero; the two checks together confirm the diagonal offset
    // is applied consistently in both directions.
    let x = tensor::<T>(&[4, 4], 0.53);
    let w = weighting::<T>(&[4, 4]);
    gradcheck(std::slice::from_ref(&x), |v| {
        weighted(
            &tril(&v[0], 0).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("tril backward must match central differences");
    gradcheck(&[x], |v| {
        weighted(
            &triu(&v[0], 1).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("triu backward must match central differences");
}

#[test]
fn tril_and_triu_backward_match_finite_differences() {
    tril_and_triu_case::<f64>();
    tril_and_triu_case::<f32>();
}

fn cat_case<T: GradcheckScalar>() {
    // The backward splits the output gradient back at the concatenation
    // boundary. The two operands have different extents along the joined axis,
    // so an off-by-one split boundary misroutes gradient rather than merely
    // reordering it.
    let a = tensor::<T>(&[2, 3], 0.59);
    let b = tensor::<T>(&[2, 5], 0.61);
    let w = weighting::<T>(&[2, 8]);
    gradcheck(&[a, b], |v| {
        weighted(
            &cat(&[&v[0], &v[1]], 1).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("cat backward must match central differences");
}

#[test]
fn cat_backward_matches_finite_differences() {
    cat_case::<f64>();
    cat_case::<f32>();
}

fn stack_case<T: GradcheckScalar>() {
    let a = tensor::<T>(&[2, 3], 0.67);
    let b = tensor::<T>(&[2, 3], 0.71);
    let w = weighting::<T>(&[2, 2, 3]);
    gradcheck(&[a, b], |v| {
        weighted(
            &stack(&[&v[0], &v[1]], 1).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("stack backward must match central differences");
}

#[test]
fn stack_backward_matches_finite_differences() {
    stack_case::<f64>();
    stack_case::<f32>();
}

fn split_case<T: GradcheckScalar>() {
    // Only the second chunk is reduced into the loss, so the first chunk's
    // gradient must be exactly zero and the second's must land at the right
    // offset — a split backward that wrote every chunk to the origin passes a
    // check that consumes all of them and fails this one.
    let x = tensor::<T>(&[2, 6], 0.73);
    let w = weighting::<T>(&[2, 3]);
    gradcheck(&[x], |v| {
        weighted(
            &split(&v[0], 3, 1).expect("invariant: test operation succeeds")[1],
            &w,
        )
    })
    .expect("split backward must match central differences");
}

#[test]
fn split_backward_matches_finite_differences() {
    split_case::<f64>();
    split_case::<f32>();
}

fn index_select_case<T: GradcheckScalar>() {
    // Row 1 is selected twice, so its gradient must accumulate; row 2 is never
    // selected and must stay at zero.
    let backend = MoiraiBackend::new();
    let x = tensor::<T>(&SHAPE, 0.79);
    let index_values: Vec<T> = [0.0, 1.0, 1.0]
        .into_iter()
        .map(<T as coeus_core::Scalar>::from_f64)
        .collect();
    let index = Var::new(
        Tensor::<T, MoiraiBackend>::from_slice_on([3], &index_values, &backend)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let w = weighting::<T>(&[3, 4]);
    gradcheck(&[x], |v| {
        weighted(
            &index_select(&v[0], 0, &index).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("index_select backward must match central differences");
}

#[test]
fn index_select_backward_matches_finite_differences() {
    index_select_case::<f64>();
    index_select_case::<f32>();
}

fn scatter_add_case<T: GradcheckScalar>() {
    // Two gradients to check at once: the destination passes its gradient
    // through unchanged, while the source gathers from the scattered positions.
    // Both are differentiated so a rule swapped between them cannot hide.
    let backend = MoiraiBackend::new();
    let base = tensor::<T>(&SHAPE, 0.83);
    let src = tensor::<T>(&[3, 2], 0.89);
    let index_values: Vec<T> = [0.0, 2.0, 1.0, 1.0, 3.0, 0.0]
        .into_iter()
        .map(<T as coeus_core::Scalar>::from_f64)
        .collect();
    let index = Var::new(
        Tensor::<T, MoiraiBackend>::from_slice_on([3, 2], &index_values, &backend)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let w = weighting::<T>(&SHAPE);
    gradcheck(&[base, src], |v| {
        weighted(
            &scatter_add(&v[0], 1, &index, &v[1]).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("scatter_add backward must match central differences");
}

#[test]
fn scatter_add_backward_matches_finite_differences() {
    scatter_add_case::<f64>();
    scatter_add_case::<f32>();
}

fn masked_fill_case<T: GradcheckScalar>() {
    // Filled positions are replaced by a constant, so they must receive exactly
    // zero gradient while the rest pass through. The mask is irregular, so a
    // backward that inverted it fails rather than coincidentally agreeing.
    let backend = MoiraiBackend::new();
    let x = tensor::<T>(&[2, 4], 0.91);
    let mask_values: Vec<T> = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        .into_iter()
        .map(<T as coeus_core::Scalar>::from_f64)
        .collect();
    let mask = Var::new(
        Tensor::<T, MoiraiBackend>::from_slice_on([2, 4], &mask_values, &backend)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let w = weighting::<T>(&[2, 4]);
    gradcheck(&[x], |v| {
        weighted(
            &masked_fill(&v[0], &mask, <T as coeus_core::Scalar>::from_f64(0.0))
                .expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("masked_fill backward must match central differences");
}

#[test]
fn masked_fill_backward_matches_finite_differences() {
    masked_fill_case::<f64>();
    masked_fill_case::<f32>();
}

fn where_cond_case<T: GradcheckScalar>() {
    // Each branch receives gradient only where it was selected. Differentiating
    // both branches together catches a backward that routed the whole gradient
    // to one of them.
    let backend = MoiraiBackend::new();
    let cond_values: Vec<T> = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        .into_iter()
        .map(<T as coeus_core::Scalar>::from_f64)
        .collect();
    let cond = Var::new(
        Tensor::<T, MoiraiBackend>::from_slice_on([2, 4], &cond_values, &backend)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let on_true = tensor::<T>(&[2, 4], 0.12);
    let on_false = tensor::<T>(&[2, 4], 0.34);
    let w = weighting::<T>(&[2, 4]);
    gradcheck(&[on_true, on_false], |v| {
        weighted(
            &where_cond(&cond, &v[0], &v[1]).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("where_cond backward must match central differences");
}

#[test]
fn where_cond_backward_matches_finite_differences() {
    where_cond_case::<f64>();
    where_cond_case::<f32>();
}

#[path = "shape/extended.rs"]
mod extended;
