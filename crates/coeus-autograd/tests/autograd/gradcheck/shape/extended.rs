use super::*;

fn diag_and_diagonal_case<T: GradcheckScalar>() {
    // diag scatters a vector onto an offset diagonal; diagonal gathers one back.
    // The offset is non-zero in both, so an implementation that ignored `k`
    // writes to the main diagonal and disagrees.
    let vector = tensor::<T>(&[3], 0.14);
    let w = weighting::<T>(&[4, 4]);
    gradcheck(&[vector], |v| {
        weighted(
            &diag(&v[0], 1).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("diag backward must match central differences");

    let matrix = tensor::<T>(&[4, 4], 0.16);
    let diag_w = weighting::<T>(&[3]);
    gradcheck(&[matrix], |v| {
        weighted(
            &diagonal(&v[0], 1).expect("invariant: test operation succeeds"),
            &diag_w,
        )
    })
    .expect("diagonal backward must match central differences");
}

#[test]
fn diag_and_diagonal_backward_match_finite_differences() {
    diag_and_diagonal_case::<f64>();
    diag_and_diagonal_case::<f32>();
}

fn diff_case<T: GradcheckScalar>() {
    // The first difference is a linear map whose transpose is a negated,
    // shifted accumulation; the boundary elements are where a sign or offset
    // error shows.
    let x = tensor::<T>(&[2, 5], 0.18);
    let w = weighting::<T>(&[2, 4]);
    gradcheck(&[x], |v| {
        weighted(
            &diff(&v[0], 1, 1).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("diff backward must match central differences");
}

#[test]
fn diff_backward_matches_finite_differences() {
    diff_case::<f64>();
    diff_case::<f32>();
}

fn contiguous_case<T: GradcheckScalar>() {
    // A materialising copy of a non-contiguous view: the gradient must be
    // scattered back through the original strides, not written densely.
    let x = tensor::<T>(&SHAPE, 0.22);
    let w = weighting::<T>(&[4, 3]);
    gradcheck(&[x], |v| {
        weighted(
            &contiguous(&transpose(&v[0], 0, 1).expect("invariant: test operation succeeds"))
                .expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("contiguous backward must match central differences");
}

#[test]
fn contiguous_backward_matches_finite_differences() {
    contiguous_case::<f64>();
    contiguous_case::<f32>();
}

fn composed_shape_chain_case<T: GradcheckScalar>() {
    // The individual checks above verify each backward in isolation. This one
    // composes four of them, so an index convention that each op applies
    // self-consistently but that disagrees between ops surfaces here and
    // nowhere else.
    let x = Sampler::signed(0.26).tensor::<T>(&[2, 6]);
    let w = weighting::<T>(&[3, 4]);
    gradcheck(&[x], |v| {
        let reshaped = reshape(&v[0], [3, 4]).expect("invariant: test operation succeeds");
        let rolled = roll(&reshaped, &[2], &[1]).expect("invariant: test operation succeeds");
        let flipped = flip(&rolled, 0).expect("invariant: test operation succeeds");
        weighted(
            &contiguous(&flipped).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("composed shape chain backward must match central differences");
}

#[test]
fn composed_shape_chain_backward_matches_finite_differences() {
    composed_shape_chain_case::<f64>();
    composed_shape_chain_case::<f32>();
}
