use coeus_autograd::{matmul, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

/// Batched matmul (bmm) backward: with an all-ones seed, the analytic
/// gradients are `grad_A = 1 @ Bᵀ` (every row of `grad_A[b]` equals the
/// per-row sums of `B[b]`) and `grad_B = Aᵀ @ 1` (every column of
/// `grad_B[b]` equals the per-column sums of `A[b]`). Regression test for
/// the backward panicking on batched B ("transpose requires 2D tensor").
#[test]
fn test_bmm_backward_accumulates_exact_gradients() {
    let backend = MoiraiBackend::new();
    // Batch 0: A0 = [[1,2,3],[4,5,6]], B0 = [[1,0],[0,1],[1,1]]
    // Batch 1: A1 = [[0,1,0],[2,0,2]], B1 = [[2,1],[1,2],[0,3]]
    let a = Var::new(
        Tensor::from_slice_on(
            vec![2, 2, 3],
            &[
                1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0,
            ],
            &backend,
        ),
        true,
    );
    let b = Var::new(
        Tensor::from_slice_on(
            vec![2, 3, 2],
            &[
                1.0f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.0, 3.0,
            ],
            &backend,
        ),
        true,
    );

    let c = matmul(&a, &b);
    assert_eq!(c.tensor.shape(), &[2, 2, 2]);
    // C0 = [[4,5],[10,11]]; C1 = [[1,2],[4,8]].
    let cs = c.tensor.as_slice();
    let expected_fwd = [4.0, 5.0, 10.0, 11.0, 1.0, 2.0, 4.0, 8.0];
    for (got, want) in cs.iter().zip(expected_fwd.iter()) {
        assert!((got - want).abs() < 1e-14, "fwd {got} vs {want}");
    }

    c.backward();

    // grad_A[b] rows = row sums of B[b]: B0 rows sum to [1,1,2]; B1 to [3,3,3].
    let ga = a.grad().unwrap();
    let ga = ga.as_slice();
    let expected_ga = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0];
    for (i, (got, want)) in ga.iter().zip(expected_ga.iter()).enumerate() {
        assert!((got - want).abs() < 1e-14, "grad_a[{i}] {got} vs {want}");
    }

    // grad_B[b] columns = column sums of A[b]: A0 cols [5,7,9]; A1 cols [2,1,2].
    let gb = b.grad().unwrap();
    let gb = gb.as_slice();
    let expected_gb = [5.0, 5.0, 7.0, 7.0, 9.0, 9.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];
    for (i, (got, want)) in gb.iter().zip(expected_gb.iter()).enumerate() {
        assert!((got - want).abs() < 1e-14, "grad_b[{i}] {got} vs {want}");
    }
}

#[test]
fn rank_four_batched_matmul_preserves_axes_and_exact_gradients() {
    let backend = MoiraiBackend::new();
    let a = Var::new(
        Tensor::from_slice_on(
            [1, 2, 2, 3],
            &[
                1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0,
            ],
            &backend,
        ),
        true,
    );
    let b = Var::new(
        Tensor::from_slice_on(
            [1, 2, 3, 2],
            &[
                1.0f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.0, 3.0,
            ],
            &backend,
        ),
        true,
    );

    let output = matmul(&a, &b);
    assert_eq!(output.tensor.shape(), &[1, 2, 2, 2]);
    assert_eq!(
        output.tensor.as_slice(),
        &[4.0, 5.0, 10.0, 11.0, 1.0, 2.0, 4.0, 8.0]
    );

    output.backward();
    assert_eq!(
        a.grad().expect("A gradient must be populated").as_slice(),
        &[1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    );
    assert_eq!(
        b.grad().expect("B gradient must be populated").as_slice(),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]
    );
}

/// A non-contiguous input (a `transpose` view fed straight into `matmul`, as
/// attention's `Q Kᵀ` does) must produce the same forward product and the same
/// gradients as the explicitly-materialized-contiguous path. Regression test
/// for the batched matmul kernel reading view inputs with shape-derived strides
/// and thus computing a wrong product (and a gradient inconsistent with it).
#[test]
fn matmul_view_input_matches_contiguous_forward_and_grad() {
    use coeus_autograd::{contiguous, sum, transpose};

    let backend = MoiraiBackend::new();
    let shape = [1usize, 2, 3, 4];
    let n: usize = shape.iter().product();
    let qd: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let kd: Vec<f32> = (0..n).map(|i| 0.5 - (i as f32) * 0.07).collect();

    // View path: transpose fed straight into matmul.
    let q1 = Var::new(Tensor::from_slice_on(shape, &qd, &backend), true);
    let k1 = Var::new(Tensor::from_slice_on(shape, &kd, &backend), true);
    let out_view = matmul(&q1, &transpose(&k1, 2, 3));
    sum(&out_view).backward();

    // Reference path: materialize the transpose contiguous first.
    let q2 = Var::new(Tensor::from_slice_on(shape, &qd, &backend), true);
    let k2 = Var::new(Tensor::from_slice_on(shape, &kd, &backend), true);
    let out_ref = matmul(&q2, &contiguous(&transpose(&k2, 2, 3)));
    sum(&out_ref).backward();

    assert_eq!(
        out_view.tensor.as_slice(),
        out_ref.tensor.as_slice(),
        "forward product must match the contiguous reference"
    );
    assert_eq!(
        q1.grad().expect("q grad").as_slice(),
        q2.grad().expect("q grad ref").as_slice(),
        "grad wrt q must match the contiguous reference"
    );
    assert_eq!(
        k1.grad().expect("k grad").as_slice(),
        k2.grad().expect("k grad ref").as_slice(),
        "grad wrt k must match the contiguous reference"
    );
}
