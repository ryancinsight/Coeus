use super::*;

#[test]
fn test_cosine_similarity_forward_and_backward() {
    // [N=2, D=2]; row0 = (3,4)·(4,3) / (5·5) = 24/25 = 0.96;
    // row1 = (1,0)·(0,1) / (1·1) = 0.  eps is negligible at 1e-12.
    let x1_data = vec![3.0_f64, 4.0, 1.0, 0.0];
    let x2_data = vec![4.0_f64, 3.0, 0.0, 1.0];
    let x1 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &x1_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let x2 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &x2_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let out = cosine_similarity(&x1, &x2, 1, 1e-12).expect("invariant: test operation succeeds");
    assert_eq!(out.tensor.shape(), &[2]);
    let s = out.tensor.as_slice();
    assert!((s[0] - 0.96).abs() < 1e-9, "row0 cos: got {}", s[0]);
    assert!(s[1].abs() < 1e-9, "row1 cos: got {}", s[1]);

    // Backward against a central finite-difference reference on sum(cos).
    out.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let analytic: Vec<f64> = x1.grad().expect("cosine x1 grad").as_slice().to_vec();
    let h = 1e-6;
    let forward_sum = |d: &[f64]| -> f64 {
        let xv = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([2, 2], d)
                .expect("invariant: test backend operation succeeds"),
            false,
        )
        .expect("invariant: test backend operation succeeds");
        cosine_similarity(&xv, &x2, 1, 1e-12)
            .expect("invariant: test operation succeeds")
            .tensor
            .as_slice()
            .iter()
            .sum::<f64>()
    };
    for i in 0..x1_data.len() {
        let mut dp = x1_data.clone();
        dp[i] += h;
        let mut dm = x1_data.clone();
        dm[i] -= h;
        let numeric = (forward_sum(&dp) - forward_sum(&dm)) / (2.0 * h);
        assert!(
            (analytic[i] - numeric).abs() < 1e-5,
            "cosine dx1[{i}]: analytic {} vs numeric {}",
            analytic[i],
            numeric
        );
    }
    assert!(x2.grad().is_some(), "cosine x2 grad");
}

#[test]
fn test_cosine_similarity_clamp_region_gradients() {
    const EPS: f64 = 1.0;
    const STEP: f64 = 1.0e-6;
    // Central differences incur O(STEP^2) truncation plus O(epsilon / STEP)
    // rounding. For these unit-scale fixtures, 1e-8 bounds both terms with a
    // conservative factor of two.
    const TOLERANCE: f64 = 1.0e-8;

    #[derive(Clone, Copy)]
    enum DifferentiatedInput {
        First,
        Second,
    }

    let x2_data = [1.0_f64, 0.0];
    let finite_difference =
        |x1_data: [f64; 2], x2_data: [f64; 2], input: DifferentiatedInput, index: usize| {
            let evaluate = |first: [f64; 2], second: [f64; 2]| {
                let x1 = Var::new(
                    Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &first)
                        .expect("invariant: test backend operation succeeds"),
                    false,
                )
                .expect("invariant: test backend operation succeeds");
                let x2 = Var::new(
                    Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &second)
                        .expect("invariant: test backend operation succeeds"),
                    false,
                )
                .expect("invariant: test backend operation succeeds");
                cosine_similarity(&x1, &x2, 1, EPS)
                    .expect("invariant: test operation succeeds")
                    .tensor
                    .as_slice()[0]
            };
            let (mut plus_x1, mut minus_x1) = (x1_data, x1_data);
            let (mut plus_x2, mut minus_x2) = (x2_data, x2_data);
            match input {
                DifferentiatedInput::First => {
                    plus_x1[index] += STEP;
                    minus_x1[index] -= STEP;
                }
                DifferentiatedInput::Second => {
                    plus_x2[index] += STEP;
                    minus_x2[index] -= STEP;
                }
            }
            (evaluate(plus_x1, plus_x2) - evaluate(minus_x1, minus_x2)) / (2.0 * STEP)
        };

    for x1_data in [[0.0, 0.0], [0.5, 0.25], [2.0, 1.0]] {
        let x1 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &x1_data)
                .expect("invariant: test backend operation succeeds"),
            true,
        )
        .expect("invariant: test backend operation succeeds");
        let x2 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &x2_data)
                .expect("invariant: test backend operation succeeds"),
            true,
        )
        .expect("invariant: test backend operation succeeds");
        cosine_similarity(&x1, &x2, 1, EPS)
            .expect("invariant: test operation succeeds")
            .backward()
            .expect("invariant: valid clamp-region fixture completes backward");

        let x1_gradient = x1.grad().expect("tracked x1 gradient");
        let x2_gradient = x2.grad().expect("tracked x2 gradient");
        for (name, input, gradient) in [
            ("x1", DifferentiatedInput::First, &x1_gradient),
            ("x2", DifferentiatedInput::Second, &x2_gradient),
        ] {
            for index in 0..2 {
                let analytic = gradient.as_slice()[index];
                let numeric = finite_difference(x1_data, x2_data, input, index);
                assert!(
                    analytic.is_finite(),
                    "{name} gradient {index} must be finite"
                );
                assert!(
                    (analytic - numeric).abs() <= TOLERANCE,
                    "{name} gradient {index}: analytic {analytic}, numeric {numeric}"
                );
            }
        }
    }

    // The max operation is non-differentiable at norm_product == eps. Coeus
    // follows the existing inclusive clamp convention and retains the norm
    // derivative at equality; for collinear unit vectors this derivative is 0.
    let boundary_x1 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0, 0.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let boundary_x2 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &x2_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    cosine_similarity(&boundary_x1, &boundary_x2, 1, EPS)
        .expect("invariant: test operation succeeds")
        .backward()
        .expect("invariant: boundary fixture completes backward");
    let boundary_gradient = boundary_x1.grad().expect("tracked boundary gradient");
    assert_eq!(boundary_gradient.as_slice(), &[0.0, 0.0]);
}

#[test]
#[should_panic(expected = "cosine_similarity requires finite eps > 0")]
fn test_cosine_similarity_rejects_zero_epsilon() {
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 1], &[1.0])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let _ = cosine_similarity(&x, &x, 1, 0.0).expect("invariant: test operation succeeds");
}

#[test]
#[should_panic(expected = "cosine_similarity requires finite eps > 0")]
fn test_cosine_similarity_rejects_nan_epsilon() {
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 1], &[1.0])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let _ = cosine_similarity(&x, &x, 1, f64::NAN).expect("invariant: test operation succeeds");
}

#[test]
#[should_panic(expected = "cosine_similarity requires finite eps > 0")]
fn test_cosine_similarity_rejects_infinite_epsilon() {
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 1], &[1.0])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let _ =
        cosine_similarity(&x, &x, 1, f64::INFINITY).expect("invariant: test operation succeeds");
}

#[test]
fn test_triplet_margin_with_distance_loss() {
    // distance(a,b) = mean(|a - b|). anchor=[0,0], positive=[2,2], negative=[1,1], margin=0.5.
    //   d_ap = mean(|[-2,-2]|) = 2 ; d_an = mean(|[-1,-1]|) = 1
    //   loss = mean(relu(d_ap - d_an + margin)) = relu(2 - 1 + 0.5) = 1.5.
    let anchor = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[0.0, 0.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let positive = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[2.0, 2.0])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let negative = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[1.0, 1.0])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let dist = |a: &Var<f64, MoiraiBackend>, b: &Var<f64, MoiraiBackend>| {
        coeus_autograd::mean(
            &coeus_autograd::abs(
                &coeus_autograd::sub(a, b).expect("invariant: test operation succeeds"),
            )
            .expect("invariant: test operation succeeds"),
        )
    };

    let loss = triplet_margin_with_distance_loss(&anchor, &positive, &negative, dist, 0.5)
        .expect("invariant: test operation succeeds");
    assert_eq!(loss.tensor.shape(), &[1]);
    assert!((loss.tensor.as_slice()[0] - 1.5).abs() < 1e-12);

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(anchor.grad().is_some(), "triplet anchor grad");
}
