use coeus_autograd::Var;
use coeus_tensor::Tensor;

#[test]
fn test_rope_forward_shape() {
    use coeus_nn::positional::RotaryEmbedding;

    let rope = RotaryEmbedding::<f64, coeus_core::MoiraiBackend>::new(16, 4, 10000.0)
        .expect("valid rotary embedding dimensions");
    let input = Var::new(
        Tensor::zeros(vec![2, 4, 3, 4]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = rope.forward(&input).expect("valid RotaryEmbedding input");
    assert_eq!(output.tensor.shape(), &[2, 4, 3, 4]);
}

#[test]
fn test_rope_backward() {
    use coeus_nn::positional::RotaryEmbedding;

    let rope = RotaryEmbedding::<f64, coeus_core::MoiraiBackend>::new(16, 4, 10000.0)
        .expect("valid rotary embedding dimensions");
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 1, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = rope.forward(&input).expect("valid RotaryEmbedding input");
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let g = input
        .grad()
        .expect("invariant: RotaryEmbedding input requires grad");
    assert_eq!(g.shape(), &[1, 2, 1, 4]);

    // NeoX split-half layout duplicates each frequency: indices (i, i+d/2) form a pair.
    // Forward: y_i = x_i*cosθ_i + rotate_half(x)_i*sinθ_i, rotate_half gives [-x2,-x3,x0,x1].
    // With upstream grad ones, dx_i = cosθ_i + sinθ_i, dx_{i+d/2} = cosθ_i - sinθ_i.
    // inv_freq = [1.0, 0.01], so θ = pos * inv_freq => pos0 [0,0], pos1 [1,0.01].
    let expected = [
        1.0_f64,                         // pos0, d0, θ=0: cos0+sin0
        1.0_f64,                         // pos0, d1, θ=0
        1.0_f64,                         // pos0, d2, θ=0: cos0-sin0
        1.0_f64,                         // pos0, d3, θ=0
        1.0_f64.cos() + 1.0_f64.sin(),   // pos1, d0, θ=1
        0.01_f64.cos() + 0.01_f64.sin(), // pos1, d1, θ=0.01
        1.0_f64.cos() - 1.0_f64.sin(),   // pos1, d2, θ=1
        0.01_f64.cos() - 0.01_f64.sin(), // pos1, d3, θ=0.01
    ];
    let grad = g.as_slice();
    for (index, (&actual, &expected_value)) in grad.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected_value).abs() < 1e-6,
            "RoPE grad[{index}] = {actual}, expected {expected_value}"
        );
    }
}

#[test]
fn test_rope_numerical_correctness() {
    use coeus_nn::positional::RotaryEmbedding;

    let rope = RotaryEmbedding::<f64, coeus_core::MoiraiBackend>::new(4, 2, 1.0)
        .expect("valid rotary embedding dimensions");
    let input = Var::new(
        Tensor::from_slice(
            vec![1, 2, 1, 2],
            &[
                1.0, 2.0, // pos 0
                3.0, 4.0, // pos 1
            ],
        )
        .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let output = rope.forward(&input).expect("valid RotaryEmbedding input");
    let out_slice = output.tensor.as_slice();

    // pos 0: angle = 0 → identity
    assert!((out_slice[0] - 1.0).abs() < 1e-6);
    assert!((out_slice[1] - 2.0).abs() < 1e-6);

    // pos 1: angle = 1.0
    let cos1 = 1.0_f64.cos();
    let sin1 = 1.0_f64.sin();
    let expected_x2 = 3.0 * cos1 - 4.0 * sin1;
    let expected_y2 = 4.0 * cos1 + 3.0 * sin1;

    assert!(
        (out_slice[2] - expected_x2).abs() < 1e-6,
        "expected: {expected_x2}, got: {}",
        out_slice[2]
    );
    assert!(
        (out_slice[3] - expected_y2).abs() < 1e-6,
        "expected: {expected_y2}, got: {}",
        out_slice[3]
    );
}

#[test]
fn test_general_transpose_autograd() {
    let input = Var::<f64>::new(
        Tensor::from_slice(
            vec![2, 3, 4],
            &(0..24).map(|i| i as f64).collect::<Vec<f64>>(),
        )
        .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let transposed =
        coeus_autograd::transpose(&input, 0, 2).expect("invariant: test operation succeeds");
    assert_eq!(transposed.tensor.shape(), &[4, 3, 2]);

    let sum = coeus_autograd::sum(&transposed).expect("invariant: test operation succeeds");
    sum.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let g = input
        .grad()
        .expect("invariant: transpose input requires grad");
    assert_eq!(g.shape(), &[2, 3, 4]);
    for &val in g.as_slice() {
        assert!((val - 1.0).abs() < 1e-6);
    }
}
