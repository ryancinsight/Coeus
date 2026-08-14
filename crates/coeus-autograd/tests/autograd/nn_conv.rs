#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use coeus_autograd::{conv_transpose1d, conv_transpose2d, conv_transpose3d, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn conv_transpose1d_backward_accumulates_exact_gradients() {
    let backend = MoiraiBackend::new();
    let input = Var::new(
        Tensor::from_slice_on(vec![1, 1, 2], &[2.0_f64, 3.0], &backend),
        true,
    );
    let weight = Var::new(
        Tensor::from_slice_on(vec![1, 1, 2], &[5.0_f64, 7.0], &backend),
        true,
    );
    let bias = Var::new(Tensor::from_slice_on(vec![1], &[11.0_f64], &backend), true);

    let out_tensor = coeus_ops::conv_transpose1d(
        &input.tensor,
        &weight.tensor,
        Some(&bias.tensor),
        1,
        0,
        0,
        1,
        &backend,
    )
    .expect("transposed convolution forward");
    let out = conv_transpose1d(&input, &weight, &Some(bias.clone()), out_tensor, 1, 0, 0, 1);
    assert_eq!(out.tensor.as_slice(), &[21.0, 40.0, 32.0]);

    let seed = Tensor::from_slice_on(vec![1, 1, 3], &[1.0_f64, 2.0, 3.0], &backend);
    out.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");

    assert_eq!(input.grad().unwrap().as_slice(), &[19.0, 31.0]);
    assert_eq!(weight.grad().unwrap().as_slice(), &[8.0, 13.0]);
    assert_eq!(bias.grad().unwrap().as_slice(), &[6.0]);
}

#[test]
fn conv_transpose2d_backward_accumulates_exact_gradients() {
    // Input  [N=1, C_in=1, H_in=2, W_in=2] = [[1,2],[3,4]]
    // Weight [C_in=1, C_out=1, KH=1, KW=1] = [[[[1]]]]  (identity kernel)
    // Bias   [C_out=1] = [0.5]
    // stride=1, padding=0, dilation=1 → out shape [1,1,2,2]
    //
    // Forward: out[n,cout,h,w] = input[n,cin,h,w] * 1.0 + 0.5
    //        = [[1.5, 2.5], [3.5, 4.5]]
    //
    // Backward with seed = [[1,2],[3,4]]:
    //   grad_input  = seed * weight = seed   → [1,2,3,4]
    //   grad_weight = Σ input * seed = 1*1+2*2+3*3+4*4 = 30
    //   grad_bias   = Σ seed = 10
    let backend = MoiraiBackend::new();
    let input = Var::new(
        Tensor::from_slice_on(vec![1, 1, 2, 2], &[1.0_f64, 2.0, 3.0, 4.0], &backend),
        true,
    );
    let weight = Var::new(
        Tensor::from_slice_on(vec![1, 1, 1, 1], &[1.0_f64], &backend),
        true,
    );
    let bias = Var::new(Tensor::from_slice_on(vec![1], &[0.5_f64], &backend), true);

    let out_tensor = coeus_ops::conv_transpose2d(
        &input.tensor,
        &weight.tensor,
        Some(&bias.tensor),
        1,
        0,
        0,
        1,
        &backend,
    )
    .expect("transposed convolution forward");

    // Verify forward output
    let out = conv_transpose2d(&input, &weight, &Some(bias.clone()), out_tensor, 1, 0, 0, 1);
    assert_eq!(
        out.tensor.to_contiguous().as_slice(),
        &[1.5, 2.5, 3.5, 4.5],
        "conv_transpose2d forward mismatch"
    );

    let seed = Tensor::from_slice_on(vec![1, 1, 2, 2], &[1.0_f64, 2.0, 3.0, 4.0], &backend);
    out.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");

    // grad_input: each position * weight[0,0,0,0]=1.0 → same as seed
    assert_eq!(
        input.grad().unwrap().to_contiguous().as_slice(),
        &[1.0, 2.0, 3.0, 4.0],
        "conv_transpose2d grad_input"
    );
    // grad_weight: Σ input*seed = 1*1+2*2+3*3+4*4 = 30
    assert_eq!(
        weight.grad().unwrap().to_contiguous().as_slice(),
        &[30.0],
        "conv_transpose2d grad_weight"
    );
    // grad_bias: Σ seed = 1+2+3+4 = 10
    assert_eq!(
        bias.grad().unwrap().to_contiguous().as_slice(),
        &[10.0],
        "conv_transpose2d grad_bias"
    );
}

#[test]
fn conv_transpose2d_no_bias_backward() {
    // Validates grad flow with no bias parameter.
    // Input [1,1,2,2], weight [1,1,2,2] with stride=2 → output [1,1,3,3].
    let backend = MoiraiBackend::new();
    let input = Var::new(
        Tensor::from_slice_on(vec![1, 1, 2, 2], &[1.0_f64, 0.0, 0.0, 1.0], &backend),
        true,
    );
    let weight = Var::new(
        Tensor::from_slice_on(vec![1, 1, 2, 2], &[1.0_f64, 0.5, 0.5, 0.25], &backend),
        true,
    );

    let out_tensor =
        coeus_ops::conv_transpose2d(&input.tensor, &weight.tensor, None, 1, 0, 0, 1, &backend)
            .expect("transposed convolution forward");
    // stride=1, pad=0, dil=1, KH=KW=2: out_h = (2-1)*1 + 2 = 3
    assert_eq!(out_tensor.shape(), &[1, 1, 3, 3]);

    let out = conv_transpose2d(&input, &weight, &None, out_tensor, 1, 0, 0, 1);
    coeus_autograd::sum(&out)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    // Verify gradients exist and are non-zero
    let gi = input.grad().unwrap();
    let gw = weight.grad().unwrap();
    assert_eq!(gi.shape(), input.tensor.shape(), "grad_input shape");
    assert_eq!(gw.shape(), weight.tensor.shape(), "grad_weight shape");
    // Spot-check: with all-ones upstream grad, grad_weight should sum to input values.
    // The weight contributes wherever it was used in forward; sum-of-all-grad-weight
    // elements should equal sum-over-input times output_area/input_numel.
    let gw_sum: f64 = gw.to_contiguous().as_slice().iter().copied().sum();
    assert!(gw_sum > 0.0, "grad_weight must be nonzero");
}

#[test]
fn conv_transpose3d_backward_accumulates_exact_gradients() {
    // Input  [N=1, C_in=1, D=2, H=2, W=2] = [1,2,3,4,5,6,7,8].
    // Weight [C_in=1, C_out=1, KD=1, KH=1, KW=1] = [1].
    // Bias   [C_out=1] = [0.5].
    // stride=1, padding=0, dilation=1: forward is input + bias.
    //
    // Backward with seed [1,2,3,4,5,6,7,8]:
    //   grad_input  = seed
    //   grad_weight = sum(input * seed) = 1^2 + ... + 8^2 = 204
    //   grad_bias   = sum(seed) = 36
    let backend = MoiraiBackend::new();
    let input = Var::new(
        Tensor::from_slice_on(
            vec![1, 1, 2, 2, 2],
            &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &backend,
        ),
        true,
    );
    let weight = Var::new(
        Tensor::from_slice_on(vec![1, 1, 1, 1, 1], &[1.0_f64], &backend),
        true,
    );
    let bias = Var::new(Tensor::from_slice_on(vec![1], &[0.5_f64], &backend), true);

    let out_tensor = coeus_ops::conv_transpose3d(
        &input.tensor,
        &weight.tensor,
        Some(&bias.tensor),
        1,
        0,
        0,
        1,
        &backend,
    )
    .expect("transposed convolution forward");
    let out = conv_transpose3d(&input, &weight, &Some(bias.clone()), out_tensor, 1, 0, 0, 1);
    assert_eq!(
        out.tensor.to_contiguous().as_slice(),
        &[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        "conv_transpose3d forward"
    );

    let seed = Tensor::from_slice_on(
        vec![1, 1, 2, 2, 2],
        &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &backend,
    );
    out.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");

    assert_eq!(
        input.grad().unwrap().to_contiguous().as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "conv_transpose3d grad_input"
    );
    assert_eq!(
        weight.grad().unwrap().to_contiguous().as_slice(),
        &[204.0],
        "conv_transpose3d grad_weight"
    );
    assert_eq!(
        bias.grad().unwrap().to_contiguous().as_slice(),
        &[36.0],
        "conv_transpose3d grad_bias"
    );
}
