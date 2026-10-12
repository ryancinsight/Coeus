use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= 1e-5,
            "{label}[{index}] = {got}, expected {want}"
        );
    }
}

#[test]
fn test_pad_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![2, 2], &[1.0f32, 2.0f32, 3.0f32, 4.0f32], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::pad(&x, &[(1, 1), (1, 1)], 0.0f32);
    assert_eq!(y.tensor.shape(), &[4, 4]);

    let y_slice = y.tensor.as_slice();
    assert_eq!(y_slice[5], 1.0);
    assert_eq!(y_slice[6], 2.0);
    assert_eq!(y_slice[9], 3.0);
    assert_eq!(y_slice[10], 4.0);

    let grad_out = Tensor::ones_on(vec![4, 4], &backend);
    y.backward_with_seed(grad_out)
        .expect("invariant: valid autograd fixture completes backward");

    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    assert_eq!(gx_slice, &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_squeeze_unsqueeze_autograd() {
    let backend = MoiraiBackend::new();

    // Test Unsqueeze: [2, 3] -> [2, 1, 3]
    let x_val = Tensor::from_slice_on(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::unsqueeze(&x, 1);
    assert_eq!(y.tensor.shape(), &[2, 1, 3]);

    let grad_out = Tensor::from_slice_on(
        vec![2, 1, 3],
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &backend,
    );
    y.backward_with_seed(grad_out)
        .expect("invariant: valid autograd fixture completes backward");

    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[2, 3]);
    assert_eq!(gx.as_slice(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    // Test Squeeze (specific axis): [2, 1, 3] -> [2, 3]
    let x2_val = Tensor::from_slice_on(vec![2, 1, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &backend);
    let x2 = Var::new(x2_val, true);

    let y2 = coeus_autograd::squeeze(&x2, Some(1));
    assert_eq!(y2.tensor.shape(), &[2, 3]);

    let grad_out2 = Tensor::from_slice_on(
        vec![2, 3],
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &backend,
    );
    y2.backward_with_seed(grad_out2)
        .expect("invariant: valid autograd fixture completes backward");

    let gx2 = x2.grad().unwrap();
    assert_eq!(gx2.shape(), &[2, 1, 3]);
    assert_eq!(gx2.as_slice(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    // Test Squeeze (all axes): [1, 2, 1, 3] -> [2, 3]
    let x3_val = Tensor::from_slice_on(
        vec![1, 2, 1, 3],
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &backend,
    );
    let x3 = Var::new(x3_val, true);

    let y3 = coeus_autograd::squeeze(&x3, None);
    assert_eq!(y3.tensor.shape(), &[2, 3]);

    let grad_out3 = Tensor::from_slice_on(
        vec![2, 3],
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &backend,
    );
    y3.backward_with_seed(grad_out3)
        .expect("invariant: valid autograd fixture completes backward");

    let gx3 = x3.grad().unwrap();
    assert_eq!(gx3.shape(), &[1, 2, 1, 3]);
    assert_eq!(gx3.as_slice(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
}

#[test]
fn structural_gradients_preserve_untouched_parent_regions() {
    let backend = MoiraiBackend::new();

    let sliced_input = Var::new(
        Tensor::from_slice_on([4], &[1.0_f32, 2.0, 3.0, 4.0], &backend),
        true,
    );
    let sliced = coeus_autograd::slice(&sliced_input, &[(1, 3)]);
    coeus_autograd::sum(&sliced)
        .backward()
        .expect("slice backward");
    assert_eq!(
        sliced_input
            .grad()
            .expect("slice input gradient")
            .as_slice(),
        &[0.0, 1.0, 1.0, 0.0]
    );

    let split_input = Var::new(
        Tensor::from_slice_on([4], &[1.0_f32, 2.0, 3.0, 4.0], &backend),
        true,
    );
    let chunks = coeus_autograd::split(&split_input, 2, 0);
    coeus_autograd::sum(&chunks[1])
        .backward()
        .expect("split backward");
    assert_eq!(
        split_input.grad().expect("split input gradient").as_slice(),
        &[0.0, 0.0, 1.0, 1.0]
    );

    let left = Var::new(Tensor::from_slice_on([2], &[1.0_f32, 2.0], &backend), true);
    let right = Var::new(Tensor::from_slice_on([2], &[3.0_f32, 4.0], &backend), true);
    let concatenated = coeus_autograd::cat(&[&left, &right], 0);
    concatenated
        .backward_with_seed(Tensor::from_slice_on(
            [4],
            &[10.0_f32, 20.0, 30.0, 40.0],
            &backend,
        ))
        .expect("cat backward");
    assert_eq!(
        left.grad().expect("left gradient").as_slice(),
        &[10.0, 20.0]
    );
    assert_eq!(
        right.grad().expect("right gradient").as_slice(),
        &[30.0, 40.0]
    );
}

/// Verify contiguous() backward is identity — gradient flows through unchanged.
#[test]
fn test_contiguous_backward_is_identity() {
    let backend = MoiraiBackend::new();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Var::new(Tensor::from_slice_on(vec![2, 3], &data, &backend), true);
    // Permute to create a non-contiguous view, then force contiguous.
    let y = coeus_autograd::contiguous(&coeus_autograd::permute(&x, &[1, 0]));
    assert_eq!(y.tensor.shape(), &[3, 2]);
    // sum(contiguous(permute(x))).backward() — grad should be all-ones (same as sum backward).
    coeus_autograd::sum(&y)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[2, 3]);
    // Every element contributed once to the sum, so all grads = 1.
    for &v in gx.as_slice() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "contiguous bwd grad should be 1.0, got {v}"
        );
    }
}

#[test]
fn test_einsum3_matmul_chain_backward() {
    let backend = MoiraiBackend::new();
    let a = Var::new(
        Tensor::from_slice_on(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0], &backend),
        true,
    );
    let b = Var::new(
        Tensor::from_slice_on(vec![2, 2], &[5.0f32, 6.0, 7.0, 8.0], &backend),
        true,
    );
    let c = Var::new(
        Tensor::from_slice_on(vec![2, 2], &[9.0f32, 10.0, 11.0, 12.0], &backend),
        true,
    );

    let y =
        coeus_autograd::einsum3("ij,jk,kl->il", &a, &b, &c).expect("valid einsum3 test subscript");
    assert_eq!(y.tensor.shape(), &[2, 2]);
    assert_eq!(y.tensor.as_slice(), &[413.0, 454.0, 937.0, 1030.0]);

    coeus_autograd::sum(&y)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    assert_close(
        a.grad().unwrap().as_slice(),
        &[233.0, 317.0, 233.0, 317.0],
        "dA",
    );
    assert_close(
        b.grad().unwrap().as_slice(),
        &[76.0, 92.0, 114.0, 138.0],
        "dB",
    );
    assert_close(
        c.grad().unwrap().as_slice(),
        &[62.0, 62.0, 72.0, 72.0],
        "dC",
    );
}

#[test]
fn test_einsum_rejects_operand_count_mismatch() {
    let backend = MoiraiBackend::new();
    let a = Var::new(
        Tensor::from_slice_on(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0], &backend),
        true,
    );
    let Err(err) = coeus_autograd::einsum("ij,jk->ik", &[&a]) else {
        panic!("count mismatch must fail");
    };
    assert!(
        matches!(
            err,
            coeus_autograd::EinsumError::OperandCountMismatch {
                subscript_operands: 2,
                provided: 1,
                ..
            }
        ),
        "expected OperandCountMismatch, got {err:?}"
    );
}

#[test]
fn test_einsum_rejects_rank_mismatch() {
    let backend = MoiraiBackend::new();
    let a = Var::new(
        Tensor::from_slice_on(vec![2, 2, 2], &[1.0f32; 8], &backend),
        true,
    );
    let Err(err) = coeus_autograd::einsum("ij->ji", &[&a]) else {
        panic!("rank mismatch must fail");
    };
    assert!(
        matches!(
            err,
            coeus_autograd::EinsumError::RankMismatch {
                operand: 0,
                expected_ndim: 2,
                actual_ndim: 3,
                ..
            }
        ),
        "expected RankMismatch, got {err:?}"
    );
}

#[test]
fn test_einsum_rejects_malformed_subscript_without_panic() {
    // Garbage with two operands reaches the non-tracked fallback, whose
    // typed rejection is mapped — this panicked before the conversion.
    let backend = MoiraiBackend::new();
    let a = Var::new(
        Tensor::from_slice_on(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0], &backend),
        true,
    );
    let b = Var::new(
        Tensor::from_slice_on(vec![2, 2], &[5.0f32, 6.0, 7.0, 8.0], &backend),
        true,
    );
    let Err(err) = coeus_autograd::einsum("ab,cd->ef", &[&a, &b]) else {
        panic!("malformed subscript must fail");
    };
    assert!(
        matches!(err, coeus_autograd::EinsumError::ForwardFailed { .. }),
        "expected ForwardFailed, got {err:?}"
    );
}

#[test]
fn test_einsum_rejects_empty_subscript_without_panic() {
    // "->" parses to empty labels; the last-two-dims swap check must
    // decline (n < 2 would underflow `n - 2`), not panic.
    let backend = MoiraiBackend::new();
    let a = Var::new(
        Tensor::from_slice_on(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0], &backend),
        true,
    );
    let Err(err) = coeus_autograd::einsum("->", &[&a]) else {
        panic!("empty subscript must fail");
    };
    assert!(
        matches!(
            err,
            coeus_autograd::EinsumError::UnsupportedPattern {
                operand_count: 1,
                ..
            }
        ),
        "expected UnsupportedPattern, got {err:?}"
    );
}

#[test]
fn test_einsum_rejects_unsupported_arity() {
    // Three parts with three Vars passes the count check but no two-operand
    // branch applies (three-operand contraction is `einsum3`'s entry).
    let backend = MoiraiBackend::new();
    let mk = |v: f32| {
        Var::new(
            Tensor::from_slice_on(vec![2, 2], &[v, v, v, v], &backend),
            true,
        )
    };
    let (a, b, c) = (mk(1.0), mk(2.0), mk(3.0));
    let Err(err) = coeus_autograd::einsum("ij,jk,kl->il", &[&a, &b, &c]) else {
        panic!("unsupported arity must fail");
    };
    assert!(
        matches!(
            err,
            coeus_autograd::EinsumError::UnsupportedPattern {
                operand_count: 3,
                ..
            }
        ),
        "expected UnsupportedPattern, got {err:?}"
    );
}

#[test]
fn test_einsum_dot_rejects_length_mismatch() {
    // Same label, one extent: without the check, `mul` broadcasts or
    // panics on unbroadcastable shapes instead of reporting the mismatch.
    let backend = MoiraiBackend::new();
    let a = Var::new(
        Tensor::from_slice_on(vec![4], &[1.0f32, 2.0, 3.0, 4.0], &backend),
        true,
    );
    let b = Var::new(
        Tensor::from_slice_on(vec![3], &[1.0f32, 0.0, 1.0], &backend),
        true,
    );
    let Err(err) = coeus_autograd::einsum("i,i->", &[&a, &b]) else {
        panic!("length mismatch must fail");
    };
    assert!(
        matches!(
            err,
            coeus_autograd::EinsumError::DimMismatch {
                operand: 1,
                expected: 4,
                actual: 3,
                ..
            }
        ),
        "expected DimMismatch, got {err:?}"
    );
}
