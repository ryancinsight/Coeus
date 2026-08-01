use coeus_core::{BackendError, FloatOps, MoiraiBackend, SequentialBackend};
use coeus_ops::fuse::{evaluate_fused_cpu, evaluate_fused_reduce_cpu, TensorExprExt};
use coeus_ops::ReductionOp;
use coeus_tensor::Tensor;

#[test]
fn fused_expression_rejects_incompatible_broadcast_shapes() {
    let backend = SequentialBackend::new();
    let left = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0, 2.0]);
    let right = Tensor::<f32, SequentialBackend>::from_slice(vec![3], &[3.0, 4.0, 5.0]);
    let expression = left.expr() + right.expr();

    let error = match evaluate_fused_cpu(&expression, &backend) {
        Ok(_) => panic!("incompatible fused shapes must be rejected"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        BackendError::IncompatibleBroadcast {
            operation: "fused expression",
            from,
            to,
        } if from == [2] && to == [3]
    ));
}

#[test]
fn fused_expression_borrows_inputs_through_parallel_dispatch() {
    const ELEMENT_COUNT: usize = 8_192;

    let backend = MoiraiBackend::new();
    let left_values = (0..ELEMENT_COUNT)
        .map(|value| {
            f32::from(u16::try_from(value).expect("test element index fits the exact f32 range"))
        })
        .collect::<Vec<_>>();
    let right_values = vec![1.0f32; ELEMENT_COUNT];
    let left = Tensor::<f32, MoiraiBackend>::from_slice(vec![ELEMENT_COUNT], &left_values);
    let right = Tensor::<f32, MoiraiBackend>::from_slice(vec![ELEMENT_COUNT], &right_values);
    let expression = left.expr() * 2.0 + right.expr();

    let output = evaluate_fused_cpu(&expression, &backend)
        .expect("borrowed fused expression should complete before its inputs are released");
    let expected = left_values
        .iter()
        .map(|&input| input * 2.0 + 1.0)
        .collect::<Vec<_>>();

    assert_eq!(output.as_slice(), expected);
}

#[test]
fn fused_empty_axis_uses_identities_and_rejects_undefined_reductions() {
    let backend = SequentialBackend::new();
    let input = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 0], &[]);
    let expression = input.expr();

    let sum = evaluate_fused_reduce_cpu(&expression, ReductionOp::Sum, 1, &backend)
        .expect("empty sum should return its additive identity");
    let product = evaluate_fused_reduce_cpu(&expression, ReductionOp::Prod, 1, &backend)
        .expect("empty product should return its multiplicative identity");
    assert_eq!(sum.as_slice(), &[0.0, 0.0]);
    assert_eq!(product.as_slice(), &[1.0, 1.0]);
    for operation in [ReductionOp::Mean, ReductionOp::Max, ReductionOp::Min] {
        let error = match evaluate_fused_reduce_cpu(&expression, operation, 1, &backend) {
            Ok(_) => panic!("undefined empty reduction must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            BackendError::EmptyReduction {
                operation: "fused reduction",
                reduction,
            } if reduction == operation
        ));
    }
}

#[test]
fn test_cpu_fusion_basic() {
    let backend = SequentialBackend::new();
    let shape = vec![3, 4];

    let a = Tensor::<f32, SequentialBackend>::from_slice(
        shape.clone(),
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let b = Tensor::<f32, SequentialBackend>::from_slice(
        shape.clone(),
        &[2.0, 0.5, 1.5, 2.5, 3.0, 1.0, -1.0, 0.0, 0.5, 2.0, -3.0, 4.0],
    );
    let c = Tensor::<f32, SequentialBackend>::from_slice(
        shape.clone(),
        &[
            -5.0, 1.0, 2.0, 3.0, 1.5, 2.5, 0.5, 10.0, 1.0, -1.0, 5.0, 6.0,
        ],
    );

    // Expression: (a.expr() * b.expr() + c.expr()).relu()
    let expr = (a.expr() * b.expr() + c.expr()).relu();

    // Evaluate CPU fused
    let fused_out = evaluate_fused_cpu(&expr, &backend).expect("fused expression should evaluate");

    // Evaluate CPU sequential manually to compare
    let mut expected = vec![0.0f32; 12];
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();
    let c_slice = c.as_slice();
    for i in 0..12 {
        let val = a_slice[i] * b_slice[i] + c_slice[i];
        expected[i] = if val > 0.0 { val } else { 0.0 };
    }

    assert_eq!(fused_out.as_slice(), &expected);
}

#[test]
fn test_cpu_fusion_broadcasting() {
    let backend = SequentialBackend::new();

    let a =
        Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 3], &[10.0, 20.0, 30.0]);

    // Expression: a + b
    let expr = a.expr() + b.expr();
    let fused_out = evaluate_fused_cpu(&expr, &backend).expect("fused expression should evaluate");

    assert_eq!(fused_out.shape(), &[2, 3]);
    let expected = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
    assert_eq!(fused_out.as_slice(), &expected);
}

#[test]
fn test_cpu_fusion_silu() {
    let backend = SequentialBackend::new();
    let shape = vec![5];
    let a =
        Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &[-2.0, -1.0, 0.0, 1.0, 2.0]);

    // Expression: a.expr().silu()
    let expr = a.expr().silu();
    let fused_out = evaluate_fused_cpu(&expr, &backend).expect("fused expression should evaluate");

    assert_eq!(fused_out.shape(), &[5]);

    // Verify values against standard silu formula
    let out_slice = fused_out.as_slice();
    let a_slice = a.as_slice();
    for i in 0..5 {
        let x = a_slice[i];
        let sig = 1.0 / (1.0 + (-x).exp());
        let expected = x * sig;
        assert!((out_slice[i] - expected).abs() < 1e-6);
    }
}

#[test]
fn test_cpu_fusion_gelu() {
    let backend = SequentialBackend::new();
    let shape = vec![5];
    let a =
        Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &[-2.0, -1.0, 0.0, 1.0, 2.0]);

    let expr = a.expr().gelu();
    let fused_out = evaluate_fused_cpu(&expr, &backend).expect("fused expression should evaluate");

    assert_eq!(fused_out.shape(), &[5]);

    let out_slice = fused_out.as_slice();
    let a_slice = a.as_slice();
    for i in 0..5 {
        let expected = a_slice[i].gelu_op();
        assert!((out_slice[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_cpu_fusion_gelu_grad() {
    let backend = SequentialBackend::new();
    let shape = vec![5];
    let a =
        Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &[-2.0, -1.0, 0.0, 1.0, 2.0]);

    let expr = a.expr().gelu_grad();
    let fused_out = evaluate_fused_cpu(&expr, &backend).expect("fused expression should evaluate");

    assert_eq!(fused_out.shape(), &[5]);

    let out_slice = fused_out.as_slice();
    let a_slice = a.as_slice();
    for i in 0..5 {
        let x = a_slice[i];
        let x2 = x * x;
        let half = 0.5;
        let inv_sqrt_two = core::f32::consts::FRAC_1_SQRT_2;
        let inv_sqrt_two_pi = 0.398_942_3;
        let expected = half * (1.0 + (x * inv_sqrt_two).erf_op())
            + x * ((0.0 - half * x2).exp()) * inv_sqrt_two_pi;
        assert!((out_slice[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_cpu_fusion_reduce_ops() {
    let backend = SequentialBackend::new();
    let a =
        Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &[1.0, -2.0, 3.0, 4.0, 5.0, -6.0]);
    let b = Tensor::<f32, SequentialBackend>::from_slice(
        vec![2, 3],
        &[10.0, 20.0, -30.0, 1.5, -2.0, 0.5],
    );

    let expr = a.expr() + b.expr();

    let sum = evaluate_fused_reduce_cpu(&expr, ReductionOp::Sum, 1, &backend)
        .expect("fused sum should evaluate");
    assert_eq!(sum.shape(), &[2, 1]);
    assert_eq!(sum.as_slice(), &[2.0, 3.0]);

    let product = evaluate_fused_reduce_cpu(&expr, ReductionOp::Prod, 1, &backend)
        .expect("fused product should evaluate");
    assert_eq!(product.shape(), &[2, 1]);
    assert_eq!(product.as_slice(), &[-5346.0, -90.75]);

    let mean = evaluate_fused_reduce_cpu(&expr, ReductionOp::Mean, 1, &backend)
        .expect("fused mean should evaluate");
    assert_eq!(mean.shape(), &[2, 1]);
    assert_eq!(mean.as_slice(), &[2.0 / 3.0, 1.0]);

    let max = evaluate_fused_reduce_cpu(&expr, ReductionOp::Max, 1, &backend)
        .expect("fused maximum should evaluate");
    assert_eq!(max.shape(), &[2, 1]);
    assert_eq!(max.as_slice(), &[18.0, 5.5]);

    let min = evaluate_fused_reduce_cpu(&expr, ReductionOp::Min, 1, &backend)
        .expect("fused minimum should evaluate");
    assert_eq!(min.shape(), &[2, 1]);
    assert_eq!(min.as_slice(), &[-27.0, -5.5]);
}
