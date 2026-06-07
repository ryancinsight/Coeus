use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use coeus_ops::fuse::{TensorExprExt, evaluate_fused_cpu};

#[test]
fn test_cpu_fusion_basic() {
    let backend = SequentialBackend::new();
    let shape = vec![3, 4];

    let a = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ]);
    let b = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &[
        2.0, 0.5, 1.5, 2.5,
        3.0, 1.0, -1.0, 0.0,
        0.5, 2.0, -3.0, 4.0,
    ]);
    let c = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &[
        -5.0, 1.0, 2.0, 3.0,
        1.5, 2.5, 0.5, 10.0,
        1.0, -1.0, 5.0, 6.0,
    ]);

    // Expression: (a.expr() * b.expr() + c.expr()).relu()
    let expr = (a.expr() * b.expr() + c.expr()).relu();
    
    // Evaluate CPU fused
    let fused_out = evaluate_fused_cpu(&expr, &backend);

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
    
    let a = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]);
    let b = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 3], &[
        10.0, 20.0, 30.0,
    ]);

    // Expression: a + b
    let expr = a.expr() + b.expr();
    let fused_out = evaluate_fused_cpu(&expr, &backend);

    assert_eq!(fused_out.shape(), &[2, 3]);
    let expected = vec![
        11.0, 22.0, 33.0,
        14.0, 25.0, 36.0,
    ];
    assert_eq!(fused_out.as_slice(), &expected);
}

#[test]
fn test_cpu_fusion_silu() {
    let backend = SequentialBackend::new();
    let shape = vec![5];
    let a = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &[-2.0, -1.0, 0.0, 1.0, 2.0]);

    // Expression: a.expr().silu()
    let expr = a.expr().silu();
    let fused_out = evaluate_fused_cpu(&expr, &backend);

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
    let a = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &[-2.0, -1.0, 0.0, 1.0, 2.0]);

    let expr = a.expr().gelu();
    let fused_out = evaluate_fused_cpu(&expr, &backend);

    assert_eq!(fused_out.shape(), &[5]);

    let out_slice = fused_out.as_slice();
    let a_slice = a.as_slice();
    for i in 0..5 {
        let x = a_slice[i] as f64;
        let v = 0.7978845608 * (x + 0.044715 * x * x * x);
        let expected = 0.5 * x * (1.0 + v.tanh());
        assert!((out_slice[i] - expected as f32).abs() < 1e-5);
    }
}

#[test]
fn test_cpu_fusion_gelu_grad() {
    let backend = SequentialBackend::new();
    let shape = vec![5];
    let a = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &[-2.0, -1.0, 0.0, 1.0, 2.0]);

    let expr = a.expr().gelu_grad();
    let fused_out = evaluate_fused_cpu(&expr, &backend);

    assert_eq!(fused_out.shape(), &[5]);

    let out_slice = fused_out.as_slice();
    let a_slice = a.as_slice();
    for i in 0..5 {
        let x = a_slice[i] as f64;
        let x2 = x * x;
        let v = 0.7978845608 * (x + 0.044715 * x * x2);
        let t = v.tanh();
        let dy = 0.7978845608 * (1.0 + 0.134145 * x2);
        let expected = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dy;
        assert!((out_slice[i] - expected as f32).abs() < 1e-5);
    }
}

