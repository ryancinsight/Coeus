use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn test_var_ops_overloads() {
    let backend = MoiraiBackend::new();
    let a_val = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend).expect("valid tensor construction");
    let b_val = Tensor::from_slice_on(vec![3], &[4.0f64, 5.0, 6.0], &backend).expect("valid tensor construction");

    let a = Var::new(a_val, true).expect("valid variable construction");
    let b = Var::new(b_val, true).expect("valid variable construction");

    // Test + operator: a + b
    let sum = {
        use coeus_autograd::add;
        add(&a, &b).expect("valid autograd operation")
    };
    let sum_op = (&a + &b).expect("valid autograd operation");
    assert_eq!(sum.tensor.as_slice(), sum_op.tensor.as_slice());

    // Test - operator: a - b
    let diff_op = (&a - &b).expect("valid autograd operation");
    let diff_s = diff_op.tensor.as_slice();
    assert!((diff_s[0] - (-3.0)).abs() < 1e-10);
    assert!((diff_s[1] - (-3.0)).abs() < 1e-10);
    assert!((diff_s[2] - (-3.0)).abs() < 1e-10);

    // Test * operator: a * b (element-wise)
    let prod_op = (&a * &b).expect("valid autograd operation");
    let prod_s = prod_op.tensor.as_slice();
    assert!((prod_s[0] - 4.0).abs() < 1e-10);
    assert!((prod_s[1] - 10.0).abs() < 1e-10);
    assert!((prod_s[2] - 18.0).abs() < 1e-10);

    // Test unary Neg: -&a
    let neg_op = (-&a).expect("valid autograd operation");
    let neg_s = neg_op.tensor.as_slice();
    assert!((neg_s[0] - (-1.0)).abs() < 1e-10);
    assert!((neg_s[1] - (-2.0)).abs() < 1e-10);

    // Test scalar Mul: &a * 5.0
    let scaled = {
        use coeus_autograd::scalar_mul;
        scalar_mul(&a, 5.0f64).expect("valid autograd operation")
    };
    let scaled_op = (&a * 5.0f64).expect("valid autograd operation");
    assert_eq!(scaled.tensor.as_slice(), scaled_op.tensor.as_slice());

    // Gradient check: (a * b).sum().backward() — grad_a = b, grad_b = a
    let a2 = Var::new(
        Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend).expect("valid tensor construction"),
        true,
    ).expect("valid variable construction");
    let b2 = Var::new(
        Tensor::from_slice_on(vec![3], &[4.0f64, 5.0, 6.0], &backend).expect("valid tensor construction"),
        true,
    ).expect("valid variable construction");
    let prod2 = (&a2 * &b2).expect("valid autograd operation");
    let loss2 = coeus_autograd::sum(&prod2).expect("valid autograd operation");
    loss2.backward().expect("valid backward propagation");
    let ga2 = a2.grad().unwrap();
    let gb2 = b2.grad().unwrap();
    assert!((ga2.as_slice()[0] - 4.0).abs() < 1e-10);
    assert!((ga2.as_slice()[1] - 5.0).abs() < 1e-10);
    assert!((ga2.as_slice()[2] - 6.0).abs() < 1e-10);
    assert!((gb2.as_slice()[0] - 1.0).abs() < 1e-10);
    assert!((gb2.as_slice()[1] - 2.0).abs() < 1e-10);
    assert!((gb2.as_slice()[2] - 3.0).abs() < 1e-10);
}
