use coeus_autograd::{maximum, minimum, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

// maximum(a,b) / minimum(a,b): elementwise, gradient routes to the selected
// operand; ties (a == b) resolve to the first argument `a`.

fn var(data: &[f64]) -> Var<f64, MoiraiBackend> {
    let backend = MoiraiBackend::new();
    Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![data.len()], data, &backend),
        true,
    )
}

#[test]
fn test_maximum_forward_and_backward() {
    // a=[1,5,3], b=[4,2,3]: max=[4,5,3].
    // grad routes to larger operand; index 2 is a tie → grad to `a`.
    //   grad_a = [0,1,1], grad_b = [1,0,0]
    let a = var(&[1.0, 5.0, 3.0]);
    let b = var(&[4.0, 2.0, 3.0]);
    let out = maximum(&a, &b);
    assert_eq!(out.tensor.as_slice(), &[4.0, 5.0, 3.0], "fwd maximum");
    sum(&out).backward();
    assert_eq!(a.grad().unwrap().as_slice(), &[0.0, 1.0, 1.0], "grad_a max");
    assert_eq!(b.grad().unwrap().as_slice(), &[1.0, 0.0, 0.0], "grad_b max");
}

#[test]
fn test_minimum_forward_and_backward() {
    // a=[1,5,3], b=[4,2,3]: min=[1,2,3].
    // grad routes to smaller operand; index 2 is a tie → grad to `a`.
    //   grad_a = [1,0,1], grad_b = [0,1,0]
    let a = var(&[1.0, 5.0, 3.0]);
    let b = var(&[4.0, 2.0, 3.0]);
    let out = minimum(&a, &b);
    assert_eq!(out.tensor.as_slice(), &[1.0, 2.0, 3.0], "fwd minimum");
    sum(&out).backward();
    assert_eq!(a.grad().unwrap().as_slice(), &[1.0, 0.0, 1.0], "grad_a min");
    assert_eq!(b.grad().unwrap().as_slice(), &[0.0, 1.0, 0.0], "grad_b min");
}

#[test]
fn test_maximum_minimum_partition_identity() {
    // For any a, b: maximum(a,b) + minimum(a,b) == a + b elementwise.
    let a = var(&[-2.0, 7.0, 0.5, 9.0]);
    let b = var(&[3.0, -1.0, 0.5, 4.0]);
    let mx = maximum(&a, &b);
    let mn = minimum(&a, &b);
    for i in 0..4 {
        let sum_mm = mx.tensor.as_slice()[i] + mn.tensor.as_slice()[i];
        let sum_ab = a.tensor.as_slice()[i] + b.tensor.as_slice()[i];
        assert!(
            (sum_mm - sum_ab).abs() < 1e-12,
            "max+min != a+b at {i}: {sum_mm} vs {sum_ab}"
        );
    }
}
