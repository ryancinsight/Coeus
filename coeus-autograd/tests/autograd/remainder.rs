use coeus_autograd::{remainder, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

// remainder(a, b) = a - floor(a / b) * b  (sign of the divisor, torch/NumPy).
//   ∂/∂a = 1,  ∂/∂b = -floor(a / b).

#[test]
fn test_remainder_forward_and_backward_positive() {
    let backend = MoiraiBackend::new();
    // a=[5,7,8], b=[3,4,3]:
    //   q = floor([5/3, 7/4, 8/3]) = [1, 1, 2]
    //   out = a - q*b = [5-3, 7-4, 8-6] = [2, 3, 2]
    //   grad_a = [1,1,1], grad_b = -q = [-1,-1,-2]
    let a = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![3], &[5.0, 7.0, 8.0], &backend),
        true,
    );
    let b = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![3], &[3.0, 4.0, 3.0], &backend),
        true,
    );
    let out = remainder(&a, &b);
    assert_eq!(out.tensor.as_slice(), &[2.0, 3.0, 2.0], "fwd remainder");
    sum(&out).backward();
    assert_eq!(a.grad().unwrap().as_slice(), &[1.0, 1.0, 1.0], "grad_a");
    assert_eq!(b.grad().unwrap().as_slice(), &[-1.0, -1.0, -2.0], "grad_b");
}

#[test]
fn test_remainder_sign_of_divisor() {
    let backend = MoiraiBackend::new();
    // Python/torch modulo carries the divisor's sign:
    //   7 % -3 = -2  (q = floor(7/-3) = floor(-2.333) = -3; 7 - (-3)*(-3) = -2)
    //   grad_a = 1, grad_b = -q = 3
    let a = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![1], &[7.0], &backend),
        true,
    );
    let b = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![1], &[-3.0], &backend),
        true,
    );
    let out = remainder(&a, &b);
    assert_eq!(out.tensor.as_slice(), &[-2.0], "fwd sign-of-divisor");
    sum(&out).backward();
    assert_eq!(a.grad().unwrap().as_slice(), &[1.0], "grad_a");
    assert_eq!(b.grad().unwrap().as_slice(), &[3.0], "grad_b");
}

#[test]
fn test_remainder_broadcast_scalar_divisor() {
    let backend = MoiraiBackend::new();
    // Broadcasting a [4] dividend against a [1] divisor: grad_b sums the
    // per-element contributions -q over the broadcast axis.
    //   a=[1,2,3,4], b=[3]: q=floor([1,2,3,4]/3)=[0,0,1,1]
    //   out=[1,2,0,1]; grad_a=[1,1,1,1]; grad_b = -sum(q) = -(0+0+1+1) = -2
    let a = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![4], &[1.0, 2.0, 3.0, 4.0], &backend),
        true,
    );
    let b = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![1], &[3.0], &backend),
        true,
    );
    let out = remainder(&a, &b);
    assert_eq!(out.tensor.as_slice(), &[1.0, 2.0, 0.0, 1.0], "fwd broadcast");
    sum(&out).backward();
    assert_eq!(a.grad().unwrap().as_slice(), &[1.0, 1.0, 1.0, 1.0], "grad_a bcast");
    assert_eq!(b.grad().unwrap().as_slice(), &[-2.0], "grad_b bcast reduced");
}
