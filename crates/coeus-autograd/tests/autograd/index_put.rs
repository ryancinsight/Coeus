#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use coeus_autograd::{index_put, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn test_index_put_backward_overwrite() {
    let backend = MoiraiBackend::new();
    // x=[1,2,3,4,5], idx=[1,3], v=[10,20], accumulate=false.
    // out = [1,10,3,20,5]. sum().backward():
    //   grad_x = 1 at kept positions {0,2,4}, 0 at overwritten {1,3}.
    //   grad_v = 1 at each inserted position.
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![5], &[1.0, 2.0, 3.0, 4.0, 5.0], &backend),
        true,
    );
    let idx = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![2], &[1.0, 3.0], &backend),
        false,
    );
    let v = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![2], &[10.0, 20.0], &backend),
        true,
    );
    let out = index_put(&x, &idx, &v, false);
    assert_eq!(
        out.tensor.as_slice(),
        &[1.0, 10.0, 3.0, 20.0, 5.0],
        "fwd overwrite"
    );
    out.backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(
        x.grad().unwrap().as_slice(),
        &[1.0, 0.0, 1.0, 0.0, 1.0],
        "grad_x"
    );
    assert_eq!(v.grad().unwrap().as_slice(), &[1.0, 1.0], "grad_v");
}

#[test]
fn test_index_put_backward_accumulate() {
    let backend = MoiraiBackend::new();
    // accumulate=true: out = x with v ADDED at idx = [1,12,3,24,5].
    //   grad_x = 1 everywhere (x fully preserved); grad_v = 1 at each idx.
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![5], &[1.0, 2.0, 3.0, 4.0, 5.0], &backend),
        true,
    );
    let idx = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![2], &[1.0, 3.0], &backend),
        false,
    );
    let v = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![2], &[10.0, 20.0], &backend),
        true,
    );
    let out = index_put(&x, &idx, &v, true);
    assert_eq!(
        out.tensor.as_slice(),
        &[1.0, 12.0, 3.0, 24.0, 5.0],
        "fwd accumulate"
    );
    out.backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(
        x.grad().unwrap().as_slice(),
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        "grad_x accum"
    );
    assert_eq!(v.grad().unwrap().as_slice(), &[1.0, 1.0], "grad_v accum");
}
