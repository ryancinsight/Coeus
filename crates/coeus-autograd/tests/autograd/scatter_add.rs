#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use coeus_autograd::{scatter_add, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn test_scatter_add_backward_src_and_input() {
    let backend = MoiraiBackend::new();
    // input=[0,0,0,0,0], idx=[4,1,3], src=[1,2,3], dim=0.
    // out = [0, 2, 0, 3, 1]. sum().backward() (output grad all ones):
    //   grad_input = 1 everywhere (input copied through unchanged).
    //   grad_src   = gather(ones, idx) = [1,1,1] (each src lands once).
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![5], &[0.0; 5], &backend),
        true,
    );
    let idx = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![3], &[4.0, 1.0, 3.0], &backend),
        false,
    );
    let src = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![3], &[1.0, 2.0, 3.0], &backend),
        true,
    );
    let out = scatter_add(&input, 0, &idx, &src);
    assert_eq!(
        out.tensor.as_slice(),
        &[0.0, 2.0, 0.0, 3.0, 1.0],
        "fwd scatter_add"
    );
    out.backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(
        input.grad().unwrap().as_slice(),
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        "grad_input"
    );
    assert_eq!(src.grad().unwrap().as_slice(), &[1.0, 1.0, 1.0], "grad_src");
}

#[test]
fn test_scatter_add_backward_duplicate_indices() {
    let backend = MoiraiBackend::new();
    // Duplicate index 1: two src elements land in the same output slot; each
    // still receives the (single) output gradient there, so grad_src stays
    // [1,1,1] and grad_input = 1 everywhere.
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![4], &[10.0, 20.0, 30.0, 40.0], &backend),
        true,
    );
    let idx = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![3], &[1.0, 1.0, 2.0], &backend),
        false,
    );
    let src = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![3], &[1.0, 2.0, 3.0], &backend),
        true,
    );
    let out = scatter_add(&input, 0, &idx, &src);
    // out[1] = 20 + 1 + 2 = 23, out[2] = 30 + 3 = 33.
    assert_eq!(out.tensor.as_slice(), &[10.0, 23.0, 33.0, 40.0], "fwd dup");
    out.backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(
        input.grad().unwrap().as_slice(),
        &[1.0, 1.0, 1.0, 1.0],
        "grad_input dup"
    );
    assert_eq!(
        src.grad().unwrap().as_slice(),
        &[1.0, 1.0, 1.0],
        "grad_src dup"
    );
}
