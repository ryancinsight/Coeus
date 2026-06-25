use coeus_autograd::{conv_transpose1d, Var};
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
    );
    let out = conv_transpose1d(&input, &weight, &Some(bias.clone()), out_tensor, 1, 0, 0, 1);
    assert_eq!(out.tensor.as_slice(), &[21.0, 40.0, 32.0]);

    let seed = Tensor::from_slice_on(vec![1, 1, 3], &[1.0_f64, 2.0, 3.0], &backend);
    out.backward_with_seed(seed);

    assert_eq!(input.grad().unwrap().as_slice(), &[19.0, 31.0]);
    assert_eq!(weight.grad().unwrap().as_slice(), &[8.0, 13.0]);
    assert_eq!(bias.grad().unwrap().as_slice(), &[6.0]);
}
