use super::*;

#[test]
fn test_cuda_unfold_fold_matches_cpu_reference() {
    let Some((sequential, cuda)) = backends() else {
        return;
    };
    let host = Tensor::<f32, SequentialBackend>::from_slice(
        [1, 1, 3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let device = to_gpu(&host, &sequential, &cuda);
    let cpu_unfold = coeus_ops::unfold2d(&host, 2, 2, 1, 1, 0, 0, 1, 1, &sequential)
        .expect("valid CPU unfold dispatch");
    let cuda_unfold = coeus_ops::unfold2d(&device, 2, 2, 1, 1, 0, 0, 1, 1, &cuda)
        .expect("valid CUDA unfold dispatch");
    assert_eq!(
        to_cpu(&cuda_unfold, &cuda, &sequential).as_slice(),
        cpu_unfold.as_slice()
    );
    let cpu_fold = coeus_ops::fold2d(&cpu_unfold, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, &sequential)
        .expect("valid CPU fold dispatch");
    let cuda_fold = coeus_ops::fold2d(&cuda_unfold, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, &cuda)
        .expect("valid CUDA fold dispatch");
    assert_eq!(
        to_cpu(&cuda_fold, &cuda, &sequential).as_slice(),
        cpu_fold.as_slice()
    );
}

// Elementwise binary.

macro_rules! binary_parity {
    ($name:ident, $op:expr, $a:expr, $b:expr) => {
        #[test]
        fn $name() {
            let Some((s, c)) = backends() else {
                return;
            };
            let a = Tensor::from_slice(vec![4, 4], &$a);
            let b = Tensor::from_slice(vec![4, 4], &$b);
            let cpu = $op(&a, &b, &s);
            let gpu = to_cpu(&$op(&to_gpu(&a, &s, &c), &to_gpu(&b, &s, &c), &c), &c, &s);
            assert_parity_tol(stringify!($name), cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
        }
    };
}

binary_parity!(
    test_cuda_parity_add,
    coeus_ops::add,
    (0..16).map(|x| x as f32).collect::<Vec<_>>(),
    (0..16).map(|x| x as f32 * 0.5 - 4.0).collect::<Vec<_>>()
);
binary_parity!(
    test_cuda_parity_sub,
    coeus_ops::sub,
    (0..16).map(|x| x as f32).collect::<Vec<_>>(),
    (0..16).map(|x| x as f32 * 0.5).collect::<Vec<_>>()
);
binary_parity!(
    test_cuda_parity_mul,
    coeus_ops::mul,
    (0..16).map(|x| x as f32 * 0.1 + 0.5).collect::<Vec<_>>(),
    (0..16).map(|x| x as f32 * 0.2 - 1.0).collect::<Vec<_>>()
);
binary_parity!(
    test_cuda_parity_div,
    coeus_ops::div,
    (0..16).map(|x| (x as f32 + 1.0) * 0.5).collect::<Vec<_>>(),
    (0..16).map(|x| (x as f32 + 1.0) * 0.25).collect::<Vec<_>>()
);

// Unary activations.

macro_rules! unary_parity {
    ($name:ident, $op:expr, $data:expr) => {
        #[test]
        fn $name() {
            let Some((s, c)) = backends() else {
                return;
            };
            let data: Vec<f32> = $data;
            let x = Tensor::from_slice(vec![data.len()], &data);
            let cpu = $op(&x, &s);
            let gpu = to_cpu(&$op(&to_gpu(&x, &s, &c), &c), &c, &s);
            assert_parity_tol(stringify!($name), cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
        }
    };
}

unary_parity!(
    test_cuda_parity_relu,
    coeus_ops::relu,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 3.0]
);
unary_parity!(
    test_cuda_parity_sigmoid,
    coeus_ops::sigmoid,
    vec![-3.0, -1.0, 0.0, 1.0, 3.0, -2.0, 0.5, 2.0]
);
unary_parity!(
    test_cuda_parity_tanh,
    coeus_ops::tanh,
    vec![-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, -1.5, 1.5]
);
unary_parity!(
    test_cuda_parity_gelu,
    coeus_ops::gelu,
    // Include |x| ~ 2.3, where the tanh-approximation of GELU diverges most
    // from the exact-erf contract, so this test genuinely guards the contract.
    vec![-3.0, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.0]
);
unary_parity!(
    test_cuda_parity_gelu_tanh,
    coeus_ops::gelu_tanh,
    vec![-3.0, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.0]
);
unary_parity!(
    test_cuda_parity_silu,
    coeus_ops::silu,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
unary_parity!(
    test_cuda_parity_mish,
    coeus_ops::mish,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
unary_parity!(
    test_cuda_parity_elu,
    coeus_ops::elu,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
unary_parity!(
    test_cuda_parity_softplus,
    coeus_ops::softplus,
    vec![-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
);
unary_parity!(
    test_cuda_parity_exp,
    coeus_ops::exp,
    vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -2.0, 2.0]
);
unary_parity!(
    test_cuda_parity_log,
    coeus_ops::log,
    vec![0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 0.25, 16.0]
);
unary_parity!(
    test_cuda_parity_lgamma,
    coeus_ops::lgamma,
    vec![-0.25, -1.5, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0]
);
unary_parity!(
    test_cuda_parity_lgamma_poles,
    coeus_ops::lgamma,
    vec![0.0, -1.0, -2.0]
);
unary_parity!(
    test_cuda_parity_sqrt,
    coeus_ops::sqrt,
    vec![0.25, 1.0, 2.0, 4.0, 9.0, 16.0, 0.5, 25.0]
);
unary_parity!(
    test_cuda_parity_neg,
    coeus_ops::neg,
    vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 3.0, -3.0]
);
unary_parity!(
    test_cuda_parity_abs,
    coeus_ops::abs,
    vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 3.0, -3.0]
);
unary_parity!(
    test_cuda_parity_cos,
    coeus_ops::cos,
    vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, core::f32::consts::PI]
);
unary_parity!(
    test_cuda_parity_sin,
    coeus_ops::sin,
    vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, core::f32::consts::PI]
);

// Unary activation gradients (the `*Grad` kernel variants). These exercise the
// device kernels driven by autograd backward, including the exact-erf
// `GeluGrad` kernel, against the CPU `eval_unary` reference.

macro_rules! unary_grad_parity {
    ($name:ident, $op:expr, $data:expr) => {
        #[test]
        fn $name() {
            let Some((s, c)) = backends() else {
                return;
            };
            let data: Vec<f32> = $data;
            let x = Tensor::from_slice(vec![data.len()], &data);
            let cpu = coeus_ops::elementwise_unary(&x, &s, $op).expect("valid CPU unary dispatch");
            let gpu = to_cpu(
                &coeus_ops::elementwise_unary(&to_gpu(&x, &s, &c), &c, $op)
                    .expect("valid CUDA unary dispatch"),
                &c,
                &s,
            );
            assert_parity_tol(stringify!($name), cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
        }
    };
}

unary_grad_parity!(
    test_cuda_parity_relu_grad,
    coeus_ops::UnaryOp::ReluGrad,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 3.0]
);
unary_grad_parity!(
    test_cuda_parity_sigmoid_grad,
    coeus_ops::UnaryOp::SigmoidGrad,
    vec![0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 0.3]
);
unary_grad_parity!(
    test_cuda_parity_tanh_grad,
    coeus_ops::UnaryOp::TanhGrad,
    vec![-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 0.3]
);
unary_grad_parity!(
    test_cuda_parity_gelu_grad,
    coeus_ops::UnaryOp::GeluGrad,
    // Span the region where the tanh-approx gradient diverges from exact erf,
    // so this guards the exact-erf GeluGrad kernel contract.
    vec![-3.0, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.0]
);
unary_grad_parity!(
    test_cuda_parity_gelu_tanh_grad,
    coeus_ops::UnaryOp::GeluTanhGrad,
    vec![-3.0, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.0]
);
unary_grad_parity!(
    test_cuda_parity_silu_grad,
    coeus_ops::UnaryOp::SiluGrad,
    vec![-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 1.5]
);
unary_grad_parity!(
    test_cuda_parity_mish_grad,
    coeus_ops::UnaryOp::MishGrad,
    vec![-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 1.5]
);
unary_grad_parity!(
    test_cuda_parity_elu_grad,
    coeus_ops::UnaryOp::EluGrad,
    vec![-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 1.5]
);
unary_grad_parity!(
    test_cuda_parity_softplus_grad,
    coeus_ops::UnaryOp::SoftplusGrad,
    vec![-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 4.0]
);

fn assert_strided_elu_parity(op: coeus_ops::UnaryOp, operation: &'static str) {
    let Some((sequential, cuda)) = backends() else {
        return;
    };
    let data = [
        -3.0f32, -2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 3.0, -0.5, 0.5, 1.5,
    ];
    let host = Tensor::<f32, SequentialBackend>::from_slice([3, 4], &data);
    let host_transposed = host.t();
    let cpu = coeus_ops::elementwise_unary(&host_transposed, &sequential, op)
        .expect("valid CPU strided ELU dispatch");
    let device_transposed = to_gpu(&host, &sequential, &cuda).t();
    let gpu = coeus_ops::elementwise_unary(&device_transposed, &cuda, op)
        .expect("valid CUDA Hephaestus strided ELU dispatch");
    let gpu = to_cpu(&gpu, &cuda, &sequential);
    assert_parity_tol(operation, cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_strided_elu_matches_cpu() {
    assert_strided_elu_parity(coeus_ops::UnaryOp::Elu, "strided_elu");
}

#[test]
fn test_cuda_strided_elu_gradient_matches_cpu() {
    assert_strided_elu_parity(coeus_ops::UnaryOp::EluGrad, "strided_elu_gradient");
}

// Reductions.
