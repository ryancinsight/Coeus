//! Parity of the unary element-wise operations and their gradients.

use coeus_tensor::Tensor;

use super::super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

macro_rules! test_unary_parity {
    ($name:ident, $op:expr, $data:expr) => {
        #[test]
        fn $name() {
            let s = seq();
            let w = wgpu();
            let data: Vec<f32> = $data;
            let x = Tensor::from_slice(vec![data.len()], &data)
                .expect("invariant: test backend operation succeeds");
            let cpu = $op(&x, &s).expect("invariant: test operation succeeds");
            let gpu = to_cpu(&$op(&to_gpu(&x), &w).expect("invariant: test operation succeeds"));
            assert_parity(stringify!($name), cpu.as_slice(), gpu.as_slice());
        }
    };
}

test_unary_parity!(
    test_wgpu_parity_relu,
    coeus_ops::relu,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 3.0]
);
test_unary_parity!(
    test_wgpu_parity_sigmoid,
    coeus_ops::sigmoid,
    vec![-3.0, -1.0, 0.0, 1.0, 3.0, -2.0, 0.5, 2.0]
);
test_unary_parity!(
    test_wgpu_parity_tanh,
    coeus_ops::tanh,
    vec![-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, -1.5, 1.5]
);
test_unary_parity!(
    test_wgpu_parity_gelu,
    coeus_ops::gelu,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
test_unary_parity!(
    test_wgpu_parity_gelu_tanh,
    coeus_ops::gelu_tanh,
    vec![-3.0, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.0]
);
test_unary_parity!(
    test_wgpu_parity_silu,
    coeus_ops::silu,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
test_unary_parity!(
    test_wgpu_parity_mish,
    coeus_ops::mish,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
test_unary_parity!(
    test_wgpu_parity_elu,
    coeus_ops::elu,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
test_unary_parity!(
    test_wgpu_parity_softplus,
    coeus_ops::softplus,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
test_unary_parity!(
    test_wgpu_parity_exp,
    coeus_ops::exp,
    vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -2.0, 2.0]
);
test_unary_parity!(
    test_wgpu_parity_log,
    coeus_ops::log,
    vec![0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 0.25, 16.0]
);
test_unary_parity!(
    test_wgpu_parity_sqrt,
    coeus_ops::sqrt,
    vec![0.25, 1.0, 2.0, 4.0, 9.0, 16.0, 0.5, 25.0]
);
test_unary_parity!(
    test_wgpu_parity_neg,
    coeus_ops::neg,
    vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 3.0, -3.0]
);
test_unary_parity!(
    test_wgpu_parity_abs,
    coeus_ops::abs,
    vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 3.0, -3.0]
);
test_unary_parity!(
    test_wgpu_parity_cos,
    coeus_ops::cos,
    vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, core::f32::consts::PI]
);
test_unary_parity!(
    test_wgpu_parity_sin,
    coeus_ops::sin,
    vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, core::f32::consts::PI]
);
test_unary_parity!(
    test_wgpu_parity_recip,
    coeus_ops::recip,
    vec![-4.0, -2.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0]
);
test_unary_parity!(
    test_wgpu_parity_sign,
    coeus_ops::sign,
    vec![-4.0, -0.25, 0.0, 0.25, 1.0, -1.0, 3.0, -3.0]
);
test_unary_parity!(
    test_wgpu_parity_floor,
    coeus_ops::floor,
    vec![-2.7, -1.2, -0.1, 0.0, 0.1, 1.2, 2.7, 3.0]
);
test_unary_parity!(
    test_wgpu_parity_ceil,
    coeus_ops::ceil,
    vec![-2.7, -1.2, -0.1, 0.0, 0.1, 1.2, 2.7, 3.0]
);
test_unary_parity!(
    test_wgpu_parity_round,
    coeus_ops::round,
    vec![-2.7, -1.6, -1.2, -0.1, 0.1, 1.2, 1.6, 2.7]
);
test_unary_parity!(
    test_wgpu_parity_trunc,
    coeus_ops::trunc,
    vec![-2.7, -1.5, -1.2, -0.1, 0.1, 1.2, 1.5, 2.7]
);

macro_rules! test_unary_grad_parity {
    ($name:ident, $op:expr, $data:expr) => {
        #[test]
        fn $name() {
            let s = seq();
            let w = wgpu();
            let data: Vec<f32> = $data;
            let x = Tensor::from_slice(vec![data.len()], &data)
                .expect("invariant: test backend operation succeeds");
            let cpu = coeus_ops::elementwise_unary(&x, &s, $op).expect("valid CPU unary dispatch");
            let gpu = to_cpu(
                &coeus_ops::elementwise_unary(&to_gpu(&x), &w, $op)
                    .expect("valid WGPU unary dispatch"),
            );
            assert_parity(stringify!($name), cpu.as_slice(), gpu.as_slice());
        }
    };
}

test_unary_grad_parity!(
    test_wgpu_parity_mish_grad,
    coeus_ops::UnaryOp::MishGrad,
    vec![-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 1.5]
);
test_unary_grad_parity!(
    test_wgpu_parity_elu_grad,
    coeus_ops::UnaryOp::EluGrad,
    vec![-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 1.5]
);
