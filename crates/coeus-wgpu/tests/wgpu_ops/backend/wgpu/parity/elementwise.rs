use coeus_ops::ElementwiseOps;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

use super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

#[test]
fn test_wgpu_parity_add() {
    let s = seq();
    let a = Tensor::from_slice(vec![4, 4], &(0..16).map(|x| x as f32).collect::<Vec<_>>())
        .expect("construct tensor");
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.5 - 4.0).collect::<Vec<_>>(),
    )
    .expect("construct tensor");
    let cpu = coeus_ops::add(&a, &b, &s).expect("evaluate addition");
    let gpu = to_cpu(&coeus_ops::add(&to_gpu(&a), &to_gpu(&b), &wgpu()).expect("evaluate addition"));
    assert_parity("add", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_sub() {
    let s = seq();
    let a = Tensor::from_slice(vec![4, 4], &(0..16).map(|x| x as f32).collect::<Vec<_>>())
        .expect("construct tensor");
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.5).collect::<Vec<_>>(),
    )
    .expect("construct tensor");
    let cpu = coeus_ops::sub(&a, &b, &s).expect("evaluate subtraction");
    let gpu = to_cpu(&coeus_ops::sub(&to_gpu(&a), &to_gpu(&b), &wgpu()).expect("evaluate subtraction"));
    assert_parity("sub", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_mul() {
    let s = seq();
    let a = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.1 + 0.5).collect::<Vec<_>>(),
    )
    .expect("construct tensor");
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.2 - 1.0).collect::<Vec<_>>(),
    )
    .expect("construct tensor");
    let cpu = coeus_ops::mul(&a, &b, &s).expect("evaluate multiplication");
    let gpu = to_cpu(&coeus_ops::mul(&to_gpu(&a), &to_gpu(&b), &wgpu()).expect("evaluate multiplication"));
    assert_parity("mul", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_div() {
    let s = seq();
    let a = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| (x as f32 + 1.0) * 0.5).collect::<Vec<_>>(),
    )
    .expect("construct tensor");
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| (x as f32 + 1.0) * 0.25).collect::<Vec<_>>(),
    )
    .expect("construct tensor");
    let cpu = coeus_ops::div(&a, &b, &s).expect("evaluate division");
    let gpu = to_cpu(&coeus_ops::div(&to_gpu(&a), &to_gpu(&b), &wgpu()).expect("evaluate division"));
    assert_parity("div", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_hephaestus_contiguous_binary_reuses_output_buffer() {
    use std::sync::Arc;

    let s = seq();
    let w = wgpu();
    let a = Tensor::from_slice(vec![4, 4], &(0..16).map(|x| x as f32).collect::<Vec<_>>())
        .expect("construct tensor");
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.5 - 4.0).collect::<Vec<_>>(),
    )
    .expect("construct tensor");
    let a_gpu = to_gpu(&a);
    let b_gpu = to_gpu(&b);
    let mut out_gpu = Tensor::<f32, WgpuBackend>::zeros_on(vec![4, 4], &w)
        .expect("construct tensor");
    let out_layout = out_gpu.layout().clone();
    let before = Arc::as_ptr(&out_gpu.storage().buffer);

    w.elementwise_binary(
        coeus_ops::BinaryOp::Add,
        a_gpu.storage(),
        a_gpu.layout(),
        b_gpu.storage(),
        b_gpu.layout(),
        out_gpu.storage_mut().expect("access tensor storage"),
        &out_layout,
    )
    .expect("valid WGPU addition output buffer");

    let after = Arc::as_ptr(&out_gpu.storage().buffer);
    assert_eq!(
        before, after,
        "delegated binary path reallocated output buffer"
    );

    let expected = coeus_ops::add(&a, &b, &s).expect("evaluate addition");
    let got = to_cpu(&out_gpu);
    assert_parity(
        "hephaestus_binary_into_add",
        expected.as_slice(),
        got.as_slice(),
    );
}

#[test]
fn test_wgpu_hephaestus_contiguous_unary_reuses_output_buffer() {
    use std::sync::Arc;

    let s = seq();
    let w = wgpu();
    let x = Tensor::from_slice(vec![8], &[-4.0f32, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0])
        .expect("construct tensor");
    let x_gpu = to_gpu(&x);
    let mut out_gpu = Tensor::<f32, WgpuBackend>::zeros_on(vec![8], &w)
        .expect("construct tensor");
    let out_layout = out_gpu.layout().clone();
    let before = Arc::as_ptr(&out_gpu.storage().buffer);

    w.elementwise_unary(
        coeus_ops::UnaryOp::Recip,
        x_gpu.storage(),
        x_gpu.layout(),
        out_gpu.storage_mut().expect("access tensor storage"),
        &out_layout,
    )
    .expect("valid WGPU reciprocal output buffer");

    let after = Arc::as_ptr(&out_gpu.storage().buffer);
    assert_eq!(
        before, after,
        "delegated unary path reallocated output buffer"
    );

    let expected = coeus_ops::recip(&x, &s).expect("evaluate reciprocal");
    let got = to_cpu(&out_gpu);
    assert_parity(
        "hephaestus_unary_into_recip",
        expected.as_slice(),
        got.as_slice(),
    );
}

#[test]
fn test_wgpu_aliasing_unary_neg_matches_cpu() {
    let s = seq();
    let w = wgpu();
    let data = vec![-4.0f32, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let x_cpu = Tensor::from_slice(vec![data.len()], &data).expect("construct tensor");
    let x_gpu = to_gpu(&x_cpu);

    // Clone shares storage; output aliases input and must use non-hephaestus fallback.
    let mut out_gpu = x_gpu.clone();
    let out_layout = out_gpu.layout().clone();
    w.elementwise_unary(
        coeus_ops::UnaryOp::Neg,
        x_gpu.storage(),
        x_gpu.layout(),
        out_gpu.storage_mut().expect("access tensor storage"),
        &out_layout,
    )
    .expect("valid WGPU negation output buffer");

    let expected = coeus_ops::neg(&x_cpu, &s).expect("evaluate negation");
    let got = to_cpu(&out_gpu);
    assert_parity("aliasing_unary_neg", expected.as_slice(), got.as_slice());
}

#[test]
fn test_wgpu_aliasing_binary_add_matches_cpu() {
    let s = seq();
    let w = wgpu();
    let a_data: Vec<f32> = (0..16).map(|x| x as f32 * 0.25 - 2.0).collect();
    let b_data: Vec<f32> = (0..16).map(|x| x as f32 * 0.1 + 0.5).collect();

    let a_cpu = Tensor::from_slice(vec![4, 4], &a_data).expect("construct tensor");
    let b_cpu = Tensor::from_slice(vec![4, 4], &b_data).expect("construct tensor");
    let a_gpu = to_gpu(&a_cpu);
    let b_gpu = to_gpu(&b_cpu);

    // Clone shares storage; output aliases left input and must use non-hephaestus fallback.
    let mut out_gpu = a_gpu.clone();
    let out_layout = out_gpu.layout().clone();
    w.elementwise_binary(
        coeus_ops::BinaryOp::Add,
        a_gpu.storage(),
        a_gpu.layout(),
        b_gpu.storage(),
        b_gpu.layout(),
        out_gpu.storage_mut().expect("access tensor storage"),
        &out_layout,
    )
    .expect("valid WGPU aliased addition output buffer");

    let expected = coeus_ops::add(&a_cpu, &b_cpu, &s).expect("evaluate addition");
    let got = to_cpu(&out_gpu);
    assert_parity("aliasing_binary_add", expected.as_slice(), got.as_slice());
}

macro_rules! test_unary_parity {
    ($name:ident, $op:expr, $data:expr) => {
        #[test]
        fn $name() {
            let s = seq();
            let w = wgpu();
            let data: Vec<f32> = $data;
            let x = Tensor::from_slice(vec![data.len()], &data).expect("construct tensor");
            let cpu = $op(&x, &s).expect("evaluate unary operation");
            let gpu = to_cpu(&$op(&to_gpu(&x), &w).expect("evaluate unary operation"));
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
