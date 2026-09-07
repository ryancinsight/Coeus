use coeus_core::{ComputeBackend, Layout};
use coeus_ops::{BinaryOp, ElementwiseOps};
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

use super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

#[test]
fn parameterized_activations_match_sequential() {
    let sequential = seq();
    let device = wgpu();
    let input = Tensor::from_slice(vec![8], &[-2.0_f32, -0.5, -0.25, -0.0, 0.0, 0.25, 0.5, 2.0]);
    let device_input = to_gpu(&input);
    for (parameter, slope) in [(0.5_f64, 0.5_f32), (1.25, 1.25)] {
        let bits = parameter.to_bits();
        for operation in [
            coeus_ops::UnaryOp::LeakyRelu(bits),
            coeus_ops::UnaryOp::LeakyReluGrad(bits),
            coeus_ops::UnaryOp::Hardshrink(bits),
            coeus_ops::UnaryOp::HardshrinkGrad(bits),
            coeus_ops::UnaryOp::Softshrink(bits),
            coeus_ops::UnaryOp::SoftshrinkGrad(bits),
            coeus_ops::UnaryOp::Celu(bits),
            coeus_ops::UnaryOp::CeluGrad(bits),
        ] {
            let expected = coeus_ops::elementwise_unary(&input, &sequential, operation)
                .expect("sequential activation dispatch");
            let actual = coeus_ops::elementwise_unary(&device_input, &device, operation)
                .expect("device activation dispatch");
            if matches!(operation, coeus_ops::UnaryOp::LeakyReluGrad(_)) {
                assert_eq!(
                    to_cpu(&actual).as_slice(),
                    &[slope, slope, slope, slope, slope, 1.0, 1.0, 1.0]
                );
            }
            assert_parity(
                "parameterized activation",
                expected.as_slice(),
                to_cpu(&actual).as_slice(),
            );
        }
        let actual = to_cpu(&coeus_ops::leaky_relu(&device_input, &device, parameter));
        let expected = coeus_ops::leaky_relu(&input, &sequential, parameter);
        assert_eq!(actual.as_slice(), expected.as_slice());
    }
}

#[test]
fn test_wgpu_parity_add() {
    let s = seq();
    let a = Tensor::from_slice(vec![4, 4], &(0..16).map(|x| x as f32).collect::<Vec<_>>());
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.5 - 4.0).collect::<Vec<_>>(),
    );
    let cpu = coeus_ops::add(&a, &b, &s);
    let gpu = to_cpu(&coeus_ops::add(&to_gpu(&a), &to_gpu(&b), &wgpu()));
    assert_parity("add", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_sub() {
    let s = seq();
    let a = Tensor::from_slice(vec![4, 4], &(0..16).map(|x| x as f32).collect::<Vec<_>>());
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.5).collect::<Vec<_>>(),
    );
    let cpu = coeus_ops::sub(&a, &b, &s);
    let gpu = to_cpu(&coeus_ops::sub(&to_gpu(&a), &to_gpu(&b), &wgpu()));
    assert_parity("sub", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_mul() {
    let s = seq();
    let a = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.1 + 0.5).collect::<Vec<_>>(),
    );
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.2 - 1.0).collect::<Vec<_>>(),
    );
    let cpu = coeus_ops::mul(&a, &b, &s);
    let gpu = to_cpu(&coeus_ops::mul(&to_gpu(&a), &to_gpu(&b), &wgpu()));
    assert_parity("mul", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_div() {
    let s = seq();
    let a = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| (x as f32 + 1.0) * 0.5).collect::<Vec<_>>(),
    );
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| (x as f32 + 1.0) * 0.25).collect::<Vec<_>>(),
    );
    let cpu = coeus_ops::div(&a, &b, &s);
    let gpu = to_cpu(&coeus_ops::div(&to_gpu(&a), &to_gpu(&b), &wgpu()));
    assert_parity("div", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_assign_compacts_shared_rank_five_view() {
    let s = seq();
    let w = wgpu();
    let values: Vec<f32> = (0..64).map(|value| value as f32).collect();
    let base = Tensor::from_slice([2, 2, 2, 2, 4], &values);
    let rhs = Tensor::from_slice([1, 1, 1, 1, 2], &[10.0, 20.0]);
    let ranges = [(0, 2), (0, 2), (0, 2), (0, 2), (1, 3)];

    let mut expected = base.slice(&ranges);
    coeus_ops::add_assign(&mut expected, &rhs, &s).expect("valid CPU rank-five assignment");
    let expected = expected.to_vec_on(&s);

    let mut actual = to_gpu(&base).slice(&ranges);
    let shared = actual.clone();
    let shared_before = to_cpu(&shared);
    let rhs_gpu = to_gpu(&rhs);
    coeus_ops::add_assign(&mut actual, &rhs_gpu, &w).expect("valid WGPU rank-five assignment");

    assert!(actual.is_contiguous(), "replacement output must be compact");
    assert_eq!(actual.layout().offset(), 0);
    let actual = to_cpu(&actual);
    assert_parity("shared_rank_five_assign", &expected, actual.as_slice());
    let shared_after = to_cpu(&shared);
    assert_parity(
        "shared_rank_five_source",
        shared_before.as_slice(),
        shared_after.as_slice(),
    );
}

#[test]
fn test_wgpu_unary_assign_detaches_shared_view() {
    let w = wgpu();
    let base = Tensor::from_slice([2, 3], &[-3.0_f32, -1.0, 0.0, 2.0, 4.0, -5.0]);
    let mut actual = to_gpu(&base).slice(&[(0, 2), (1, 3)]);
    let shared = actual.clone();

    coeus_ops::neg_assign(&mut actual, &w).expect("WGPU neg assignment");

    assert!(actual.is_contiguous(), "replacement output must be compact");
    assert_eq!(actual.layout().offset(), 0);
    assert_parity(
        "shared_unary_assign",
        &[1.0, 0.0, -4.0, 5.0],
        to_cpu(&actual).as_slice(),
    );
    let shared = to_cpu(&shared);
    assert_parity(
        "shared_unary_source",
        &[-1.0, 0.0, 4.0, -5.0],
        shared.as_slice(),
    );
}

#[test]
fn test_wgpu_partial_update_preserves_parent_and_shared_source() {
    let backend = wgpu();
    let parent_layout = Layout::new([2, 3].into());
    let destination_layout = parent_layout.slice(&[(0, 2), (1, 3)]);
    let rhs_layout = Layout::new([2, 2].into());
    let mut destination = backend.allocate::<f32>(6);
    let mut rhs = backend.allocate::<f32>(4);
    backend.copy_to_device(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut destination);
    backend.copy_to_device(&[10.0, 20.0, 30.0, 40.0], &mut rhs);
    let shared = destination.clone();

    backend
        .elementwise_binary_update(
            BinaryOp::Add,
            &mut destination,
            &destination_layout,
            &rhs,
            &rhs_layout,
        )
        .expect("WGPU partial update");

    let mut actual = [0.0; 6];
    backend.copy_to_host(&destination, &mut actual);
    assert_parity(
        "partial_update",
        &[1.0, 12.0, 23.0, 4.0, 35.0, 46.0],
        &actual,
    );
    let mut shared_values = [0.0; 6];
    backend.copy_to_host(&shared, &mut shared_values);
    assert_parity(
        "partial_update_shared",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &shared_values,
    );
}

#[test]
fn test_wgpu_hephaestus_contiguous_binary_reuses_output_buffer() {
    let s = seq();
    let w = wgpu();
    let a = Tensor::from_slice(vec![4, 4], &(0..16).map(|x| x as f32).collect::<Vec<_>>());
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.5 - 4.0).collect::<Vec<_>>(),
    );
    let a_gpu = to_gpu(&a);
    let b_gpu = to_gpu(&b);
    let mut out_gpu = Tensor::<f32, WgpuBackend>::zeros_on(vec![4, 4], &w);
    let out_layout = out_gpu.layout().clone();
    let allocation_id = out_gpu.storage().allocation_id();

    w.elementwise_binary(
        coeus_ops::BinaryOp::Add,
        a_gpu.storage(),
        a_gpu.layout(),
        b_gpu.storage(),
        b_gpu.layout(),
        out_gpu.storage_mut(),
        &out_layout,
    )
    .expect("valid WGPU addition output buffer");

    assert!(
        out_gpu.storage().allocation_id() == allocation_id,
        "delegated binary path reallocated output buffer"
    );

    let expected = coeus_ops::add(&a, &b, &s);
    let got = to_cpu(&out_gpu);
    assert_parity(
        "hephaestus_binary_into_add",
        expected.as_slice(),
        got.as_slice(),
    );
}

#[test]
fn test_wgpu_hephaestus_contiguous_unary_reuses_output_buffer() {
    let s = seq();
    let w = wgpu();
    let x = Tensor::from_slice(vec![8], &[-4.0f32, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]);
    let x_gpu = to_gpu(&x);
    let mut out_gpu = Tensor::<f32, WgpuBackend>::zeros_on(vec![8], &w);
    let out_layout = out_gpu.layout().clone();
    let allocation_id = out_gpu.storage().allocation_id();

    w.elementwise_unary(
        coeus_ops::UnaryOp::Recip,
        x_gpu.storage(),
        x_gpu.layout(),
        out_gpu.storage_mut(),
        &out_layout,
    )
    .expect("valid WGPU reciprocal output buffer");

    assert!(
        out_gpu.storage().allocation_id() == allocation_id,
        "delegated unary path reallocated output buffer"
    );

    let expected = coeus_ops::recip(&x, &s);
    let got = to_cpu(&out_gpu);
    assert_parity(
        "hephaestus_unary_into_recip",
        expected.as_slice(),
        got.as_slice(),
    );
}

#[test]
fn test_wgpu_aliasing_unary_neg_rejects_provider_bypass() {
    let w = wgpu();
    let data = vec![-4.0f32, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let x_cpu = Tensor::from_slice(vec![data.len()], &data);
    let x_gpu = to_gpu(&x_cpu);

    // Clone shares storage; Hephaestus owns the rejection of aliased buffers.
    let mut out_storage = x_gpu.storage().clone();
    let before = to_cpu(&x_gpu);
    let error = w
        .elementwise_unary(
            coeus_ops::UnaryOp::Neg,
            x_gpu.storage(),
            x_gpu.layout(),
            &mut out_storage,
            x_gpu.layout(),
        )
        .expect_err("aliased negation must not bypass Hephaestus");
    assert!(
        format!("{error:?}").contains("must not alias"),
        "aliased negation must surface the provider rejection, got: {error:?}"
    );
    let mut after = vec![0.0; before.as_slice().len()];
    w.copy_to_host(&out_storage, &mut after);
    assert_eq!(after.as_slice(), before.as_slice());
}

#[test]
fn test_wgpu_aliasing_elu_rejects_provider_bypass() {
    let w = wgpu();
    let x_cpu = Tensor::from_slice(vec![2, 2], &[-2.0f32, -0.5, 0.5, 2.0]);
    let x_gpu = to_gpu(&x_cpu);

    // Hephaestus requires distinct input and output buffers. An aliased ELU
    // must fail instead of executing a consumer-owned fallback expression.
    let mut out_storage = x_gpu.storage().clone();
    let error = w
        .elementwise_unary(
            coeus_ops::UnaryOp::Elu,
            x_gpu.storage(),
            x_gpu.layout(),
            &mut out_storage,
            x_gpu.layout(),
        )
        .expect_err("aliased ELU must not bypass Hephaestus");

    // The Hephaestus provider owns the dispatch and rejects the aliased
    // buffers itself; a silent consumer-owned fallback would return Ok.
    assert!(
        format!("{error:?}").contains("must not alias"),
        "aliased ELU must surface the provider's aliasing rejection, got: {error:?}"
    );

    let transposed = x_gpu.t();
    let mut strided_out_storage = transposed.storage().clone();
    let error = w
        .elementwise_unary(
            coeus_ops::UnaryOp::Elu,
            transposed.storage(),
            transposed.layout(),
            &mut strided_out_storage,
            transposed.layout(),
        )
        .expect_err("aliased strided ELU must not bypass Hephaestus");

    // The strided path may reject the operation form before the provider's
    // aliasing check runs; either typed rejection proves no consumer-owned
    // fallback executed (a silent fallback would return Ok).
    let rendered = format!("{error:?}");
    assert!(
        rendered.contains("must not alias"),
        "aliased strided ELU must fail with a typed rejection, got: {error:?}"
    );
}

#[test]
fn test_wgpu_aliasing_binary_add_rejects_provider_bypass() {
    let w = wgpu();
    let a_data: Vec<f32> = (0..16).map(|x| x as f32 * 0.25 - 2.0).collect();
    let b_data: Vec<f32> = (0..16).map(|x| x as f32 * 0.1 + 0.5).collect();

    let a_cpu = Tensor::from_slice(vec![4, 4], &a_data);
    let b_cpu = Tensor::from_slice(vec![4, 4], &b_data);
    let a_gpu = to_gpu(&a_cpu);
    let b_gpu = to_gpu(&b_cpu);

    // Clone shares storage; Hephaestus owns the rejection of aliased buffers.
    let mut out_storage = a_gpu.storage().clone();
    let before = to_cpu(&a_gpu);
    let error = w
        .elementwise_binary(
            coeus_ops::BinaryOp::Add,
            a_gpu.storage(),
            a_gpu.layout(),
            b_gpu.storage(),
            b_gpu.layout(),
            &mut out_storage,
            a_gpu.layout(),
        )
        .expect_err("aliased addition must not bypass Hephaestus");
    assert!(
        format!("{error:?}").contains("must not alias"),
        "aliased addition must surface the provider rejection, got: {error:?}"
    );
    let mut after = vec![0.0; before.as_slice().len()];
    w.copy_to_host(&out_storage, &mut after);
    assert_eq!(after.as_slice(), before.as_slice());
}

macro_rules! test_unary_parity {
    ($name:ident, $op:expr, $data:expr) => {
        #[test]
        fn $name() {
            let s = seq();
            let w = wgpu();
            let data: Vec<f32> = $data;
            let x = Tensor::from_slice(vec![data.len()], &data);
            let cpu = $op(&x, &s);
            let gpu = to_cpu(&$op(&to_gpu(&x), &w));
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
            let x = Tensor::from_slice(vec![data.len()], &data);
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

#[test]
fn test_wgpu_parameterized_activations_match_cpu() {
    let sequential = seq();
    let backend = wgpu();
    let values = [-2.0_f32, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0];
    let input = Tensor::from_slice(vec![values.len()], &values);
    let device_input = to_gpu(&input);
    let hardtanh = u64::from((-1.0_f32).to_bits()) | (u64::from(1.0_f32.to_bits()) << 32);
    let threshold = u64::from(0.25_f32.to_bits()) | (u64::from((-0.5_f32).to_bits()) << 32);

    for operation in [
        coeus_ops::UnaryOp::Hardtanh(hardtanh),
        coeus_ops::UnaryOp::HardtanhGrad(hardtanh),
        coeus_ops::UnaryOp::Threshold(threshold),
        coeus_ops::UnaryOp::ThresholdGrad(threshold),
    ] {
        let expected = coeus_ops::elementwise_unary(&input, &sequential, operation)
            .expect("valid CPU parameterized activation");
        let actual = to_cpu(
            &coeus_ops::elementwise_unary(&device_input, &backend, operation)
                .expect("valid WGPU parameterized activation"),
        );
        assert_parity(
            "parameterized activation",
            expected.as_slice(),
            actual.as_slice(),
        );
    }
}
