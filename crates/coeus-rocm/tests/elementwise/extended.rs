use super::*;

#[test]
fn native_elementwise_operations_match_leto_with_broadcasting() {
    if !require_device() {
        return;
    }

    let backend = Backend::new();
    assert_eq!(backend.name(), "rocm");
    let lhs_layout = Layout::new([2, 3].into());
    let rhs_layout = Layout::new([1, 3].into());
    let output_layout = Layout::new([2, 3].into());
    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [2.0_f32, 4.0, 6.0];
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut device_lhs = unsafe { backend.allocate::<f32>(lhs.len()) }
        .expect("invariant: test backend operation succeeds");
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut device_rhs = unsafe { backend.allocate::<f32>(rhs.len()) }
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&lhs, &mut device_lhs)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&rhs, &mut device_rhs)
        .expect("invariant: test backend operation succeeds");

    for operation in [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Eq,
        BinaryOp::Ne,
        BinaryOp::Lt,
        BinaryOp::Gt,
        BinaryOp::Le,
        BinaryOp::Ge,
    ] {
        let mut expected = [0.0_f32; 6];
        coeus_leto::elementwise_binary_into(
            operation,
            &lhs_layout,
            &lhs,
            &rhs_layout,
            &rhs,
            &output_layout,
            &mut expected,
        )
        .expect("Leto binary elementwise oracle failed");
        // SAFETY: the dispatched provider operation overwrites every logical output element before the buffer is read.
        let mut actual = unsafe { backend.allocate::<f32>(lhs.len()) }
            .expect("invariant: test backend operation succeeds");
        backend
            .elementwise_binary(
                operation,
                &device_lhs,
                &lhs_layout,
                &device_rhs,
                &rhs_layout,
                &mut actual,
                &output_layout,
            )
            .expect("ROCm binary elementwise dispatch failed");
        let mut actual_values = [0.0_f32; 6];
        backend
            .copy_to_host(&actual, &mut actual_values)
            .expect("invariant: test backend operation succeeds");
        assert_close(&actual_values, &expected, "binary");
    }

    assert_integer_comparisons(&backend, &[1_i32, -2, 3, 3, 7, 0], &[1, 0, 4, 3, 2, 9]);
    assert_integer_comparisons(&backend, &[1_u32, 2, 3, 3, 7, 0], &[1, 0, 4, 3, 2, 9]);

    for (shape, lhs, rhs) in [
        (
            vec![6],
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![6.0_f32; 6],
        ),
        (
            vec![2, 2, 2],
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2.0_f32; 8],
        ),
        (
            vec![1, 2, 2, 2],
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2.0_f32; 8],
        ),
    ] {
        let layout = Layout::new(shape.into());
        // SAFETY: the following `copy_to_device` initializes every element before any read.
        let mut device_lhs = unsafe { backend.allocate::<f32>(lhs.len()) }
            .expect("invariant: test backend operation succeeds");
        // SAFETY: the following `copy_to_device` initializes every element before any read.
        let mut device_rhs = unsafe { backend.allocate::<f32>(rhs.len()) }
            .expect("invariant: test backend operation succeeds");
        backend
            .copy_to_device(&lhs, &mut device_lhs)
            .expect("invariant: test backend operation succeeds");
        backend
            .copy_to_device(&rhs, &mut device_rhs)
            .expect("invariant: test backend operation succeeds");
        let mut expected = vec![0.0_f32; lhs.len()];
        coeus_leto::elementwise_binary_into(
            BinaryOp::Add,
            &layout,
            &lhs,
            &layout,
            &rhs,
            &layout,
            &mut expected,
        )
        .expect("Leto ranked elementwise oracle failed");
        // SAFETY: the dispatched provider operation overwrites every logical output element before the buffer is read.
        let mut actual = unsafe { backend.allocate::<f32>(lhs.len()) }
            .expect("invariant: test backend operation succeeds");
        backend
            .elementwise_binary(
                BinaryOp::Add,
                &device_lhs,
                &layout,
                &device_rhs,
                &layout,
                &mut actual,
                &layout,
            )
            .expect("ROCm ranked elementwise dispatch failed");
        let mut actual_values = vec![0.0_f32; lhs.len()];
        backend
            .copy_to_host(&actual, &mut actual_values)
            .expect("invariant: test backend operation succeeds");
        assert_close(&actual_values, &expected, "ranked binary");
    }

    let unary_input = [0.25_f32, 0.5, 1.0, 2.0, 3.0, 4.0];
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut device_unary_input = unsafe { backend.allocate::<f32>(unary_input.len()) }
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&unary_input, &mut device_unary_input)
        .expect("invariant: test backend operation succeeds");
    for operation in [
        CpuUnaryOp::Sin,
        CpuUnaryOp::Cos,
        CpuUnaryOp::Exp,
        CpuUnaryOp::Log,
        CpuUnaryOp::Neg,
        CpuUnaryOp::Abs,
        CpuUnaryOp::Sqrt,
        CpuUnaryOp::Recip,
    ] {
        let mut expected = [0.0_f32; 6];
        coeus_leto::elementwise_unary_into(
            operation,
            &lhs_layout,
            &unary_input,
            &output_layout,
            &mut expected,
        )
        .expect("Leto unary elementwise oracle failed");
        // SAFETY: the dispatched provider operation overwrites every logical output element before the buffer is read.
        let mut actual = unsafe { backend.allocate::<f32>(unary_input.len()) }
            .expect("invariant: test backend operation succeeds");
        backend
            .elementwise_unary(
                operation,
                &device_unary_input,
                &lhs_layout,
                &mut actual,
                &output_layout,
            )
            .expect("ROCm unary elementwise dispatch failed");
        let mut actual_values = [0.0_f32; 6];
        backend
            .copy_to_host(&actual, &mut actual_values)
            .expect("invariant: test backend operation succeeds");
        assert_close(&actual_values, &expected, "unary");
    }

    macro_rules! assert_unary_math {
        ($operation:expr, $input:expr, $label:expr) => {{
            let math_input: &[f32] = $input;
            let math_layout = Layout::new([math_input.len()].into());
            // SAFETY: the following `copy_to_device` initializes every element before any read.
            let mut device_math_input = unsafe { backend.allocate::<f32>(math_input.len()) }.expect("invariant: test backend operation succeeds");
            backend
                .copy_to_device(math_input, &mut device_math_input)
                .expect("invariant: test backend operation succeeds");
            let mut expected = vec![0.0_f32; math_input.len()];
            coeus_leto::elementwise_unary_into(
                $operation,
                &math_layout,
                math_input,
                &math_layout,
                &mut expected,
            )
            .expect("Leto unary math elementwise oracle failed");
            // SAFETY: the dispatched provider operation overwrites every logical output element before the buffer is read.
            let mut actual = unsafe { backend.allocate::<f32>(math_input.len()) }.expect("invariant: test backend operation succeeds");
            backend
                .elementwise_unary(
                    $operation,
                    &device_math_input,
                    &math_layout,
                    &mut actual,
                    &math_layout,
                )
                .expect("ROCm unary math elementwise dispatch failed");
            let mut actual_values = vec![0.0_f32; math_input.len()];
            backend
                .copy_to_host(&actual, &mut actual_values)
                .expect("invariant: test backend operation succeeds");
            assert_close(&actual_values, &expected, $label);
        }};
    }

    let bounded_math_input = [-0.75_f32, -0.25, 0.0, 0.25, 0.75];
    for operation in [
        CpuUnaryOp::Tan,
        CpuUnaryOp::Asin,
        CpuUnaryOp::Acos,
        CpuUnaryOp::Atan,
        CpuUnaryOp::Sinh,
        CpuUnaryOp::Atanh,
        CpuUnaryOp::Asinh,
        CpuUnaryOp::Expm1,
        CpuUnaryOp::Log1p,
        CpuUnaryOp::Sign,
        CpuUnaryOp::Floor,
        CpuUnaryOp::Ceil,
        CpuUnaryOp::Round,
        CpuUnaryOp::Trunc,
        CpuUnaryOp::Erf,
        CpuUnaryOp::Erfc,
    ] {
        assert_unary_math!(operation, &bounded_math_input, "bounded unary math");
    }

    let positive_math_input = [0.25_f32, 0.5, 1.0, 2.0, 4.0];
    for operation in [
        CpuUnaryOp::Cosh,
        CpuUnaryOp::Log2,
        CpuUnaryOp::Log10,
        CpuUnaryOp::Exp2,
    ] {
        assert_unary_math!(operation, &positive_math_input, "positive unary math");
    }

    let lgamma_reflection_input = [-0.25_f32, -1.5, 0.25, 0.5, 1.0, 4.0];
    assert_unary_math!(
        CpuUnaryOp::Lgamma,
        &lgamma_reflection_input,
        "lgamma reflection"
    );
    let lgamma_pole_input = [0.0_f32, -1.0, -2.0];
    assert_unary_math!(CpuUnaryOp::Lgamma, &lgamma_pole_input, "lgamma poles");

    let acosh_input = [1.0_f32, 1.25, 2.0, 4.0, 8.0];
    assert_unary_math!(CpuUnaryOp::Acosh, &acosh_input, "acosh");

    let activation_input = [-3.0_f32, -1.0, 0.0, 0.25, 1.0, 3.0];
    let activation_layout = Layout::new([6].into());
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut device_activation_input = unsafe { backend.allocate::<f32>(activation_input.len()) }
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&activation_input, &mut device_activation_input)
        .expect("invariant: test backend operation succeeds");
    let hardtanh = u64::from((-1.0_f32).to_bits()) | (u64::from(1.0_f32.to_bits()) << 32);
    let threshold = u64::from(0.25_f32.to_bits()) | (u64::from((-0.5_f32).to_bits()) << 32);
    for operation in [
        CpuUnaryOp::Relu,
        CpuUnaryOp::Sigmoid,
        CpuUnaryOp::Tanh,
        CpuUnaryOp::Gelu,
        CpuUnaryOp::GeluGrad,
        CpuUnaryOp::GeluTanh,
        CpuUnaryOp::Silu,
        CpuUnaryOp::Mish,
        CpuUnaryOp::Elu,
        CpuUnaryOp::Softplus,
        CpuUnaryOp::ReluGrad,
        CpuUnaryOp::SigmoidGrad,
        CpuUnaryOp::TanhGrad,
        CpuUnaryOp::GeluTanhGrad,
        CpuUnaryOp::SiluGrad,
        CpuUnaryOp::MishGrad,
        CpuUnaryOp::EluGrad,
        CpuUnaryOp::SoftplusGrad,
        CpuUnaryOp::Hardtanh(hardtanh),
        CpuUnaryOp::HardtanhGrad(hardtanh),
        CpuUnaryOp::Threshold(threshold),
        CpuUnaryOp::ThresholdGrad(threshold),
    ] {
        let mut expected = [0.0_f32; 6];
        coeus_leto::elementwise_unary_into(
            operation,
            &activation_layout,
            &activation_input,
            &activation_layout,
            &mut expected,
        )
        .expect("Leto activation elementwise oracle failed");
        // SAFETY: the dispatched provider operation overwrites every logical output element before the buffer is read.
        let mut actual = unsafe { backend.allocate::<f32>(activation_input.len()) }
            .expect("invariant: test backend operation succeeds");
        backend
            .elementwise_unary(
                operation,
                &device_activation_input,
                &activation_layout,
                &mut actual,
                &activation_layout,
            )
            .expect("ROCm activation elementwise dispatch failed");
        let mut actual_values = [0.0_f32; 6];
        backend
            .copy_to_host(&actual, &mut actual_values)
            .expect("invariant: test backend operation succeeds");
        assert_close(&actual_values, &expected, "activation");
    }
}
