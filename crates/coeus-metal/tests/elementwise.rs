use coeus_core::{BinaryOp, ComputeBackend, CpuUnaryOp, Layout, Scalar};
use coeus_metal::MetalBackend;
use coeus_ops::ElementwiseOps;
use std::fmt::Debug;

fn require_device() -> bool {
    let available = hephaestus_metal::MetalDevice::try_default().is_ok();
    if !available {
        assert_ne!(
            std::env::var("HEPHAESTUS_METAL_REQUIRE_DEVICE").as_deref(),
            Ok("1"),
            "Metal CI requires an acquired device"
        );
    }
    available
}

fn assert_close(actual: &[f32], expected: &[f32], operation: &str) {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = f32::EPSILON * 512.0 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "Metal {operation} mismatch at {index}: actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

fn assert_integer_comparisons<T>(backend: &MetalBackend, lhs: &[T], rhs: &[T])
where
    T: Scalar + leto_ops::Scalar + Debug + PartialEq,
    coeus_metal::MetalProvider: coeus_hephaestus::ElementwiseProvider<T>,
{
    let layout = Layout::new([lhs.len()].into());
    let mut device_lhs = backend.allocate::<T>(lhs.len());
    let mut device_rhs = backend.allocate::<T>(rhs.len());
    backend.copy_to_device(lhs, &mut device_lhs);
    backend.copy_to_device(rhs, &mut device_rhs);

    for operation in [
        BinaryOp::Eq,
        BinaryOp::Ne,
        BinaryOp::Lt,
        BinaryOp::Gt,
        BinaryOp::Le,
        BinaryOp::Ge,
    ] {
        let mut expected = vec![T::zero(); lhs.len()];
        coeus_leto::elementwise_binary_into(
            operation,
            &layout,
            lhs,
            &layout,
            rhs,
            &layout,
            &mut expected,
        )
        .expect("Leto integer comparison oracle failed");
        let mut actual = backend.allocate::<T>(lhs.len());
        backend
            .elementwise_binary(
                operation,
                &device_lhs,
                &layout,
                &device_rhs,
                &layout,
                &mut actual,
                &layout,
            )
            .expect("Metal integer comparison dispatch failed");
        let mut actual_values = vec![T::zero(); lhs.len()];
        backend.copy_to_host(&actual, &mut actual_values);
        assert_eq!(
            actual_values, expected,
            "Metal integer {operation:?} mismatch"
        );
    }
}

#[test]
fn native_elementwise_operations_match_leto_with_broadcasting() {
    if !require_device() {
        return;
    }

    let backend = MetalBackend::new();
    assert_eq!(backend.name(), "metal");
    let lhs_layout = Layout::new([2, 3].into());
    let rhs_layout = Layout::new([1, 3].into());
    let output_layout = Layout::new([2, 3].into());
    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [2.0_f32, 4.0, 6.0];
    let mut device_lhs = backend.allocate::<f32>(lhs.len());
    let mut device_rhs = backend.allocate::<f32>(rhs.len());
    backend.copy_to_device(&lhs, &mut device_lhs);
    backend.copy_to_device(&rhs, &mut device_rhs);

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
        let mut actual = backend.allocate::<f32>(lhs.len());
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
            .expect("Metal binary elementwise dispatch failed");
        let mut actual_values = [0.0_f32; 6];
        backend.copy_to_host(&actual, &mut actual_values);
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
        let mut device_lhs = backend.allocate::<f32>(lhs.len());
        let mut device_rhs = backend.allocate::<f32>(rhs.len());
        backend.copy_to_device(&lhs, &mut device_lhs);
        backend.copy_to_device(&rhs, &mut device_rhs);
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
        let mut actual = backend.allocate::<f32>(lhs.len());
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
            .expect("Metal ranked elementwise dispatch failed");
        let mut actual_values = vec![0.0_f32; lhs.len()];
        backend.copy_to_host(&actual, &mut actual_values);
        assert_close(&actual_values, &expected, "ranked binary");
    }

    let unary_input = [0.25_f32, 0.5, 1.0, 2.0, 3.0, 4.0];
    let mut device_unary_input = backend.allocate::<f32>(unary_input.len());
    backend.copy_to_device(&unary_input, &mut device_unary_input);
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
        let mut actual = backend.allocate::<f32>(unary_input.len());
        backend
            .elementwise_unary(
                operation,
                &device_unary_input,
                &lhs_layout,
                &mut actual,
                &output_layout,
            )
            .expect("Metal unary elementwise dispatch failed");
        let mut actual_values = [0.0_f32; 6];
        backend.copy_to_host(&actual, &mut actual_values);
        assert_close(&actual_values, &expected, "unary");
    }

    macro_rules! assert_unary_math {
        ($operation:expr, $input:expr, $label:expr) => {{
            let math_input: &[f32] = $input;
            let math_layout = Layout::new([math_input.len()].into());
            let mut device_math_input = backend.allocate::<f32>(math_input.len());
            backend.copy_to_device(math_input, &mut device_math_input);
            let mut expected = vec![0.0_f32; math_input.len()];
            coeus_leto::elementwise_unary_into(
                $operation,
                &math_layout,
                math_input,
                &math_layout,
                &mut expected,
            )
            .expect("Leto unary math elementwise oracle failed");
            let mut actual = backend.allocate::<f32>(math_input.len());
            backend
                .elementwise_unary(
                    $operation,
                    &device_math_input,
                    &math_layout,
                    &mut actual,
                    &math_layout,
                )
                .expect("Metal unary math elementwise dispatch failed");
            let mut actual_values = vec![0.0_f32; math_input.len()];
            backend.copy_to_host(&actual, &mut actual_values);
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

    let acosh_input = [1.0_f32, 1.25, 2.0, 4.0, 8.0];
    assert_unary_math!(CpuUnaryOp::Acosh, &acosh_input, "acosh");

    let activation_input = [-3.0_f32, -1.0, 0.0, 0.25, 1.0, 3.0];
    let activation_layout = Layout::new([6].into());
    let mut device_activation_input = backend.allocate::<f32>(activation_input.len());
    backend.copy_to_device(&activation_input, &mut device_activation_input);
    for operation in [
        CpuUnaryOp::Relu,
        CpuUnaryOp::Sigmoid,
        CpuUnaryOp::Tanh,
        CpuUnaryOp::Gelu,
        CpuUnaryOp::GeluGrad,
        CpuUnaryOp::GeluTanh,
        CpuUnaryOp::Silu,
        CpuUnaryOp::Softplus,
        CpuUnaryOp::ReluGrad,
        CpuUnaryOp::SigmoidGrad,
        CpuUnaryOp::TanhGrad,
        CpuUnaryOp::GeluTanhGrad,
        CpuUnaryOp::SiluGrad,
        CpuUnaryOp::SoftplusGrad,
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
        let mut actual = backend.allocate::<f32>(activation_input.len());
        backend
            .elementwise_unary(
                operation,
                &device_activation_input,
                &activation_layout,
                &mut actual,
                &activation_layout,
            )
            .expect("Metal activation elementwise dispatch failed");
        let mut actual_values = [0.0_f32; 6];
        backend.copy_to_host(&actual, &mut actual_values);
        assert_close(&actual_values, &expected, "activation");
    }
}
