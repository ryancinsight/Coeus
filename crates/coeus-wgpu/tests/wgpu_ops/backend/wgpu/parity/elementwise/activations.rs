//! Parity of the parameterized activation operations.

use coeus_tensor::Tensor;

use super::super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

#[test]
fn parameterized_activations_match_sequential() {
    let sequential = seq();
    let device = wgpu();
    let input = Tensor::from_slice(vec![8], &[-2.0_f32, -0.5, -0.25, -0.0, 0.0, 0.25, 0.5, 2.0])
        .expect("invariant: test backend operation succeeds");
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
        let actual = to_cpu(
            &coeus_ops::leaky_relu(&device_input, &device, parameter)
                .expect("invariant: test operation succeeds"),
        );
        let expected = coeus_ops::leaky_relu(&input, &sequential, parameter)
            .expect("invariant: test operation succeeds");
        assert_eq!(actual.as_slice(), expected.as_slice());
    }
}

#[test]
fn test_wgpu_parameterized_activations_match_cpu() {
    let sequential = seq();
    let backend = wgpu();
    let values = [-2.0_f32, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0];
    let input = Tensor::from_slice(vec![values.len()], &values)
        .expect("invariant: test backend operation succeeds");
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
