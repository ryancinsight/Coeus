use coeus_core::CpuUnaryOp;

#[test]
fn single_activation_parameters_decode_the_public_encoding() {
    for (parameter, expected) in [(0.0_f64, 0.0_f32), (0.5, 0.5), (1.25, 1.25), (-2.0, -2.0)] {
        let bits = parameter.to_bits();
        for operation in [
            CpuUnaryOp::LeakyRelu(bits),
            CpuUnaryOp::LeakyReluGrad(bits),
            CpuUnaryOp::Hardshrink(bits),
            CpuUnaryOp::HardshrinkGrad(bits),
            CpuUnaryOp::Softshrink(bits),
            CpuUnaryOp::SoftshrinkGrad(bits),
            CpuUnaryOp::Celu(bits),
            CpuUnaryOp::CeluGrad(bits),
        ] {
            assert_eq!(operation.parameter_pair(), Some([expected, 0.0]));
        }
    }
}

#[test]
fn paired_activation_parameters_preserve_both_words() {
    let first = -1.25_f32;
    let second = 2.5_f32;
    let bits = u64::from(first.to_bits()) | (u64::from(second.to_bits()) << 32);
    for operation in [
        CpuUnaryOp::Hardtanh(bits),
        CpuUnaryOp::HardtanhGrad(bits),
        CpuUnaryOp::Threshold(bits),
        CpuUnaryOp::ThresholdGrad(bits),
    ] {
        assert_eq!(operation.parameter_pair(), Some([first, second]));
    }
    assert_eq!(CpuUnaryOp::Relu.parameter_pair(), None);
}
