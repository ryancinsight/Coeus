//! Public const decoders retain every packed word and IEEE narrowing boundary.

use coeus_core::CpuUnaryOp;

// Low word is the first parameter, independent of the machine's byte order.
const PAIRED: &[(u64, [u32; 2])] = &[
    (0, [0, 0]),
    (0xffff_ffff_ffff_ffff, [0xffff_ffff, 0xffff_ffff]),
    (0x8000_0000_0000_0000, [0, 0x8000_0000]),
    (0x0000_0001_8000_0000, [0x8000_0000, 1]),
    (0x0080_0000_007f_ffff, [0x007f_ffff, 0x0080_0000]),
    (0x7f80_0000_7f7f_ffff, [0x7f7f_ffff, 0x7f80_0000]),
    (0x7fc1_2345_ff80_0000, [0xff80_0000, 0x7fc1_2345]),
    (0xff81_2345_7f81_2345, [0x7f81_2345, 0xff81_2345]),
];

// Expected encodings follow round-to-nearest, ties-to-even. Neighbours are
// adjacent binary64 encodings, so they distinguish each side of a midpoint.
const SINGLE: &[(u64, u32)] = &[
    (0, 0),
    (1, 0),                     // Smallest binary64 subnormal rounds to binary32 zero.
    (0x0010_0000_0000_0000, 0), // Smallest binary64 normal also rounds to zero.
    (0x368f_ffff_ffff_ffff, 0), // Below half the smallest binary32 subnormal.
    (0x3690_0000_0000_0000, 0), // 2^-150: tie to even zero.
    (0x3690_0000_0000_0001, 1),
    (0x36a0_0000_0000_0000, 1), // 2^-149, smallest binary32 subnormal.
    (0x380f_ffff_dfff_ffff, 0x007f_ffff),
    (0x380f_ffff_e000_0000, 0x0080_0000), // Tie to smallest binary32 normal.
    (0x380f_ffff_e000_0001, 0x0080_0000),
    (0x3810_0000_0000_0000, 0x0080_0000),
    (0x3ff0_0000_0fff_ffff, 0x3f80_0000),
    (0x3ff0_0000_1000_0000, 0x3f80_0000), // Even lower neighbour at 1 + 2^-24.
    (0x3ff0_0000_1000_0001, 0x3f80_0001),
    (0x3ff0_0000_2fff_ffff, 0x3f80_0001),
    (0x3ff0_0000_3000_0000, 0x3f80_0002), // Even upper neighbour.
    (0x3ff0_0000_3000_0001, 0x3f80_0002),
    (0x47ef_ffff_e000_0000, 0x7f7f_ffff), // Largest finite binary32.
    (0x47ef_ffff_efff_ffff, 0x7f7f_ffff),
    (0x47ef_ffff_f000_0000, 0x7f80_0000), // Overflow midpoint rounds to infinity.
    (0x47ef_ffff_f000_0001, 0x7f80_0000),
    (0x7fef_ffff_ffff_ffff, 0x7f80_0000), // Largest finite binary64.
    (0x7ff0_0000_0000_0000, 0x7f80_0000),
];

const fn single_operations(bits: u64) -> [CpuUnaryOp; 8] {
    [
        CpuUnaryOp::LeakyRelu(bits),
        CpuUnaryOp::LeakyReluGrad(bits),
        CpuUnaryOp::Hardshrink(bits),
        CpuUnaryOp::HardshrinkGrad(bits),
        CpuUnaryOp::Softshrink(bits),
        CpuUnaryOp::SoftshrinkGrad(bits),
        CpuUnaryOp::Celu(bits),
        CpuUnaryOp::CeluGrad(bits),
    ]
}

const fn decode_single_cases() -> [[[[f32; 2]; 8]; 2]; SINGLE.len()] {
    let mut decoded = [[[[0.0; 2]; 8]; 2]; SINGLE.len()];
    let mut case = 0;
    while case < SINGLE.len() {
        let mut sign = 0;
        while sign < 2 {
            let bits = SINGLE[case].0 | if sign == 0 { 0 } else { 1 << 63 };
            let operations = single_operations(bits);
            let mut operation = 0;
            while operation < operations.len() {
                decoded[case][sign][operation] = match operations[operation].parameter_pair() {
                    Some(pair) => pair,
                    None => panic!("single-parameter operation must decode"),
                };
                operation += 1;
            }
            sign += 1;
        }
        case += 1;
    }
    decoded
}

const SINGLE_DECODED: [[[[f32; 2]; 8]; 2]; SINGLE.len()] = decode_single_cases();

const fn decode_paired_cases() -> [[[f32; 2]; 5]; PAIRED.len()] {
    let mut decoded = [[[0.0; 2]; 5]; PAIRED.len()];
    let mut case = 0;
    while case < PAIRED.len() {
        let bits = PAIRED[case].0;
        decoded[case][0] = CpuUnaryOp::decode_parameter_pair(bits);
        let operations = [
            CpuUnaryOp::Hardtanh(bits),
            CpuUnaryOp::HardtanhGrad(bits),
            CpuUnaryOp::Threshold(bits),
            CpuUnaryOp::ThresholdGrad(bits),
        ];
        let mut operation = 0;
        while operation < operations.len() {
            decoded[case][operation + 1] = match operations[operation].parameter_pair() {
                Some(pair) => pair,
                None => panic!("paired operation must decode"),
            };
            operation += 1;
        }
        case += 1;
    }
    decoded
}

const PAIRED_DECODED: [[[f32; 2]; 5]; PAIRED.len()] = decode_paired_cases();

#[test]
fn const_paired_parameters_preserve_all_word_bits() {
    for (decoded, (_, expected)) in PAIRED_DECODED.iter().zip(PAIRED) {
        for pair in decoded {
            assert_eq!(pair.map(f32::to_bits), *expected);
        }
    }
}

#[test]
fn const_single_parameters_round_at_ieee_boundaries() {
    for (decoded, (_, expected)) in SINGLE_DECODED.iter().zip(SINGLE) {
        for (sign, operations) in decoded.iter().enumerate() {
            let expected = expected | if sign == 0 { 0 } else { 1 << 31 };
            for pair in operations {
                assert_eq!(pair[0].to_bits(), expected);
                assert_eq!(pair[1].to_bits(), 0);
            }
        }
    }
}

#[test]
fn nan_parameters_remain_nan() {
    const NAN: Option<[f32; 2]> = CpuUnaryOp::Celu(0x7ff0_0000_0000_0001).parameter_pair();
    assert!(NAN.expect("const NaN parameter")[0].is_nan());
    for bits in [
        0x7ff0_0000_0000_0001,
        0x7ff8_0000_0000_0001,
        0xfff0_0000_0000_0001,
        0xfff8_0000_0000_0001,
    ] {
        for operation in single_operations(bits) {
            let pair = operation.parameter_pair().expect("NaN parameter");
            assert!(pair[0].is_nan());
            assert_eq!(pair[1].to_bits(), 0);
        }
    }
}
