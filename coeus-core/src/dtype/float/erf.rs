// Error-function approximation for scalar float operations.
//
// The implementation uses the standard Cephes piecewise rational form:
// erf(x) is evaluated directly for |x| <= 1, and as 1 - erfc(x) for larger
// magnitudes. This keeps GELU parity close to the former libm-backed contract
// without retaining the workspace libm dependency.

const ERF_T: [f64; 5] = [
    9.604_973_739_870_516,
    90.026_019_720_384_27,
    2_232.005_345_946_843,
    7_003.325_141_128_051,
    55_592.301_301_039_49,
];

const ERF_U: [f64; 5] = [
    33.561_714_164_750_31,
    521.357_949_780_152_7,
    4_594.323_829_709_801,
    22_629.000_061_389_094,
    49_267.394_260_863_59,
];

const ERFC_P: [f64; 9] = [
    2.461_969_814_735_305e-10,
    0.564_189_564_831_068_8,
    7.463_210_564_422_699,
    48.637_197_098_568_14,
    196.520_832_956_077_1,
    526.445_194_995_477_3,
    934.528_527_171_957_6,
    1_027.551_886_895_157,
    557.535_335_369_399_4,
];

const ERFC_Q: [f64; 8] = [
    13.228_195_115_474_499,
    86.707_214_088_598_97,
    354.937_778_887_819_9,
    975.708_501_743_205_5,
    1_823.909_166_879_097_3,
    2_246.337_608_187_109_7,
    1_656.663_091_941_613_5,
    557.535_340_817_727_7,
];

const ERFC_R: [f64; 6] = [
    0.564_189_583_547_755_1,
    1.275_366_707_599_781,
    5.019_050_422_511_805,
    6.160_210_979_930_536,
    7.409_742_699_504_489,
    2.978_866_653_721_002,
];

const ERFC_S: [f64; 6] = [
    2.260_528_632_201_172_6,
    9.396_035_249_380_015,
    12.048_953_980_809_666,
    17.081_445_074_756_59,
    9.608_968_090_632_86,
    3.369_076_451_000_815,
];

#[inline(always)]
fn polevl(x: f64, coefficients: &[f64]) -> f64 {
    let Some((&first, rest)) = coefficients.split_first() else {
        return 0.0;
    };
    rest.iter()
        .fold(first, |acc, &coefficient| acc * x + coefficient)
}

#[inline(always)]
fn p1evl(x: f64, coefficients: &[f64]) -> f64 {
    let Some((&first, rest)) = coefficients.split_first() else {
        return x;
    };
    rest.iter()
        .fold(x + first, |acc, &coefficient| acc * x + coefficient)
}

#[inline(always)]
pub(crate) fn erf_f32(x: f32) -> f32 {
    erf_f64(f64::from(x)) as f32
}

#[inline(always)]
pub(crate) fn erf_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return x.signum();
    }

    let z = x.abs();
    if z > 26.0 {
        return x.signum();
    }

    let value = if z <= 1.0 {
        let squared = z * z;
        z * polevl(squared, &ERF_T) / p1evl(squared, &ERF_U)
    } else {
        let erfc = erfc_positive(z);
        1.0 - erfc
    };

    if x < 0.0 {
        -value
    } else {
        value
    }
}

#[inline(always)]
fn erfc_positive(x: f64) -> f64 {
    let (numerator, denominator) = if x < 8.0 {
        (polevl(x, &ERFC_P), p1evl(x, &ERFC_Q))
    } else {
        (polevl(x, &ERFC_R), p1evl(x, &ERFC_S))
    };
    (-x * x).exp() * numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_reference_values() {
        let cases = [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (4.0, 0.999_999_984_582_742_1),
        ];

        for (input, expected) in cases {
            let actual = erf_f64(input);
            assert!(
                (actual - expected).abs() <= 2.0e-15,
                "erf_f64({input}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn erf_is_odd_for_finite_inputs() {
        for input in [0.125, 0.5, 1.0, 2.0, 6.0, 12.0] {
            assert_eq!(erf_f64(-input), -erf_f64(input));
        }
    }

    #[test]
    fn erf_handles_special_values() {
        assert_eq!(erf_f64(f64::INFINITY), 1.0);
        assert_eq!(erf_f64(f64::NEG_INFINITY), -1.0);
        assert!(erf_f64(f64::NAN).is_nan());
    }

    #[test]
    fn erf_f32_matches_reference_values() {
        let cases = [
            (0.0_f32, 0.0_f32),
            (0.5_f32, 0.520_499_9_f32),
            (1.0_f32, 0.842_700_8_f32),
            (2.0_f32, 0.995_322_3_f32),
        ];

        for (input, expected) in cases {
            let actual = erf_f32(input);
            assert!(
                (actual - expected).abs() <= 2.0e-7,
                "erf_f32({input}) = {actual}, expected {expected}"
            );
        }
    }
}
