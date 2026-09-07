use coeus_core::{Float, Scalar};
use coeus_ops::CtcBatch;
use eunomia::{Bf16, F16};

pub(super) trait BinaryPrecision: Float + std::ops::Neg<Output = Self> {
    const EPSILON: Self;
}

impl BinaryPrecision for f32 {
    const EPSILON: Self = Self::EPSILON;
}

impl BinaryPrecision for f64 {
    const EPSILON: Self = Self::EPSILON;
}

impl BinaryPrecision for F16 {
    // Binary16 has ten stored significand bits: epsilon = 2^-10.
    const EPSILON: Self = Self(0x1400);
}

impl BinaryPrecision for Bf16 {
    // Bfloat16 has seven stored significand bits: epsilon = 2^-7.
    const EPSILON: Self = Self(0x3c00);
}

pub(super) struct AlignmentOracle<T> {
    pub loss: T,
    pub gradient: Vec<T>,
    pub path_count: usize,
}

/// Enumerate C^T raw paths, collapse adjacent repetitions before deleting blanks,
/// and sum path products. This does not use the provider's extended-target DP.
pub(super) fn enumerate<T: Float + std::ops::Neg<Output = T>>(
    log_probs: &[T],
    shape: [usize; 3],
    sequences: CtcBatch<'_>,
    seed: T,
) -> AlignmentOracle<T> {
    let [frames, batch, classes] = shape;
    let CtcBatch {
        targets,
        input_lengths,
        target_lengths,
        blank,
    } = sequences;
    let mut loss = T::zero();
    let mut gradient = vec![T::zero(); log_probs.len()];
    let mut target_offset = 0;
    let mut path_count = 0;
    for (sample, (&length, &target_length)) in input_lengths.iter().zip(target_lengths).enumerate()
    {
        let target = &targets[target_offset..target_offset + target_length];
        target_offset += target_length;
        let paths = classes.pow(u32::try_from(length).expect("invariant: fixture length fits u32"));
        path_count = path_count.max(paths);
        let mut probability = T::zero();
        let mut occupancy = vec![T::zero(); length * classes];
        for encoded in 0..paths {
            let mut remainder = encoded;
            let mut path = vec![0; length];
            for symbol in &mut path {
                *symbol = remainder % classes;
                remainder /= classes;
            }
            let mut previous = None;
            let collapsed = path
                .iter()
                .copied()
                .filter(|&symbol| {
                    let emit = previous != Some(symbol) && symbol != blank;
                    previous = Some(symbol);
                    emit
                })
                .collect::<Vec<_>>();
            if collapsed != target {
                continue;
            }
            let weight = path
                .iter()
                .enumerate()
                .fold(T::one(), |product, (time, &symbol)| {
                    product * Float::exp(log_probs[(time * batch + sample) * classes + symbol])
                });
            probability += weight;
            for (time, &symbol) in path.iter().enumerate() {
                occupancy[time * classes + symbol] += weight;
            }
        }
        assert!(
            probability > T::zero(),
            "fixture has a positive-probability alignment"
        );
        let divisor = count::<T>(batch * target_length.max(1));
        loss -= Float::ln(probability) / divisor;
        for time in 0..length {
            for class in 0..classes {
                gradient[(time * batch + sample) * classes + class] =
                    -seed * (occupancy[time * classes + class] / probability) / divisor;
            }
        }
    }
    assert_eq!(gradient.len(), frames * batch * classes);
    AlignmentOracle {
        loss,
        gradient,
        path_count,
    }
}

pub(super) fn count<T: Float>(value: usize) -> T {
    // Fixture extents are at most 81: exact in every tested scalar format.
    let value = u16::try_from(value).expect("invariant: small fixture extent fits u16");
    <T as Scalar>::from_f64(f64::from(value))
}

pub(super) fn close<T: BinaryPrecision>(actual: T, expected: T, operations: usize) {
    // gamma_k = k*u/(1-k*u), u=epsilon/2. A path contributes at most T
    // products; each DP step has at most two exp, two adds, a log and an add.
    // The caller counts both forward/backward paths, positive path summation,
    // normalization, and log-softmax when present. The scale includes one for
    // absolute error near zero; positive fixture path weights stay normal and
    // losses < 4. This bounds these short fixtures, not arbitrary log inputs
    // with cancellation or underflow. Transcendental rounding is included in
    // the count; it is a numerical test model, not a libm accuracy proof.
    let unit = T::EPSILON / count::<T>(2);
    let accumulated = count::<T>(operations) * unit;
    assert!(accumulated < T::one(), "roundoff model requires k*u < 1");
    let scale = T::one() + Float::abs(actual) + Float::abs(expected);
    let bound = accumulated / (T::one() - accumulated) * scale;
    assert!(
        Float::abs(actual - expected) <= bound,
        "actual={actual:?}, expected={expected:?}, bound={bound:?}"
    );
}
