//! CTC likelihood and derivatives from exhaustive short alignment enumeration.

#[path = "ctc_loss_tests/invalid.rs"]
mod invalid;
#[path = "ctc_loss_tests/layouts.rs"]
mod layouts;
#[path = "ctc_loss_tests/oracle.rs"]
mod oracle;

use coeus_autograd::{ctc_loss, log_softmax, Var};
use coeus_core::{BackendError, Float, MoiraiBackend, Scalar, SequentialBackend};
use coeus_nn::ctc_loss as nn_ctc_loss;
use coeus_ops::{BackendOps, CtcBatch, CtcOps};
use coeus_tensor::Tensor;
use eunomia::{Bf16, F16};

use oracle::{close, count, enumerate, BinaryPrecision};

fn variable<T: Float, B: BackendOps<T> + Default>(values: &[T], shape: [usize; 3]) -> Var<T, B> {
    Var::new(Tensor::from_slice(shape, values), true)
}

fn probability_logs<T: Float>(values: &[f64]) -> Vec<T> {
    values
        .iter()
        .map(|&value| Float::ln(<T as Scalar>::from_f64(value)))
        .collect()
}

fn seeded_backward<T: Float, B: BackendOps<T> + Default>(loss: &Var<T, B>, seed: T) {
    loss.backward_with_seed(Tensor::from_slice([1], &[seed]))
        .expect("invariant: a finite CTC fixture has a defined derivative");
}

fn alignment_case<T, B>(
    probabilities: &[f64],
    shape: [usize; 3],
    targets: &[usize],
    input_lengths: &[usize],
    target_lengths: &[usize],
    blank: usize,
) where
    T: BinaryPrecision,
    B: BackendOps<T> + CtcOps<T> + Default,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    let logs = probability_logs::<T>(probabilities);
    let seed = <T as Scalar>::from_f64(2.5);
    let expected = enumerate(
        &logs,
        shape,
        CtcBatch {
            targets,
            input_lengths,
            target_lengths,
            blank,
        },
        seed,
    );
    let input = variable::<T, B>(&logs, shape);
    let loss = ctc_loss(&input, targets, input_lengths, target_lengths, blank)
        .expect("invariant: finite alignment fixture satisfies CTC input contracts");
    let operations = 12 * shape[0] + 2 * expected.path_count + 2 * shape[2] + 8;
    close(loss.tensor.as_slice()[0], expected.loss, operations);
    seeded_backward(&loss, seed);
    let gradient = input
        .grad()
        .expect("invariant: tracked CTC input receives a gradient");
    assert_eq!(gradient.shape(), &shape);
    for (index, (&actual, &reference)) in gradient
        .as_slice()
        .iter()
        .zip(&expected.gradient)
        .enumerate()
    {
        let time = index / (shape[1] * shape[2]);
        let sample = index / shape[2] % shape[1];
        if time >= input_lengths[sample] {
            assert_eq!(actual, T::zero(), "padded frame {time}, sample {sample}");
        } else {
            close(actual, reference, operations);
        }
    }
    let untracked = Var::new(Tensor::<T, B>::from_slice(shape, &logs), false);
    let wrapper = nn_ctc_loss(&untracked, targets, input_lengths, target_lengths, blank)
        .expect("invariant: NN wrapper accepts the same valid CTC fixture");
    assert_eq!(wrapper.tensor.as_slice(), loss.tensor.as_slice());
}

fn empty_target_retains_the_blank_path_probability<T, B>()
where
    T: BinaryPrecision,
    B: BackendOps<T> + CtcOps<T> + Default,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    let half = <T as Scalar>::from_f64(0.5);
    let log_half = Float::ln(half);
    let input = variable::<T, B>(&[log_half, log_half], [1, 1, 2]);
    let loss = ctc_loss(&input, &[], &[1], &[0], 0)
        .expect("invariant: an empty target has the all-blank alignment");
    assert_eq!(loss.tensor.as_slice(), &[-log_half]);
    seeded_backward(&loss, count::<T>(2));
    assert_eq!(
        input
            .grad()
            .expect("invariant: blank path is tracked")
            .as_slice(),
        &[-count::<T>(2), T::zero()]
    );
}

fn exact_seed_and_accumulation<T, B>()
where
    T: BinaryPrecision,
    B: BackendOps<T> + CtcOps<T> + Default,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    // One frame forces the target symbol. Independent log inputs may be zero
    // without normalization; posterior [0,1,0] and dyadic seed are exact.
    let input = variable::<T, B>(&[T::zero(); 3], [1, 1, 3]);
    let initial = <T as Scalar>::from_f64(0.75);
    input.set_grad(Tensor::from_slice([1, 1, 3], &[initial; 3]));
    let loss = ctc_loss(&input, &[1], &[1], &[1], 0)
        .expect("invariant: a one-frame target has one alignment");
    assert_eq!(loss.tensor.as_slice(), &[T::zero()]);
    seeded_backward(&loss, <T as Scalar>::from_f64(2.5));
    assert_eq!(
        input
            .grad()
            .expect("invariant: seeded leaf gradient exists")
            .as_slice(),
        &[initial, <T as Scalar>::from_f64(-1.75), initial]
    );
}

fn logits_follow_the_softmax_chain_rule<T, B>()
where
    T: BinaryPrecision,
    B: BackendOps<T> + CtcOps<T> + Default,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    let shape = [2, 1, 3];
    let logits = [0., 1., -1., 1., 0., -1.].map(<T as Scalar>::from_f64);
    let mut logs = Vec::new();
    let mut probabilities = Vec::new();
    for row in logits.chunks_exact(3) {
        let weights = row.iter().copied().map(Float::exp).collect::<Vec<_>>();
        let total = weights
            .iter()
            .copied()
            .fold(T::zero(), |sum, value| sum + value);
        for weight in weights {
            let probability = weight / total;
            probabilities.push(probability);
            logs.push(Float::ln(probability));
        }
    }
    let seed = <T as Scalar>::from_f64(-1.5);
    let mut expected = enumerate(
        &logs,
        shape,
        CtcBatch {
            targets: &[1],
            input_lengths: &[2],
            target_lengths: &[1],
            blank: 0,
        },
        seed,
    );
    for row in expected
        .gradient
        .chunks_exact_mut(3)
        .zip(probabilities.chunks_exact(3))
    {
        let (gradient, probability) = row;
        let total = gradient
            .iter()
            .copied()
            .fold(T::zero(), |sum, value| sum + value);
        for (value, &probability) in gradient.iter_mut().zip(probability) {
            *value -= probability * total;
        }
    }
    let input = variable::<T, B>(&logits, shape);
    let log_probs = log_softmax(&input, 2);
    let loss = ctc_loss(&log_probs, &[1], &[2], &[1], 0)
        .expect("invariant: finite logits define a positive target likelihood");
    let operations = 12 * shape[0] + 2 * expected.path_count + 4 * shape[2] + 8;
    close(loss.tensor.as_slice()[0], expected.loss, operations);
    seeded_backward(&loss, seed);
    let gradient = input
        .grad()
        .expect("invariant: log-softmax propagates to its tracked logits");
    for (&actual, &expected) in gradient.as_slice().iter().zip(&expected.gradient) {
        close(actual, expected, operations);
    }
}

fn cases<T, B>()
where
    T: BinaryPrecision,
    B: BackendOps<T> + CtcOps<T> + Default + coeus_core::ComputeBackend<Error = BackendError>,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    empty_target_retains_the_blank_path_probability::<T, B>();
    exact_seed_and_accumulation::<T, B>();
    alignment_case::<T, B>(&[0.125, 0.75, 0.125], [1, 1, 3], &[1], &[1], &[1], 0);
    alignment_case::<T, B>(
        &[0.5, 0.25, 0.25, 0.25, 0.5, 0.25],
        [2, 1, 3],
        &[1],
        &[2],
        &[1],
        0,
    );
    alignment_case::<T, B>(&[0.5; 6], [3, 1, 2], &[1, 1], &[3], &[2], 0);
    alignment_case::<T, B>(
        &[0.5, 0.25, 0.25, 0.25, 0.25, 0.5],
        [2, 1, 3],
        &[0],
        &[2],
        &[1],
        2,
    );
    alignment_case::<T, B>(
        &[
            0.5,
            0.5,
            0.25,
            0.75,
            0.25,
            0.75,
            f64::NAN,
            f64::INFINITY,
            0.5,
            0.5,
            2.,
            0.5,
        ],
        [3, 2, 2],
        &[1, 1],
        &[3, 1],
        &[2, 0],
        0,
    );
    alignment_case::<T, B>(&[], [0, 1, 2], &[], &[0], &[0], 0);
    alignment_case::<T, B>(&[1., 0.], [1, 1, 2], &[], &[1], &[0], 0);
    logits_follow_the_softmax_chain_rule::<T, B>();
    invalid::cases::<T, B>();
    layouts::cases::<T, B>();
}

#[test]
fn likelihood_and_gradient_match_alignment_enumeration() {
    cases::<f32, SequentialBackend>();
    cases::<f64, SequentialBackend>();
    cases::<F16, SequentialBackend>();
    cases::<Bf16, SequentialBackend>();
    cases::<f32, MoiraiBackend>();
    cases::<f64, MoiraiBackend>();
    cases::<F16, MoiraiBackend>();
    cases::<Bf16, MoiraiBackend>();
}
