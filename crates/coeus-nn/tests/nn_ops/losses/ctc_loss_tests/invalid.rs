use super::{
    count, ctc_loss, probability_logs, variable, BackendError, BackendOps, BinaryPrecision, CtcOps,
    Float, Scalar, Tensor,
};

fn rejected<T, B>(
    logs: &[T],
    targets: &[usize],
    input_lengths: &[usize],
    target_lengths: &[usize],
    blank: usize,
) -> BackendError
where
    T: BinaryPrecision,
    B: BackendOps<T> + CtcOps<T> + Default + coeus_core::ComputeBackend<Error = BackendError>,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    let input = variable::<T, B>(logs, [2, 1, 2]);
    let initial = <T as Scalar>::from_f64(0.75);
    input.set_grad(Tensor::from_slice([2, 1, 2], &[initial; 4]));
    let error = match ctc_loss(&input, targets, input_lengths, target_lengths, blank) {
        Err(error) => error,
        Ok(_) => panic!("invalid CTC fixture must return a typed error"),
    };
    assert_eq!(
        input
            .grad()
            .expect("invariant: prefilled gradient remains present")
            .as_slice(),
        &[initial; 4]
    );
    error
}

pub(super) fn cases<T, B>()
where
    T: BinaryPrecision,
    B: BackendOps<T> + CtcOps<T> + Default + coeus_core::ComputeBackend<Error = BackendError>,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    let logs = probability_logs::<T>(&[0.5; 4]);
    assert_eq!(
        rejected::<T, B>(&logs, &[1], &[], &[1], 0),
        BackendError::SequenceLengthCounts {
            batch: 1,
            inputs: 0,
            targets: 1
        }
    );
    assert_eq!(
        rejected::<T, B>(&logs, &[1], &[2], &[], 0),
        BackendError::SequenceLengthCounts {
            batch: 1,
            inputs: 1,
            targets: 0
        }
    );
    assert_eq!(
        rejected::<T, B>(&logs, &[1], &[3], &[1], 0),
        BackendError::SequenceInputLength {
            sample: 0,
            actual: 3,
            maximum: 2
        }
    );
    for (target, blank, expected_label) in [(1, 2, 2), (0, 0, 0), (2, 0, 2)] {
        assert_eq!(
            rejected::<T, B>(&logs, &[target], &[2], &[1], blank),
            BackendError::SequenceLabel {
                label: expected_label,
                blank,
                classes: 2
            }
        );
    }
    match rejected::<T, B>(&logs, &[1], &[2], &[2], 0) {
        BackendError::ShapeMismatch { lhs, rhs, .. } => assert_eq!((lhs, rhs), (vec![1], vec![2])),
        error => panic!("concatenated target mismatch must retain lengths: {error:?}"),
    }
    for bad in [<T as Float>::NAN, <T as Float>::INFINITY, T::one()] {
        let mut invalid = logs.clone();
        invalid[1] = bad;
        match rejected::<T, B>(&invalid, &[1], &[2], &[1], 0) {
            BackendError::InvalidLogProbability { index, .. } => assert_eq!(index, [0, 0, 1]),
            error => panic!("invalid active log probability must retain its index: {error:?}"),
        }
    }
    impossible_paths_preserve_existing_gradients::<T, B>();
}

fn impossible_paths_preserve_existing_gradients<T, B>()
where
    T: BinaryPrecision,
    B: BackendOps<T> + CtcOps<T> + Default + coeus_core::ComputeBackend<Error = BackendError>,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    // Sample zero is feasible. Sample one needs a blank between repeated labels
    // and cannot fit in two frames. Rejection must precede *all* accumulation.
    let input = variable::<T, B>(&probability_logs::<T>(&[0.5; 8]), [2, 2, 2]);
    let initial = <T as Scalar>::from_f64(0.75);
    input.set_grad(Tensor::from_slice([2, 2, 2], &[initial; 8]));
    let loss = ctc_loss(&input, &[1, 1, 1], &[2, 2], &[1, 2], 0)
        .expect("invariant: impossible alignments have a defined infinite forward loss");
    assert_eq!(loss.tensor.as_slice(), &[<T as Float>::INFINITY]);
    match loss.backward_with_seed(Tensor::from_slice([1], &[count::<T>(2)])) {
        Err(BackendError::UndefinedGradient { sample, .. }) => assert_eq!(sample, 1),
        result => panic!("impossible path must reject its derivative: {result:?}"),
    }
    assert_eq!(
        input
            .grad()
            .expect("invariant: prefilled gradient remains present")
            .as_slice(),
        &[initial; 8]
    );

    // A structurally feasible target with zero probability is also undefined.
    let input = variable::<T, B>(&[T::zero(), <T as Float>::NEG_INFINITY], [1, 1, 2]);
    input.set_grad(Tensor::from_slice([1, 1, 2], &[initial; 2]));
    let loss = ctc_loss(&input, &[1], &[1], &[1], 0)
        .expect("invariant: zero-probability paths retain infinite forward loss");
    assert_eq!(loss.tensor.as_slice(), &[<T as Float>::INFINITY]);
    match loss.backward() {
        Err(BackendError::UndefinedGradient { sample, .. }) => assert_eq!(sample, 0),
        result => panic!("zero-probability path must reject its derivative: {result:?}"),
    }
    assert_eq!(
        input
            .grad()
            .expect("invariant: prefilled gradient remains present")
            .as_slice(),
        &[initial; 2]
    );
}
