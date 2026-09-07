//! Direct operation-boundary checks, including untouched backing-storage lanes.

use coeus_core::{BackendError, ComputeBackend, Float, Layout, Scalar};
use coeus_ops::{CtcBatch, CtcOps};

fn layout(shape: &[usize], strides: &[usize], offset: usize) -> Layout {
    Layout::from_shape_strides(shape.into(), strides.into(), offset)
}

fn sequences() -> CtcBatch<'static> {
    CtcBatch {
        targets: &[1],
        input_lengths: &[1],
        target_lengths: &[1],
        blank: 0,
    }
}

fn upload<T: Scalar, B: ComputeBackend>(backend: &B, values: &[T]) -> B::DeviceBuffer<T> {
    let mut buffer = backend.allocate(values.len());
    backend.copy_to_device(values, &mut buffer);
    buffer
}

fn exact_storage<T: Scalar, B: ComputeBackend>(
    backend: &B,
    buffer: &B::DeviceBuffer<T>,
    expected: &[T],
) {
    let mut actual = vec![T::zero(); expected.len()];
    backend.copy_to_host(buffer, &mut actual);
    assert_eq!(actual, expected);
}

fn error<T>(result: Result<T, BackendError>) -> BackendError {
    match result {
        Err(error) => error,
        Ok(_) => panic!("invalid CTC descriptor must return its typed error"),
    }
}

fn stride_count_error(
    actual: BackendError,
    expected_operation: &'static str,
    shape_count: usize,
    stride_count: usize,
) {
    match actual {
        BackendError::Storage { operation, reason } => {
            assert_eq!(operation, expected_operation);
            assert_eq!(
                reason,
                format!("shape rank {shape_count} does not match stride count {stride_count}")
            );
        }
        other => panic!("mismatched shape and stride counts must be a storage error: {other:?}"),
    }
}

pub(super) fn cases<T, B>()
where
    T: Float,
    B: CtcOps<T> + ComputeBackend<Error = BackendError> + Default,
{
    offset_storage::<T, B>();
    forward_descriptors::<T, B>();
    backward_descriptors::<T, B>();
}

fn offset_storage<T, B>()
where
    T: Float,
    B: CtcOps<T> + ComputeBackend<Error = BackendError> + Default,
{
    let backend = B::default();
    let sentinel = <T as Scalar>::from_f64(7.0);
    // Both active log probabilities are zero. Only class one can emit, so the
    // posterior and dyadic seeded update are exact in every tested format.
    let input = upload(
        &backend,
        &[<T as Float>::NAN, T::zero(), <T as Float>::NAN, T::zero()],
    );
    let mut loss = upload(&backend, &[sentinel; 3]);
    let scalar = layout(&[1], &[1], 1);
    let state = backend
        .ctc_forward(
            &input,
            &layout(&[1, 1, 2], &[4, 4, 2], 1),
            sequences(),
            &mut loss,
            &scalar,
        )
        .expect("invariant: offset input and scalar output have valid footprints");
    exact_storage(&backend, &loss, &[sentinel, T::zero(), sentinel]);
    let seed = <T as Scalar>::from_f64(2.5);
    let upstream = upload(
        &backend,
        &[
            <T as Float>::NAN,
            <T as Float>::NAN,
            seed,
            <T as Float>::NAN,
        ],
    );
    let initial = <T as Scalar>::from_f64(0.75);
    let mut gradient = upload(&backend, &[initial; 7]);
    backend
        .ctc_backward_accumulate(
            &state,
            &upstream,
            &layout(&[1], &[1], 2),
            &mut gradient,
            &layout(&[1, 1, 2], &[6, 6, 3], 2),
        )
        .expect("invariant: offset gradient is injective and fits its backing storage");
    let mut expected = [initial; 7];
    expected[5] = <T as Scalar>::from_f64(-1.75);
    exact_storage(&backend, &gradient, &expected);
}

fn forward_descriptors<T, B>()
where
    T: Float,
    B: CtcOps<T> + ComputeBackend<Error = BackendError> + Default,
{
    let backend = B::default();
    let input = upload(&backend, &[T::zero(); 2]);
    let valid = Layout::new([1, 1, 2].into());
    let scalar = Layout::new([1].into());
    let sentinel = <T as Scalar>::from_f64(7.0);
    let mut loss = upload(&backend, &[sentinel; 2]);
    for (input_layout, loss_layout, operation, shape_count, stride_count) in [
        (
            layout(&[1, 1, 2], &[], 0),
            scalar.clone(),
            "ctc_log_probs",
            3,
            0,
        ),
        (
            layout(&[1, 1, 2], &[2, 2], 0),
            scalar.clone(),
            "ctc_log_probs",
            3,
            2,
        ),
        (
            layout(&[1, 1, 2], &[2, 2, 1, 1], 0),
            scalar.clone(),
            "ctc_log_probs",
            3,
            4,
        ),
        (valid.clone(), layout(&[1], &[], 0), "ctc_loss", 1, 0),
        (valid.clone(), layout(&[1], &[1, 1], 0), "ctc_loss", 1, 2),
    ] {
        stride_count_error(
            error(backend.ctc_forward(&input, &input_layout, sequences(), &mut loss, &loss_layout)),
            operation,
            shape_count,
            stride_count,
        );
        exact_storage(&backend, &loss, &[sentinel; 2]);
    }
    for shape in [vec![2], vec![1, 2], vec![1, 1, 1, 2]] {
        let rejected = error(backend.ctc_forward(
            &input,
            &Layout::new(shape.clone().into()),
            sequences(),
            &mut loss,
            &scalar,
        ));
        match rejected {
            BackendError::LayoutRankMismatch { lhs, rhs, .. } => {
                assert_eq!((lhs, rhs), (shape.len(), 3));
            }
            other => panic!("CTC rank must be exact, without left padding: {other:?}"),
        }
        exact_storage(&backend, &loss, &[sentinel; 2]);
    }
    for shape in [vec![], vec![2], vec![1, 1]] {
        assert_eq!(
            error(backend.ctc_forward(
                &input,
                &valid,
                sequences(),
                &mut loss,
                &Layout::new(shape.clone().into()),
            )),
            BackendError::ShapeMismatch {
                operation: "ctc_loss",
                lhs: shape,
                rhs: vec![1]
            },
        );
        exact_storage(&backend, &loss, &[sentinel; 2]);
    }
    match error(backend.ctc_forward(
        &input,
        &layout(&[1, 1, 2], &[4, 4, 2], 1),
        sequences(),
        &mut loss,
        &scalar,
    )) {
        BackendError::Storage { operation, .. } => assert_eq!(operation, "ctc_log_probs"),
        other => panic!("out-of-bounds input must be a storage error: {other:?}"),
    }
    exact_storage(&backend, &loss, &[sentinel; 2]);
    match error(backend.ctc_forward(
        &input,
        &valid,
        sequences(),
        &mut loss,
        &layout(&[1], &[1], 2),
    )) {
        BackendError::Storage { operation, .. } => assert_eq!(operation, "ctc_loss"),
        other => panic!("out-of-bounds scalar output must be a storage error: {other:?}"),
    }
    exact_storage(&backend, &loss, &[sentinel; 2]);
}

fn backward_descriptors<T, B>()
where
    T: Float,
    B: CtcOps<T> + ComputeBackend<Error = BackendError> + Default,
{
    let backend = B::default();
    let input = upload(&backend, &[T::zero(); 2]);
    let valid = Layout::new([1, 1, 2].into());
    let scalar = Layout::new([1].into());
    let mut loss = upload(&backend, &[T::zero()]);
    let state = backend
        .ctc_forward(&input, &valid, sequences(), &mut loss, &scalar)
        .expect("invariant: one-frame target has a unique alignment");
    let upstream = upload(&backend, &[T::one(); 2]);
    let sentinel = <T as Scalar>::from_f64(7.0);
    let mut gradient = upload(&backend, &[sentinel; 2]);
    for (upstream_layout, gradient_layout, operation, shape_count, stride_count) in [
        (
            layout(&[1], &[], 0),
            valid.clone(),
            "ctc_output_gradient",
            1,
            0,
        ),
        (
            layout(&[1], &[1, 1], 0),
            valid.clone(),
            "ctc_output_gradient",
            1,
            2,
        ),
        (
            scalar.clone(),
            layout(&[1, 1, 2], &[], 0),
            "ctc_gradient",
            3,
            0,
        ),
        (
            scalar.clone(),
            layout(&[1, 1, 2], &[2, 2], 0),
            "ctc_gradient",
            3,
            2,
        ),
        (
            scalar.clone(),
            layout(&[1, 1, 2], &[2, 2, 1, 1], 0),
            "ctc_gradient",
            3,
            4,
        ),
    ] {
        stride_count_error(
            error(backend.ctc_backward_accumulate(
                &state,
                &upstream,
                &upstream_layout,
                &mut gradient,
                &gradient_layout,
            )),
            operation,
            shape_count,
            stride_count,
        );
        exact_storage(&backend, &gradient, &[sentinel; 2]);
    }
    for shape in [vec![], vec![2], vec![1, 1]] {
        assert_eq!(
            error(backend.ctc_backward_accumulate(
                &state,
                &upstream,
                &Layout::new(shape.clone().into()),
                &mut gradient,
                &valid,
            )),
            BackendError::ShapeMismatch {
                operation: "ctc_output_gradient",
                lhs: shape,
                rhs: vec![1],
            }
        );
        exact_storage(&backend, &gradient, &[sentinel; 2]);
    }
    for shape in [vec![2], vec![1, 2], vec![1, 1, 1, 2]] {
        match error(backend.ctc_backward_accumulate(
            &state,
            &upstream,
            &scalar,
            &mut gradient,
            &Layout::new(shape.clone().into()),
        )) {
            BackendError::LayoutRankMismatch { lhs, rhs, .. } => {
                assert_eq!((lhs, rhs), (shape.len(), 3));
            }
            other => panic!("gradient rank must be exact, without left padding: {other:?}"),
        }
        exact_storage(&backend, &gradient, &[sentinel; 2]);
    }
    assert_eq!(
        error(backend.ctc_backward_accumulate(
            &state,
            &upstream,
            &scalar,
            &mut gradient,
            &layout(&[1, 1, 2], &[0, 0, 0], 0),
        )),
        BackendError::AliasedLayout {
            operation: "ctc_backward"
        }
    );
    exact_storage(&backend, &gradient, &[sentinel; 2]);
    for (upstream_layout, gradient_layout, operation) in [
        (layout(&[1], &[1], 2), valid.clone(), "ctc_output_gradient"),
        (scalar, layout(&[1, 1, 2], &[4, 4, 2], 1), "ctc_gradient"),
    ] {
        match error(backend.ctc_backward_accumulate(
            &state,
            &upstream,
            &upstream_layout,
            &mut gradient,
            &gradient_layout,
        )) {
            BackendError::Storage {
                operation: actual, ..
            } => assert_eq!(actual, operation),
            other => panic!("out-of-bounds descriptor must be a storage error: {other:?}"),
        }
        exact_storage(&backend, &gradient, &[sentinel; 2]);
    }
}
