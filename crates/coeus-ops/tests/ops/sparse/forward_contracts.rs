//! Contracts that a provider-owned forward implementation must preserve.

use coeus_core::{BackendError, CpuAddressableStorage, Layout, SequentialBackend};
use coeus_sparse::CsrTensor;
use coeus_tensor::Tensor;
use tracing_subscriber::fmt::format::FmtSpan;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .without_time()
            .with_test_writer()
            .with_span_events(FmtSpan::FULL)
            .finish(),
    )
}

fn tensor(shape: &[usize], data: &[f64]) -> Tensor<f64, SequentialBackend> {
    Tensor::from_slice(shape.to_vec(), data).expect("valid test tensor")
}

fn csr(
    shape: [usize; 2],
    values: &[f64],
    columns: &[i64],
    offsets: &[i64],
) -> CsrTensor<f64, SequentialBackend> {
    CsrTensor::new(
        shape.into(),
        tensor(&[values.len()], values),
        Tensor::from_slice([columns.len()], columns).expect("index tensor"),
        Tensor::from_slice([offsets.len()], offsets).expect("offset tensor"),
    )
}

#[test]
fn stored_order_and_duplicate_columns_remain_observable() {
    let _trace = trace();
    let _case = tracing::info_span!("stored_order", scalar = "f64").entered();
    let (a, x, b) = {
        let _phase = tracing::info_span!("arrange").entered();
        // ((2^53 + 1) - 2^53) rounds to 0; coalescing column 1 changes it to 1.
        let a = csr(
            [2, 2],
            &[2.0_f64.powi(53), 1.0, -2.0_f64.powi(53)],
            &[1, 0, 1],
            &[0, 3, 3],
        );
        (a, tensor(&[2], &[1.0, 1.0]), tensor(&[2, 2], &[1.0; 4]))
    };
    let (y, c) = {
        let _phase = tracing::info_span!("act").entered();
        (
            coeus_ops::spmv(&a, &x, &SequentialBackend).expect("stored-order SpMV"),
            coeus_ops::spmm(&a, &b, &SequentialBackend).expect("stored-order SpMM"),
        )
    };
    let _phase = tracing::info_span!("assert").entered();
    assert_eq!(y.as_slice(), &[0.0, 0.0]);
    assert_eq!(c.as_slice(), &[0.0; 4]);
    assert!(y.as_slice().iter().all(|value| value.to_bits() == 0));
    let error = leto_ops::CsrView::from_slices(
        &[2.0_f64.powi(53), 1.0, -2.0_f64.powi(53)],
        &[1, 0, 1],
        &[0, 3, 3],
        2,
        2,
    )
    .expect_err("current Leto requires strictly increasing row columns");
    assert!(
        matches!(error, leto::LetoError::StorageError { reason } if reason.contains("strictly increasing"))
    );
}

#[test]
fn matrix_product_keeps_separate_product_and_sum_rounding() {
    let _trace = trace();
    let _case = tracing::info_span!("separate_rounding", scalar = "f64").entered();
    let (a, b) = {
        let _phase = tracing::info_span!("arrange").entered();
        // (1+2^-52)(1-2^-52) = 1-2^-104 rounds to 1 before adding -1.
        let a = csr([1, 2], &[-1.0, 1.0 + f64::EPSILON], &[0, 1], &[0, 2]);
        let mut dense = [1.0; 32];
        dense[16..].fill(1.0 - f64::EPSILON);
        (a, tensor(&[2, 16], &dense))
    };
    let c = {
        let _phase = tracing::info_span!("act").entered();
        coeus_ops::spmm(&a, &b, &SequentialBackend).expect("valid rounded product")
    };
    let _phase = tracing::info_span!("assert").entered();
    assert_eq!(c.shape(), &[1, 16]);
    assert!(c.as_slice().iter().all(|value| value.to_bits() == 0));
    assert_eq!(
        (1.0 + f64::EPSILON).mul_add(1.0 - f64::EPSILON, -1.0),
        -2.0_f64.powi(-104)
    );
}

#[test]
fn invalid_signed_indices_return_typed_errors() {
    let _trace = trace();
    let _case = tracing::info_span!("signed_index_validation").entered();
    let (x, b) = {
        let _phase = tracing::info_span!("arrange").entered();
        (tensor(&[1], &[2.0]), tensor(&[1, 1], &[3.0]))
    };
    for (column, offsets) in [(-1, [0, 1]), (i64::MAX, [0, 1]), (0, [0, -1]), (0, [0, 2])] {
        let a = {
            let _phase = tracing::info_span!("arrange", column).entered();
            csr([1, 1], &[5.0], &[column], &offsets)
        };
        let errors = {
            let _phase = tracing::info_span!("act").entered();
            [
                coeus_ops::spmv(&a, &x, &SequentialBackend)
                    .err()
                    .expect("invalid CSR must reject"),
                coeus_ops::spmm(&a, &b, &SequentialBackend)
                    .err()
                    .expect("invalid CSR must reject"),
            ]
        };
        let _phase = tracing::info_span!("assert").entered();
        for error in errors {
            match (column, offsets[1], error) {
                (-1, _, BackendError::InvalidNumericInput { reason, .. }) => {
                    assert!(reason.contains("negative CSR column index"))
                }
                (
                    i64::MAX,
                    _,
                    BackendError::IndexOutOfRange {
                        position, bound, ..
                    },
                ) => assert_eq!((position, bound), (0, 1)),
                (_, -1, BackendError::InvalidNumericInput { reason, .. }) => {
                    assert!(reason.contains("negative CSR row offset"))
                }
                (_, 2, BackendError::Storage { reason, .. }) => {
                    assert!(reason.contains("outside 0..=1"))
                }
                (_, _, error) => panic!("unexpected rejection: {error:?}"),
            }
        }
    }
}

#[test]
fn dense_layout_rejections_and_contiguous_offsets_stay_typed() {
    let _trace = trace();
    let _case = tracing::info_span!("dense_layout_validation").entered();
    let (a, mut x, mut b) = {
        let _phase = tracing::info_span!("arrange").entered();
        (
            csr([1, 2], &[2.0, 3.0], &[0, 1], &[0, 2]),
            tensor(&[4], &[99.0, 1.0, 2.0, 99.0]),
            tensor(&[6], &[99.0, 1.0, 2.0, 3.0, 4.0, 99.0]),
        )
    };
    *x.storage_and_layout_mut().1 = Layout::from_shape_strides([2].into(), vec![1].into(), 1);
    *b.storage_and_layout_mut().1 = Layout::from_shape_strides([2, 2].into(), vec![2, 1].into(), 1);
    let (y, c) = {
        let _phase = tracing::info_span!("act").entered();
        (
            coeus_ops::spmv(&a, &x, &SequentialBackend).expect("offset vector"),
            coeus_ops::spmm(&a, &b, &SequentialBackend).expect("offset matrix"),
        )
    };
    {
        let _phase = tracing::info_span!("assert").entered();
        assert_eq!(y.as_slice(), &[8.0]);
        assert_eq!(c.as_slice(), &[11.0, 16.0]);
        assert_eq!(x.storage().as_slice(), &[99.0, 1.0, 2.0, 99.0]);
    }
    *x.storage_and_layout_mut().1 = Layout::from_shape_strides([2].into(), vec![2].into(), 0);
    *b.storage_and_layout_mut().1 = Layout::from_shape_strides([2, 2].into(), vec![1, 2].into(), 1);
    let (vector_error, matrix_error) = {
        let _phase = tracing::info_span!("act").entered();
        (
            coeus_ops::spmv(&a, &x, &SequentialBackend)
                .err()
                .expect("strided vector rejects"),
            coeus_ops::spmm(&a, &b, &SequentialBackend)
                .err()
                .expect("transposed matrix rejects"),
        )
    };
    let _phase = tracing::info_span!("assert").entered();
    for error in [vector_error, matrix_error] {
        assert!(
            matches!(error, BackendError::Storage { reason, .. } if reason.contains("contiguous"))
        );
    }
}

#[test]
fn empty_dimensions_initialize_every_output() {
    let _trace = trace();
    let _case = tracing::info_span!("empty_dimensions").entered();
    for (rows, shared, columns) in [(0, 0, 0), (0, 2, 3), (3, 0, 2), (3, 2, 0), (3, 2, 4)] {
        let (a, x, b) = {
            let _phase = tracing::info_span!("arrange", rows, shared, columns).entered();
            (
                csr([rows, shared], &[], &[], &vec![0; rows + 1]),
                tensor(&[shared], &vec![7.0; shared]),
                tensor(&[shared, columns], &vec![11.0; shared * columns]),
            )
        };
        let (y, c) = {
            let _phase = tracing::info_span!("act").entered();
            (
                coeus_ops::spmv(&a, &x, &SequentialBackend).expect("empty SpMV"),
                coeus_ops::spmm(&a, &b, &SequentialBackend).expect("empty SpMM"),
            )
        };
        let _phase = tracing::info_span!("assert").entered();
        assert_eq!(y.shape(), &[rows]);
        assert_eq!(c.shape(), &[rows, columns]);
        assert_eq!(y.as_slice(), vec![0.0; rows]);
        assert_eq!(c.as_slice(), vec![0.0; rows * columns]);
    }
}
