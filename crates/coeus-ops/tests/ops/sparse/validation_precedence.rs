//! Multiple-fault witnesses for the existing CSR diagnostic contract.

use coeus_core::{BackendError, SequentialBackend};
use coeus_sparse::CsrTensor;
use coeus_tensor::Tensor;
use tracing_subscriber::fmt::format::FmtSpan;

#[test]
fn csr_errors_precede_dense_errors_and_keep_column_priority() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .without_time()
            .with_test_writer()
            .with_span_events(FmtSpan::FULL)
            .finish(),
    );
    let _case = tracing::info_span!("csr_diagnostic_precedence").entered();
    for (column, offsets) in [
        (-1, [1, -1]),
        (2, [0, 2]),
        (0, [1, -1]),
        (0, [0, -1]),
        (0, [0, 0]),
    ] {
        let (csr, dense) = {
            let _phase = tracing::info_span!("arrange", column, ?offsets).entered();
            (
                CsrTensor::new(
                    [1, 1].into(),
                    Tensor::<f64, SequentialBackend>::from_slice([1], &[5.0]).expect("values"),
                    Tensor::from_slice([1], &[column]).expect("column"),
                    Tensor::from_slice([2], &offsets).expect("offsets"),
                ),
                Tensor::<f64, SequentialBackend>::from_slice([], &[3.0]).expect("rank-zero dense"),
            )
        };
        let errors = {
            let _phase = tracing::info_span!("act").entered();
            [
                (
                    "sparse matrix-vector multiplication",
                    coeus_ops::spmv(&csr, &dense, &SequentialBackend),
                ),
                (
                    "sparse-dense matrix multiplication",
                    coeus_ops::spmm(&csr, &dense, &SequentialBackend),
                ),
            ]
        };
        let _phase = tracing::info_span!("assert").entered();
        for (operation, result) in errors {
            let expected = match (column, offsets[0], offsets[1]) {
                (-1, _, _) => BackendError::InvalidNumericInput {
                    operation,
                    reason: "negative CSR column index -1 at position 0".to_owned(),
                },
                (2, _, _) => BackendError::IndexOutOfRange {
                    operation,
                    position: 0,
                    index: 2,
                    bound: 1,
                },
                (_, 1, _) => BackendError::Storage {
                    operation,
                    reason: "CSR row offsets must start at zero".to_owned(),
                },
                (_, _, -1) => BackendError::InvalidNumericInput {
                    operation,
                    reason: "negative CSR row offset at position 1".to_owned(),
                },
                (_, _, 0) => BackendError::Storage {
                    operation,
                    reason: "CSR final row offset 0 does not equal nnz 1".to_owned(),
                },
                _ => unreachable!("exhaustive fixture cases"),
            };
            assert_eq!(
                result.err().expect("multiple malformed inputs reject"),
                expected
            );
        }
    }
}

#[test]
fn component_rank_rejection_precedes_invalid_indices() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .without_time()
            .with_test_writer()
            .with_span_events(FmtSpan::FULL)
            .finish(),
    );
    let _case = tracing::info_span!("csr_component_precedence").entered();
    let (csr, dense) = {
        let _phase = tracing::info_span!("arrange").entered();
        (
            CsrTensor::new(
                [1, 1].into(),
                Tensor::<f64, SequentialBackend>::from_slice([1], &[5.0]).expect("values"),
                Tensor::from_slice([1, 1], &[-1]).expect("rank-two column component"),
                Tensor::from_slice([2], &[1, -1]).expect("invalid offsets"),
            ),
            Tensor::<f64, SequentialBackend>::from_slice([1], &[3.0]).expect("dense"),
        )
    };
    let result = {
        let _phase = tracing::info_span!("act").entered();
        coeus_ops::spmv(&csr, &dense, &SequentialBackend)
    };
    let _phase = tracing::info_span!("assert").entered();
    assert_eq!(
        result.err().expect("component mismatch rejects"),
        BackendError::UnsupportedRank {
            operation: "sparse matrix-vector multiplication",
            rank: 2,
            max_rank: 1,
        }
    );
}
