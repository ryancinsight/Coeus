//! Acceptance and rejection contracts of the existing usize-index bridge.

use coeus_leto::{spmm_into, spmv_into, CsrDispatch};
use leto::{ArrayView1, ArrayView2, Layout, LetoError};
use leto_ops::{spmm_into as owned_matrix_product, spmv_into as owned_vector_product, CsrMatrix};
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

#[test]
fn malformed_csr_and_dense_shapes_reject_before_writes() {
    let _trace = trace();
    let _case = tracing::info_span!("sparse_rejection_atomicity").entered();
    for (columns, offsets) in [
        (vec![0, 1], vec![0, 1]),
        (vec![0], vec![0, 1, 2]),
        (vec![0, 1], vec![1, 1, 2]),
        (vec![0, 1], vec![0, 3, 2]),
        (vec![0, 2], vec![0, 1, 2]),
        (vec![0, 0], vec![0, 2, 2]),
        (vec![1, 0], vec![0, 2, 2]),
    ] {
        let (mut y, mut c) = {
            let _phase = tracing::info_span!("arrange").entered();
            ([41.0_f64, 43.0], [47.0, 53.0, 59.0, 61.0])
        };
        let errors = {
            let _phase = tracing::info_span!("act").entered();
            let descriptor = CsrDispatch::new(&[2.0, 3.0], &columns, &offsets, 2, 2);
            [
                spmv_into(descriptor, &[5.0, 7.0], &mut y).expect_err("malformed CSR"),
                spmm_into(descriptor, &[5.0, 7.0, 11.0, 13.0], 2, &mut c)
                    .expect_err("malformed CSR"),
            ]
        };
        let _phase = tracing::info_span!("assert").entered();
        let original_error = CsrMatrix::from_parts(vec![2.0, 3.0], columns, offsets, 2, 2)
            .expect_err("owned validator also rejects");
        for error in errors {
            assert!(matches!(error, LetoError::StorageError { .. }));
            assert_eq!(error.to_string(), original_error.to_string());
        }
        assert_eq!(y, [41.0, 43.0]);
        assert_eq!(c, [47.0, 53.0, 59.0, 61.0]);
    }
    let _phase = tracing::info_span!("act").entered();
    let descriptor = CsrDispatch::new(&[2.0_f64], &[0], &[0, 1], 1, 2);
    let mut y = [17.0];
    let mut c = [19.0, 23.0];
    assert!(matches!(
        spmv_into(descriptor, &[3.0], &mut y),
        Err(LetoError::StorageError { .. })
    ));
    assert!(matches!(
        spmm_into(descriptor, &[3.0], 2, &mut c),
        Err(LetoError::StorageError { .. })
    ));
    assert_eq!(y, [17.0]);
    assert_eq!(c, [19.0, 23.0]);
    assert!(matches!(
        spmv_into(descriptor, &[3.0, 5.0], &mut []),
        Err(LetoError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        spmm_into(descriptor, &[3.0, 5.0, 7.0, 11.0], 2, &mut []),
        Err(LetoError::ShapeMismatch { .. })
    ));
}

#[test]
fn all_small_canonical_patterns_match_the_owned_kernel_and_dense_sum() {
    let _trace = trace();
    let _case = tracing::info_span!("exhaustive_sparse_patterns", scalar = "f64").entered();
    for rows in 0..=3 {
        for shared in 0..=3 {
            for mask in 0usize..(1usize << (rows * shared)) {
                let (values, columns, offsets, dense, expected) = {
                    let _phase = tracing::info_span!("arrange", rows, shared, mask).entered();
                    let mut values = Vec::new();
                    let mut columns = Vec::new();
                    let mut offsets = vec![0];
                    let mut dense = Vec::new();
                    for index in 0..shared * 3 {
                        dense.push(
                            f64::from(i32::try_from(index).expect("small fixture index")) - 2.0,
                        );
                    }
                    let mut expected = vec![0.0; rows * 3];
                    for row in 0..rows {
                        for column in 0..shared {
                            if mask & (1 << (row * shared + column)) != 0 {
                                let value = f64::from(
                                    i32::try_from(row + column + 1).expect("small fixture value"),
                                );
                                values.push(value);
                                columns.push(column);
                                for j in 0..3 {
                                    expected[row * 3 + j] += value * dense[column * 3 + j];
                                }
                            }
                        }
                        offsets.push(values.len());
                    }
                    (values, columns, offsets, dense, expected)
                };
                let (actual, owned, vector) = {
                    let _phase = tracing::info_span!("act").entered();
                    let descriptor = CsrDispatch::new(&values, &columns, &offsets, rows, shared);
                    let matrix = CsrMatrix::from_parts(
                        values.clone(),
                        columns.clone(),
                        offsets.clone(),
                        rows,
                        shared,
                    )
                    .expect("canonical CSR");
                    let b = ArrayView2::try_new(
                        Layout::c_contiguous([shared, 3]).expect("dense shape"),
                        &dense,
                    )
                    .expect("dense storage");
                    let x: Vec<_> = (0..shared).map(|i| dense[i * 3]).collect();
                    let x_view = ArrayView1::try_new(
                        Layout::c_contiguous([shared]).expect("vector shape"),
                        &x,
                    )
                    .expect("vector storage");
                    let mut actual = vec![f64::NAN; rows * 3];
                    let mut owned = actual.clone();
                    let mut vector = vec![f64::NAN; rows];
                    let mut owned_vector = vector.clone();
                    spmm_into(descriptor, &dense, 3, &mut actual).expect("borrowed matrix product");
                    owned_matrix_product(&matrix, &b, &mut owned).expect("owned matrix product");
                    spmv_into(descriptor, &x, &mut vector).expect("borrowed vector product");
                    owned_vector_product(&matrix, &x_view, &mut owned_vector)
                        .expect("owned vector product");
                    assert_eq!(vector, owned_vector);
                    (actual, owned, vector)
                };
                let _phase = tracing::info_span!("assert").entered();
                assert_eq!(actual, expected);
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    owned.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(
                    vector,
                    expected.iter().step_by(3).copied().collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn zero_dense_columns_and_zero_shared_dimension_are_fully_initialized() {
    let _trace = trace();
    let _case = tracing::info_span!("sparse_empty_shapes").entered();
    let mut output = [f64::NAN; 6];
    let _phase = tracing::info_span!("act").entered();
    spmm_into(
        CsrDispatch::new(&[], &[], &[0, 0, 0], 2, 0),
        &[],
        3,
        &mut output,
    )
    .expect("empty dot products");
    spmm_into(
        CsrDispatch::new(&[2.0], &[1], &[0, 1, 1], 2, 2),
        &[],
        0,
        &mut [],
    )
    .expect("zero output columns");
    let _phase = tracing::info_span!("assert").entered();
    assert_eq!(output, [0.0; 6]);
}

fn check_bits<T: leto_ops::Scalar, Bits: std::fmt::Debug + PartialEq>(
    values: &[T],
    dense: &[T],
    bits: impl Fn(T) -> Bits,
) {
    let _case =
        tracing::info_span!("special_values", scalar = std::any::type_name::<T>()).entered();
    let (matrix, descriptor, b, mut owned, mut borrowed) = {
        let _phase = tracing::info_span!("arrange").entered();
        (
            CsrMatrix::from_parts(values.to_vec(), vec![0, 1, 0, 1], vec![0, 2, 3, 4], 3, 2)
                .expect("canonical CSR"),
            CsrDispatch::new(values, &[0, 1, 0, 1], &[0, 2, 3, 4], 3, 2),
            ArrayView2::try_new(Layout::c_contiguous([2, 16]).expect("dense layout"), dense)
                .expect("dense view"),
            vec![T::ZERO; 48],
            vec![T::ZERO; 48],
        )
    };
    {
        let _phase = tracing::info_span!("act").entered();
        owned_matrix_product(&matrix, &b, &mut owned).expect("owned product");
        spmm_into(descriptor, dense, 16, &mut borrowed).expect("borrowed product");
    }
    let _phase = tracing::info_span!("assert").entered();
    assert_eq!(
        owned.into_iter().map(&bits).collect::<Vec<_>>(),
        borrowed.into_iter().map(bits).collect::<Vec<_>>()
    );
}

#[test]
fn special_values_preserve_owned_kernel_bits_in_both_native_precisions() {
    let _trace = trace();
    let mut narrow = [1.0_f32; 32];
    narrow[16..].fill(1.0 - f32::EPSILON);
    check_bits(
        &[
            -1.0_f32,
            1.0 + f32::EPSILON,
            -0.0,
            f32::from_bits(0x7fc0_0017),
        ],
        &narrow,
        f32::to_bits,
    );
    let mut wide = [1.0_f64; 32];
    wide[16..].fill(1.0 - f64::EPSILON);
    check_bits(
        &[
            -1.0_f64,
            1.0 + f64::EPSILON,
            -0.0,
            f64::from_bits(0x7ff8_0000_0000_0017),
        ],
        &wide,
        f64::to_bits,
    );
}
