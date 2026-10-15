use coeus_core::{Backend, CpuAddressableStorageMut, MoiraiBackend, Scalar, SequentialBackend};
use coeus_sparse::CsrTensor;
use coeus_tensor::Tensor;
use eunomia::{Bf16, Complex, F16};
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

fn scalar_family<T: Scalar>() {
    let _case =
        tracing::info_span!("forward_scalar", scalar = std::any::type_name::<T>()).entered();
    let (a, x, b, two, three) = {
        let _phase = tracing::info_span!("arrange").entered();
        let two = T::ONE + T::ONE;
        (
            CsrTensor::<T, SequentialBackend>::new(
                [2, 2].into(),
                Tensor::from_slice([3], &[T::ONE; 3]).expect("values"),
                Tensor::from_slice([3], &[1i64, 0, 1]).expect("columns"),
                Tensor::from_slice([3], &[0i64, 3, 3]).expect("offsets"),
            ),
            Tensor::from_slice([2], &[T::ONE; 2]).expect("vector"),
            Tensor::from_slice([2, 2], &[T::ONE, T::ZERO, T::ONE, T::ONE]).expect("matrix"),
            two,
            two + T::ONE,
        )
    };
    let (vector, matrix) = {
        let _phase = tracing::info_span!("act").entered();
        (
            coeus_ops::spmv(&a, &x, &SequentialBackend).expect("vector output"),
            coeus_ops::spmm(&a, &b, &SequentialBackend).expect("matrix output"),
        )
    };
    let _phase = tracing::info_span!("assert").entered();
    assert_eq!(vector.as_slice(), &[three, T::ZERO]);
    assert_eq!(matrix.as_slice(), &[three, two, T::ZERO, T::ZERO]);
}

#[test]
fn public_forward_api_retains_every_scalar_family() {
    let _trace = trace();
    scalar_family::<u8>();
    scalar_family::<u16>();
    scalar_family::<u32>();
    scalar_family::<u64>();
    scalar_family::<i8>();
    scalar_family::<i16>();
    scalar_family::<i32>();
    scalar_family::<i64>();
    scalar_family::<f32>();
    scalar_family::<f64>();
    scalar_family::<F16>();
    scalar_family::<Bf16>();
    scalar_family::<Complex<f32>>();
    scalar_family::<Complex<f64>>();
    scalar_family::<Complex<F16>>();
    scalar_family::<Complex<Bf16>>();
}

fn wide_product<B>(backend: &B)
where
    B: Backend,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorageMut<i64>,
{
    let _case = tracing::info_span!("wide_forward", backend = std::any::type_name::<B>()).entered();
    let (a, dense, expected) = {
        let _phase = tracing::info_span!("arrange").entered();
        let (rows, width) = (4097, 257);
        let mut values = Vec::new();
        let mut columns = Vec::new();
        let mut offsets = vec![0i64];
        let mut expected = vec![0.0; rows * width];
        let dense: Vec<_> = (0..2 * width)
            .map(|p| f64::from(i32::try_from(p % 19).expect("small residue")) / 8.0 - 1.0)
            .collect();
        for row in 0..rows {
            if row % 7 != 0 {
                let scale = f64::from(i32::try_from(row % 5 + 1).expect("small residue")) / 8.0;
                values.extend([scale, 2.0 * scale, 3.0 * scale]);
                columns.extend([1i64, 0, 1]);
                for column in 0..width {
                    // The dyadic expression is exact, so both stored and
                    // closed-form evaluation give identical binary64 bits.
                    expected[row * width + column] =
                        2.0 * scale * dense[column] + 4.0 * scale * dense[width + column];
                }
            }
            offsets.push(i64::try_from(values.len()).expect("test-sized CSR"));
        }
        let a = CsrTensor::<f64, B>::new(
            [rows, 2].into(),
            Tensor::from_slice_on([values.len()], &values, backend).expect("values"),
            Tensor::from_slice_on([columns.len()], &columns, backend).expect("columns"),
            Tensor::from_slice_on([offsets.len()], &offsets, backend).expect("offsets"),
        );
        (
            a,
            Tensor::from_slice_on([2, width], &dense, backend).expect("dense"),
            expected,
        )
    };
    let output = {
        let _phase = tracing::info_span!("act").entered();
        coeus_ops::spmm(&a, &dense, backend).expect("wide product")
    };
    let _phase = tracing::info_span!("assert").entered();
    assert_eq!(output.shape(), &[4097, 257]);
    for (position, (&actual, &expected)) in output.as_slice().iter().zip(&expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "output slot {position}"
        );
    }
}

#[test]
fn wide_output_preserves_backend_parallelism_and_initializes_every_slot() {
    let _trace = trace();
    wide_product(&SequentialBackend);
    wide_product(&MoiraiBackend);
}
