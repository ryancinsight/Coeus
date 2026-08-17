//! Differential verification of the CPU matmul `BackendOps` path.
//!
//! `SequentialBackend` and `MoiraiBackend` delegate 2-D matmul through
//! `coeus-leto::matmul_into`. The reference below is an independent row-major
//! triple loop, and the inputs are small integers exactly representable in
//! `f32`/`f64`; the products and sums stay below the exact-integer range, so
//! bitwise equality is the correct oracle for both scalar widths.

use coeus_core::Layout;
use coeus_core::{
    ComputeBackend, CpuAddressableStorageMut, MoiraiBackend, Scalar, SequentialBackend, Shape,
    Strides,
};
use coeus_ops::backend_ops::MatmulOps;
use coeus_ops::CpuBackend;

fn layout(shape: &[usize]) -> Layout {
    Layout::new(Shape::from(shape.to_vec()))
}

fn matmul_reference<T: Scalar>(a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T> {
    let mut out = vec![T::zero(); m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = T::zero();
            for inner in 0..k {
                acc += a[row * k + inner] * b[inner * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn device_matmul<T, B>(
    backend: &B,
    a: &[T],
    a_layout: &Layout,
    b: &[T],
    b_layout: &Layout,
    c_layout: &Layout,
) -> Vec<T>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let mut a_buffer = ComputeBackend::allocate::<T>(backend, a.len());
    let mut b_buffer = ComputeBackend::allocate::<T>(backend, b.len());
    let mut c_buffer = ComputeBackend::allocate::<T>(backend, c_layout.numel());

    backend.copy_to_device(a, &mut a_buffer);
    backend.copy_to_device(b, &mut b_buffer);
    backend
        .matmul(
            &a_buffer,
            a_layout,
            &b_buffer,
            b_layout,
            &mut c_buffer,
            c_layout,
        )
        .expect("valid matmul test layouts");

    let mut out = vec![T::zero(); c_layout.numel()];
    backend.copy_to_host(&c_buffer, &mut out);
    out
}

fn check_contiguous_matmul<T, B>(backend: &B)
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let a: Vec<T> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        .into_iter()
        .map(T::from_f64)
        .collect();
    let b: Vec<T> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        .into_iter()
        .map(T::from_f64)
        .collect();
    let expected = matmul_reference(&a, 2, 3, &b, 2);

    let got = device_matmul(
        backend,
        &a,
        &layout(&[2, 3]),
        &b,
        &layout(&[3, 2]),
        &layout(&[2, 2]),
    );

    assert_same_bits(&got, &expected);
}

fn check_transposed_input_matmul<T, B>(backend: &B)
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let a_storage: Vec<T> = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        .into_iter()
        .map(T::from_f64)
        .collect();
    let a_logical: Vec<T> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        .into_iter()
        .map(T::from_f64)
        .collect();
    let b: Vec<T> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        .into_iter()
        .map(T::from_f64)
        .collect();
    let a_transposed_layout = Layout::from_shape_strides(
        Shape::from(vec![2usize, 3]),
        Strides::from_slice(&[1usize, 2]),
        0,
    );
    let expected = matmul_reference(&a_logical, 2, 3, &b, 2);

    let got = device_matmul(
        backend,
        &a_storage,
        &a_transposed_layout,
        &b,
        &layout(&[3, 2]),
        &layout(&[2, 2]),
    );

    assert_same_bits(&got, &expected);
}

fn assert_same_bits<T: Scalar>(got: &[T], expected: &[T]) {
    assert_eq!(got.len(), expected.len());
    for (index, (&actual, &reference)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            Scalar::to_f64(actual).to_bits(),
            Scalar::to_f64(reference).to_bits(),
            "matmul mismatch at index {index}",
        );
    }
}

#[test]
fn sequential_matmul_matches_reference() {
    let backend = SequentialBackend;
    check_contiguous_matmul::<f32, _>(&backend);
    check_contiguous_matmul::<f64, _>(&backend);
    check_transposed_input_matmul::<f32, _>(&backend);
    check_transposed_input_matmul::<f64, _>(&backend);
}

#[test]
fn moirai_matmul_matches_reference() {
    let backend = MoiraiBackend;
    check_contiguous_matmul::<f32, _>(&backend);
    check_contiguous_matmul::<f64, _>(&backend);
    check_transposed_input_matmul::<f32, _>(&backend);
    check_transposed_input_matmul::<f64, _>(&backend);
}

#[test]
fn test_parallel_matmul_loop() {
    let backend = MoiraiBackend;
    let m = 256;
    let k = 256;
    let n = 256;
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];

    for _i in 0..100 {
        let got = device_matmul(
            &backend,
            &a,
            &layout(&[m, k]),
            &b,
            &layout(&[k, n]),
            &layout(&[m, n]),
        );
        assert_eq!(got.len(), m * n);
    }
}
