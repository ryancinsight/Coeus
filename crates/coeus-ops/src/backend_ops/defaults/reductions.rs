use crate::backend_ops::CpuBackend;
use coeus_core::{Layout, Scalar};

/// Default: copy to host, run `coeus_cpu::argmax_into`, copy back.
pub fn argmax<T, B>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut B::DeviceBuffer<i64>,
    c_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
{
    let mut host_a = vec![T::ZERO; a_layout.shape().iter().product()];
    backend.copy_to_host(a, &mut host_a)?;

    let mut host_c = vec![0i64; c_layout.shape().iter().product()];
    coeus_cpu::argmax_into(a_layout, &host_a, axis, c_layout, &mut host_c)
        .map_err(|error| crate::backend_ops::cpu_impl::error::map_leto_error("argmax", error))?;

    backend.copy_to_device(&host_c, c)
}

/// Default: copy to host, run `coeus_cpu::argmin_into`, copy back.
pub fn argmin<T, B>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut B::DeviceBuffer<i64>,
    c_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
{
    let mut host_a = vec![T::ZERO; a_layout.shape().iter().product()];
    backend.copy_to_host(a, &mut host_a)?;

    let mut host_c = vec![0i64; c_layout.shape().iter().product()];
    coeus_cpu::argmin_into(a_layout, &host_a, axis, c_layout, &mut host_c)
        .map_err(|error| crate::backend_ops::cpu_impl::error::map_leto_error("argmin", error))?;

    backend.copy_to_device(&host_c, c)
}

/// Default: copy to host, run `topk_impl`, copy back.
#[expect(
    clippy::too_many_arguments,
    reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097"
)]
pub fn topk<T, B>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    k: usize,
    axis: usize,
    largest: bool,
    values: &mut B::DeviceBuffer<T>,
    values_layout: &Layout,
    indices: &mut B::DeviceBuffer<i64>,
    indices_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
{
    let mut host_a = vec![T::ZERO; a_layout.shape().iter().product()];
    backend.copy_to_host(a, &mut host_a)?;

    let mut host_values = vec![T::ZERO; values_layout.shape().iter().product()];
    let mut host_indices = vec![0i64; indices_layout.shape().iter().product()];

    crate::reduction::topk::topk_impl(
        &host_a,
        a_layout.shape(),
        k,
        axis,
        largest,
        &mut host_values,
        &mut host_indices,
    );

    backend.copy_to_device(&host_values, values)?;
    backend.copy_to_device(&host_indices, indices)
}

#[cfg(test)]
mod tests {
    use coeus_core::{BackendError, CpuAddressableStorage, CpuStorage, Shape};

    use super::*;

    fn layout(dims: &[usize]) -> Layout {
        Layout::new(Shape::from(dims.to_vec()))
    }

    fn storage_f64(values: &[f64]) -> CpuStorage<f64> {
        CpuStorage::from_slice(values).expect("CPU storage allocation succeeds")
    }

    fn storage_i64(values: &[i64]) -> CpuStorage<i64> {
        CpuStorage::from_slice(values).expect("CPU storage allocation succeeds")
    }

    #[test]
    fn default_argmax_propagates_errors_before_any_write() {
        let backend = coeus_core::SequentialBackend;
        let a_layout = layout(&[2, 3]);
        let a = storage_f64(&[1.0, 4.0, -2.0, 5.0, 3.0, 6.0]);
        let c_layout = layout(&[2, 1]);

        // Out-of-range axis is a typed error, never a panic; output untouched.
        let mut c = storage_i64(&[7, 7]);
        let err = super::argmax(&backend, &a, &a_layout, 5, &mut c, &c_layout)
            .expect_err("axis 5 on a rank-2 tensor must fail");
        match err {
            BackendError::Storage { operation, reason } => {
                assert_eq!(operation, "argmax");
                assert!(reason.contains("axis"), "unexpected reason: {reason}");
            }
            other => panic!("expected Storage error, got {other:?}"),
        }
        assert_eq!(c.as_slice(), &[7, 7]);

        // Malformed output layout rejected before any write.
        let mut wrong = storage_i64(&[7, 7, 7, 7]);
        let bad_layout = layout(&[2, 2]);
        let err = super::argmax(&backend, &a, &a_layout, 1, &mut wrong, &bad_layout)
            .expect_err("wrong output shape must fail");
        assert!(
            matches!(err, BackendError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got {err:?}"
        );
        assert_eq!(wrong.as_slice(), &[7, 7, 7, 7]);

        // Valid call writes exact first-tie indices.
        let mut out = storage_i64(&[0, 0]);
        super::argmax(&backend, &a, &a_layout, 1, &mut out, &c_layout)
            .expect("valid argmax succeeds");
        assert_eq!(out.as_slice(), &[1, 2]);
    }

    #[test]
    fn default_argmin_mirrors_argmax_contract() {
        let backend = coeus_core::SequentialBackend;
        let a_layout = layout(&[2, 3]);
        let a = storage_f64(&[1.0, 4.0, -2.0, 5.0, 3.0, 6.0]);
        let c_layout = layout(&[1, 3]);

        let mut out = storage_i64(&[0, 0, 0]);
        super::argmin(&backend, &a, &a_layout, 0, &mut out, &c_layout)
            .expect("valid argmin succeeds");
        assert_eq!(out.as_slice(), &[0, 1, 0]);

        let err = super::argmin(&backend, &a, &a_layout, 2, &mut out, &c_layout)
            .expect_err("axis 2 on a rank-2 tensor must fail");
        match err {
            BackendError::Storage { operation, reason } => {
                assert_eq!(operation, "argmin");
                assert!(reason.contains("axis"), "unexpected reason: {reason}");
            }
            other => panic!("expected Storage error, got {other:?}"),
        }
    }
}
