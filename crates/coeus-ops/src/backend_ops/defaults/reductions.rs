use crate::backend_ops::CpuBackend;
use coeus_core::{Layout, Scalar};

/// Default: copy to host, run `coeus_leto::argmax_into`, copy back.
pub fn argmax<T, B>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut B::DeviceBuffer<i64>,
    c_layout: &Layout,
) where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
{
    let mut host_a = vec![T::zero(); a_layout.shape().iter().product()];
    backend.copy_to_host(a, &mut host_a);

    let mut host_c = vec![0i64; c_layout.shape().iter().product()];
    coeus_leto::argmax_into(a_layout, &host_a, axis, c_layout, &mut host_c)
        .expect("argmax default impl failed");

    backend.copy_to_device(&host_c, c);
}

/// Default: copy to host, run `coeus_leto::argmin_into`, copy back.
pub fn argmin<T, B>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut B::DeviceBuffer<i64>,
    c_layout: &Layout,
) where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
{
    let mut host_a = vec![T::zero(); a_layout.shape().iter().product()];
    backend.copy_to_host(a, &mut host_a);

    let mut host_c = vec![0i64; c_layout.shape().iter().product()];
    coeus_leto::argmin_into(a_layout, &host_a, axis, c_layout, &mut host_c)
        .expect("argmin default impl failed");

    backend.copy_to_device(&host_c, c);
}

/// Default: copy to host, run `topk_impl`, copy back.
#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
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
) where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
{
    let mut host_a = vec![T::zero(); a_layout.shape().iter().product()];
    backend.copy_to_host(a, &mut host_a);

    let mut host_values = vec![T::zero(); values_layout.shape().iter().product()];
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

    backend.copy_to_device(&host_values, values);
    backend.copy_to_device(&host_indices, indices);
}
