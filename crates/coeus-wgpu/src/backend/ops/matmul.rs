use crate::backend::{WgpuBackendError, WgpuScalar};
use crate::kernels;
use coeus_core::Layout;

pub(super) fn dispatch_matmul<T: WgpuScalar>(
    a: &crate::backend::WgpuStorage<T>,
    a_layout: &Layout,
    b: &crate::backend::WgpuStorage<T>,
    b_layout: &Layout,
    c: &mut crate::backend::WgpuStorage<T>,
    c_layout: &Layout,
) -> Result<(), WgpuBackendError> {
    kernels::dispatch_matmul::<T>(
        &a.buffer, a_layout, &b.buffer, b_layout, &c.buffer, c_layout,
    )
    .map_err(|error| WgpuBackendError::Layout(error.into()))
}
