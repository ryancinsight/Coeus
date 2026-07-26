use super::*;
use crate::backend::WgpuBackendError;

pub(super) fn dispatch_reduce<T: WgpuScalar>(
    op: coeus_ops::ReductionOp,
    a: &crate::backend::WgpuStorage<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut crate::backend::WgpuStorage<T>,
    c_layout: &Layout,
) -> Result<(), WgpuBackendError> {
    kernels::dispatch_reduce::<T>(op, &a.buffer, a_layout, axis, &c.buffer, c_layout)
}
