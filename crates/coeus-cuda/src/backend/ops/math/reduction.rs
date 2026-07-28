use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::Layout;

impl CudaBackend {
    pub(crate) fn cuda_reduce<T: CudaScalar>(
        &self,
        op: coeus_ops::ReductionOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) -> Result<(), crate::CudaBackendError> {
        if get_cuda_context().is_some()
            && kernels::dispatch_reduce(op, a, a_layout, axis, c, c_layout)
        {
            return Ok(());
        }
        self.fallback_reduce(op, a, a_layout, axis, c, c_layout)
    }
}
