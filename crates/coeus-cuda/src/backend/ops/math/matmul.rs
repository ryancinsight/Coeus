use super::super::cast::{cast_storage, cast_storage_mut};
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::Layout;

impl CudaBackend {
    pub(crate) fn cuda_matmul<T: CudaScalar>(
        &self,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        b: &CudaStorage<T>,
        b_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) -> Result<(), crate::CudaBackendError> {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let a_f32 = cast_storage::<T, f32>(a);
            let b_f32 = cast_storage::<T, f32>(b);
            let mut c_f32 = cast_storage_mut::<T, f32>(c);
            if kernels::launch_matmul_tiled(
                &a_f32, &b_f32, &mut c_f32, a_layout, b_layout, c_layout,
            ) {
                return Ok(());
            }
        }
        Err(crate::CudaBackendError::kernel(
            "matrix multiplication",
            "native CUDA dispatch requirements are not satisfied",
        ))
    }
}
