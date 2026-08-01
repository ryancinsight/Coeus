use super::source::SOURCE;
use crate::backend::CudaScalar;

pub(super) fn kernel<T: CudaScalar>(
    name: &str,
) -> Option<std::sync::Arc<crate::kernels::fuse::SafeCachedKernel>> {
    let source = SOURCE.replace("{TYPE}", T::CUDA_TYPE);
    crate::kernels::fuse::get_or_create_kernel(
        &format!("unfold_fold_{name}_{}", T::CUDA_TYPE),
        &source,
        name,
    )
}

pub(super) fn launch(
    _name: &str,
    total: usize,
    function: crate::driver::CUfunction,
    args: &mut [*mut std::ffi::c_void],
) -> bool {
    crate::kernels::launch_1d(function, total, args)
}
