use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::Layout;

fn cast_storage<T, U>(storage: &CudaStorage<T>) -> CudaStorage<U> {
    let buffer = unsafe {
        std::mem::transmute::<
            std::sync::Arc<hephaestus_cuda::CudaBuffer<T>>,
            std::sync::Arc<hephaestus_cuda::CudaBuffer<U>>,
        >(storage.buffer.clone())
    };
    CudaStorage { buffer }
}

fn cast_storage_mut<T, U>(storage: &mut CudaStorage<T>) -> CudaStorage<U> {
    let buffer = unsafe {
        std::mem::transmute::<
            std::sync::Arc<hephaestus_cuda::CudaBuffer<T>>,
            std::sync::Arc<hephaestus_cuda::CudaBuffer<U>>,
        >(storage.buffer.clone())
    };
    CudaStorage { buffer }
}

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_elementwise_binary<T: CudaScalar>(
        &self,
        op: coeus_ops::BinaryOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        b: &CudaStorage<T>,
        b_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) {
        if get_cuda_context().is_some() {
            let n = c_layout.shape().iter().product();
            // The contiguous kernel computes `c[i] = a[i] op b[i]` with no
            // broadcasting, so it is only valid when both operands already share
            // the output shape. A broadcast operand (e.g. `[3,1]` against
            // `[3,2]`) must go through the strided kernel, which resolves each
            // output coordinate against per-operand strides.
            let same_shape =
                a_layout.shape() == c_layout.shape() && b_layout.shape() == c_layout.shape();
            if same_shape
                && a_layout.is_contiguous()
                && b_layout.is_contiguous()
                && c_layout.is_contiguous()
            {
                if kernels::launch_contiguous_binary(op, a, b, c, n) {
                    return;
                }
            } else if kernels::launch_strided_binary(op, a, a_layout, b, b_layout, c, c_layout, n) {
                return;
            }
        }
        self.fallback_binary(op, a, a_layout, b, b_layout, c, c_layout);
    }

    pub(crate) fn cuda_elementwise_unary<T: CudaScalar>(
        &self,
        op: coeus_ops::UnaryOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) {
        if get_cuda_context().is_some() {
            let n = c_layout.shape().iter().product();
            if a_layout.is_contiguous() && c_layout.is_contiguous() {
                if kernels::launch_contiguous_unary(op, a, c, n) {
                    return;
                }
            } else {
                if kernels::launch_strided_unary(op, a, a_layout, c, c_layout, n) {
                    return;
                }
            }
        }
        self.fallback_unary(op, a, a_layout, c, c_layout);
    }

    pub(crate) fn cuda_matmul<T: CudaScalar>(
        &self,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        b: &CudaStorage<T>,
        b_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let a_f32 = cast_storage::<T, f32>(a);
            let b_f32 = cast_storage::<T, f32>(b);
            let mut c_f32 = cast_storage_mut::<T, f32>(c);
            if kernels::launch_matmul_tiled(
                &a_f32, &b_f32, &mut c_f32, a_layout, b_layout, c_layout,
            ) {
                return;
            }
        }
        self.fallback_matmul(a, a_layout, b, b_layout, c, c_layout);
    }

    pub(crate) fn cuda_reduce<T: CudaScalar>(
        &self,
        op: coeus_ops::ReductionOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) {
        if get_cuda_context().is_some()
            && kernels::dispatch_reduce(op, a, a_layout, axis, c, c_layout)
        {
            return;
        }
        self.fallback_reduce(op, a, a_layout, axis, c, c_layout);
    }
}
