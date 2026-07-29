use super::GpuLayoutInfo;
use crate::driver::{CudaDriver, get_cuda_context};
use crate::kernels::validation::{launch_grid_size_for_block, layout_fits_cuda_storage};
use crate::storage::CudaStorage;
use coeus_core::{Layout, Storage};

fn layouts_fit_storage(
    a_layout: &Layout,
    a_len: usize,
    b_layout: &Layout,
    b_len: usize,
    c_layout: &Layout,
    c_len: usize,
) -> bool {
    layout_fits_cuda_storage(a_layout, a_len, false)
        && layout_fits_cuda_storage(b_layout, b_len, false)
        && layout_fits_cuda_storage(c_layout, c_len, true)
}

/// Launch the tiled matrix multiplication kernel on the GPU.
///
/// Computes `c = a × b` using the PTX-compiled tiled matmul kernel. Returns `true`
/// if the kernel launched successfully, `false` if the driver or context is unavailable.
pub fn launch_matmul_tiled(
    a: &CudaStorage<f32>,
    b: &CudaStorage<f32>,
    c: &mut CudaStorage<f32>,
    a_layout: &Layout,
    b_layout: &Layout,
    c_layout: &Layout,
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(_ctx) = get_cuda_context() else {
        return false;
    };
    if !layouts_fit_storage(a_layout, a.len(), b_layout, b.len(), c_layout, c.len()) {
        return false;
    }
    let [m, k] = a_layout.shape() else {
        return false;
    };
    let [b_k, n] = b_layout.shape() else {
        return false;
    };
    let [c_m, c_n] = c_layout.shape() else {
        return false;
    };
    if *m == 0
        || *k == 0
        || *b_k == 0
        || *n == 0
        || *c_m == 0
        || *c_n == 0
        || *k != *b_k
        || *m != *c_m
        || *n != *c_n
    {
        return false;
    }
    let Some(grid_x) = launch_grid_size_for_block(*n, 16) else {
        return false;
    };
    let Some(grid_y) = launch_grid_size_for_block(*m, 16) else {
        return false;
    };

    let cuda_src = r#"
struct GpuLayoutInfo {
    unsigned int offset;
    unsigned int ndim;
    unsigned int shape[8];
    unsigned int strides[8];
};

extern "C" __global__ void matmul_kernel(
    const float* a,
    GpuLayoutInfo a_layout,
    const float* b,
    GpuLayoutInfo b_layout,
    float* c,
    GpuLayoutInfo c_layout
) {
    __shared__ float A_shared[256];
    __shared__ float B_shared[256];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int dx = blockDim.x;
    unsigned int dy = blockDim.y;

    unsigned int col = bx * dx + tx;
    unsigned int row = by * dy + ty;

    unsigned int m = a_layout.shape[0];
    unsigned int k = a_layout.shape[1];
    unsigned int n = b_layout.shape[1];

    unsigned int stride_a_row = a_layout.strides[0];
    unsigned int stride_a_col = a_layout.strides[1];
    unsigned int stride_b_row = b_layout.strides[0];
    unsigned int stride_b_col = b_layout.strides[1];

    float sum = 0.0f;

    unsigned int num_tiles = (k + 15) / 16;
    unsigned int local_idx = ty * 16 + tx;

    for (unsigned int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        unsigned int col_a = tile_idx * 16 + tx;
        float val_a = 0.0f;
        if (row < m && col_a < k) {
            unsigned int offset_a = a_layout.offset + row * stride_a_row + col_a * stride_a_col;
            val_a = a[offset_a];
        }
        A_shared[local_idx] = val_a;

        unsigned int row_b = tile_idx * 16 + ty;
        float val_b = 0.0f;
        if (row_b < k && col < n) {
            unsigned int offset_b = b_layout.offset + row_b * stride_b_row + col * stride_b_col;
            val_b = b[offset_b];
        }
        B_shared[local_idx] = val_b;

        __syncthreads();

        for (unsigned int i = 0; i < 16; ++i) {
            sum += A_shared[ty * 16 + i] * B_shared[i * 16 + tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        unsigned int stride_c_row = c_layout.strides[0];
        unsigned int stride_c_col = c_layout.strides[1];
        unsigned int offset_c = c_layout.offset + row * stride_c_row + col * stride_c_col;
        c[offset_c] = sum;
    }
}

"#;

    let key = "matmul_tiled_f32".to_string();
    let Some(kernel) = super::fuse::get_or_create_kernel(&key, cuda_src, "matmul_kernel") else {
        return false;
    };

    let Ok(gpu_a_layout) = GpuLayoutInfo::try_from(a_layout) else {
        return false;
    };
    let Ok(gpu_b_layout) = GpuLayoutInfo::try_from(b_layout) else {
        return false;
    };
    let Ok(gpu_c_layout) = GpuLayoutInfo::try_from(c_layout) else {
        return false;
    };

    let mut a_ptr = a.cu_deviceptr();
    let mut b_ptr = b.cu_deviceptr();
    let mut c_ptr = c.cu_deviceptr();

    let mut args: [*mut std::ffi::c_void; 6] = [
        &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
        &gpu_a_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
        &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
        &gpu_b_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
        &mut c_ptr as *mut u64 as *mut std::ffi::c_void,
        &gpu_c_layout as *const GpuLayoutInfo as *mut std::ffi::c_void,
    ];

    unsafe {
        let res = (drv.cu_launch_kernel)(
            kernel.func,
            grid_x,
            grid_y,
            1,
            16,
            16,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        res == 0
    }
}

#[cfg(test)]
mod tests {
    use super::layouts_fit_storage;
    use coeus_core::Layout;

    #[test]
    fn matmul_rejects_each_layout_that_exceeds_storage() {
        let a = Layout::new([2, 3].into());
        let b = Layout::new([3, 4].into());
        let c = Layout::new([2, 4].into());

        assert!(!layouts_fit_storage(&a, 5, &b, 12, &c, 8));
        assert!(!layouts_fit_storage(&a, 6, &b, 11, &c, 8));
        assert!(!layouts_fit_storage(&a, 6, &b, 12, &c, 7));
        assert!(layouts_fit_storage(&a, 6, &b, 12, &c, 8));
    }

    #[test]
    fn matmul_rejects_writable_zero_stride_output() {
        let a = Layout::new([2, 3].into());
        let b = Layout::new([3, 4].into());
        let c = Layout::from_shape_strides([2, 4].into(), vec![0, 1].into(), 0);

        assert!(!layouts_fit_storage(&a, 6, &b, 12, &c, 4));
    }
}
