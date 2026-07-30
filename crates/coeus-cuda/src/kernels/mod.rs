/// Kernel module for scaled-dot-product attention operations.
pub mod attention;
/// Kernel module for fused element-wise expression compilation and dispatch.
pub mod fuse;
/// Kernel module for tiled matrix multiplication kernel launch.
pub mod launch_matmul;
/// Kernel module for element-wise operator kernel launches.
pub mod launch_ops;
/// Kernel module for optimizer step kernels.
pub mod optim;
/// Kernel module for pooling operations.
pub mod pool;
/// Kernel module for embedded PTX kernel source.
pub mod ptx;
/// Kernel module for reduction operations.
pub mod reduce;
/// Kernel module for sliding-window unfold and adjoint fold operations.
pub mod unfold_fold;
mod validation;

pub(crate) use validation::{checked_numel, layout_fits_cuda_storage};

pub use attention::{launch_sdp_attention, launch_sdp_attention_backward};
pub use fuse::dispatch_fused;
pub use launch_matmul::launch_matmul_tiled;
pub use launch_ops::{
    launch_contiguous_binary, launch_contiguous_unary, launch_strided_binary, launch_strided_unary,
};
pub use optim::{
    launch_adagrad_step, launch_adam_step, launch_adamw_step, launch_rmsprop_step, launch_sgd_step,
};
pub use pool::{
    dispatch_avg_pool1d, dispatch_avg_pool1d_backward, dispatch_avg_pool2d,
    dispatch_avg_pool2d_backward, dispatch_avg_pool3d, dispatch_avg_pool3d_backward,
    dispatch_max_pool1d, dispatch_max_pool1d_backward, dispatch_max_pool2d,
    dispatch_max_pool2d_backward, dispatch_max_pool3d, dispatch_max_pool3d_backward,
};
pub use reduce::{dispatch_fused_reduce, dispatch_reduce};
pub use unfold_fold::{dispatch_fold1d, dispatch_fold2d, dispatch_unfold1d, dispatch_unfold2d};

use crate::driver::{get_cuda_context, CUfunction, CUmodule, CudaDriver};
use crate::kernels::ptx::PTX_SOURCE;
use coeus_core::Layout;
use std::sync::OnceLock;

/// Maximum rank representable by the CUDA layout descriptor.
pub(crate) const CUDA_LAYOUT_MAX_DIMS: usize = 8;

/// Failure while converting a host layout to the CUDA `u32` ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum CudaLayoutError {
    /// The layout has more dimensions than the CUDA descriptor can encode.
    #[error("layout rank {0} exceeds the CUDA limit of {CUDA_LAYOUT_MAX_DIMS}")]
    RankTooLarge(usize),
    /// The layout shape and stride vectors have different lengths.
    #[error("layout shape rank {shape} differs from stride rank {strides}")]
    RankMismatch { shape: usize, strides: usize },
    /// A layout value does not fit in the CUDA descriptor's `u32` field.
    #[error("layout {field} value {value} exceeds the CUDA u32 ABI")]
    ValueTooLarge {
        /// Name of the descriptor field.
        field: &'static str,
        /// Host-side value that failed conversion.
        value: usize,
    },
}

/// GPU-side layout descriptor passed to CUDA kernels as a POD struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuLayoutInfo {
    /// Element offset into the underlying buffer.
    pub(crate) offset: u32,
    /// Number of dimensions in the layout.
    pub(crate) ndim: u32,
    /// Per-dimension shape, padded to 8 entries.
    pub(crate) shape: [u32; CUDA_LAYOUT_MAX_DIMS],
    /// Per-dimension strides, padded to 8 entries.
    pub(crate) strides: [u32; CUDA_LAYOUT_MAX_DIMS],
}

impl TryFrom<&Layout> for GpuLayoutInfo {
    type Error = CudaLayoutError;

    fn try_from(layout: &Layout) -> Result<Self, Self::Error> {
        let shape = layout.shape();
        let strides = layout.strides();
        if shape.len() != strides.len() {
            return Err(CudaLayoutError::RankMismatch {
                shape: shape.len(),
                strides: strides.len(),
            });
        }
        if shape.len() > CUDA_LAYOUT_MAX_DIMS {
            return Err(CudaLayoutError::RankTooLarge(shape.len()));
        }

        let mut gpu_shape = [0u32; CUDA_LAYOUT_MAX_DIMS];
        let mut gpu_strides = [0u32; CUDA_LAYOUT_MAX_DIMS];
        for ((gpu_shape, &shape), (gpu_stride, &stride)) in gpu_shape
            .iter_mut()
            .zip(shape)
            .zip(gpu_strides.iter_mut().zip(strides))
        {
            *gpu_shape = u32::try_from(shape).map_err(|_| CudaLayoutError::ValueTooLarge {
                field: "shape",
                value: shape,
            })?;
            *gpu_stride = u32::try_from(stride).map_err(|_| CudaLayoutError::ValueTooLarge {
                field: "stride",
                value: stride,
            })?;
        }

        Ok(Self {
            offset: u32::try_from(layout.offset()).map_err(|_| CudaLayoutError::ValueTooLarge {
                field: "offset",
                value: layout.offset(),
            })?,
            ndim: u32::try_from(shape.len()).map_err(|_| CudaLayoutError::ValueTooLarge {
                field: "rank",
                value: shape.len(),
            })?,
            shape: gpu_shape,
            strides: gpu_strides,
        })
    }
}

struct CudaModuleWrapper {
    module: CUmodule,
}

impl Drop for CudaModuleWrapper {
    fn drop(&mut self) {
        if !self.module.is_null() {
            if let Some(drv) = CudaDriver::get() {
                unsafe {
                    (drv.cu_module_unload)(self.module);
                }
            }
        }
    }
}

unsafe impl Send for CudaModuleWrapper {}
unsafe impl Sync for CudaModuleWrapper {}

static CUDA_MODULE: OnceLock<Option<CudaModuleWrapper>> = OnceLock::new();

/// Retrieve the lazily-loaded CUDA module singleton containing the embedded PTX kernels.
pub fn get_cuda_module() -> Option<CUmodule> {
    CUDA_MODULE
        .get_or_init(|| {
            let drv = CudaDriver::get()?;
            let _ctx = get_cuda_context()?;

            let ptx_src = format!("{}\0", PTX_SOURCE);
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                let res = (drv.cu_module_load_data)(
                    &mut module,
                    ptx_src.as_ptr() as *const std::ffi::c_void,
                );
                if res == 0 {
                    Some(CudaModuleWrapper { module })
                } else {
                    panic!("cu_module_load_data failed with error code: {}", res);
                }
            }
        })
        .as_ref()
        .map(|wrapper| wrapper.module)
}

/// Look up a kernel function by name from the loaded CUDA module.
pub fn get_cuda_function(name: &str) -> Option<CUfunction> {
    let drv = CudaDriver::get()?;
    let module = get_cuda_module()?;
    let c_name = std::ffi::CString::new(name).ok()?;
    let mut func: CUfunction = std::ptr::null_mut();
    unsafe {
        let res = (drv.cu_module_get_function)(&mut func, module, c_name.as_ptr());
        if res == 0 {
            Some(func)
        } else {
            None
        }
    }
}

/// Launch a kernel over a flat 1-D grid of `total` threads (256/block).
///
/// Shared by the NVRTC-compiled kernels (attention, conv_transpose) that map
/// one thread to one output element. Returns `false` if the driver is absent or
/// the launch fails so the operation boundary can report the dispatch failure.
pub(crate) fn launch_1d(
    func: CUfunction,
    total: usize,
    args: &mut [*mut std::ffi::c_void],
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(grid_size) = crate::kernels::validation::launch_grid_size(total) else {
        return false;
    };
    unsafe {
        let res = (drv.cu_launch_kernel)(
            func,
            grid_size,
            1,
            1,
            crate::kernels::validation::CUDA_BLOCK_SIZE,
            1,
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
    use super::{CudaLayoutError, GpuLayoutInfo, CUDA_LAYOUT_MAX_DIMS};
    use coeus_core::Layout;

    #[test]
    fn converts_representable_layout_without_changing_values() {
        let layout = Layout::from_shape_strides(vec![2, 3].into(), vec![3, 1].into(), 4);

        let gpu_layout = GpuLayoutInfo::try_from(&layout).expect("representable layout");

        assert_eq!(gpu_layout.offset, 4);
        assert_eq!(gpu_layout.ndim, 2);
        assert_eq!(&gpu_layout.shape[..2], &[2, 3]);
        assert_eq!(&gpu_layout.strides[..2], &[3, 1]);
    }

    #[test]
    fn rejects_layouts_with_unsupported_rank() {
        let layout = Layout::new(vec![1; CUDA_LAYOUT_MAX_DIMS + 1].into());

        assert!(matches!(
            GpuLayoutInfo::try_from(&layout),
            Err(CudaLayoutError::RankTooLarge(rank))
                if rank == CUDA_LAYOUT_MAX_DIMS + 1
        ));
    }

    #[test]
    fn rejects_layouts_with_mismatched_shape_and_stride_rank() {
        let layout = Layout::from_shape_strides(vec![2, 3].into(), vec![3].into(), 0);

        assert!(matches!(
            GpuLayoutInfo::try_from(&layout),
            Err(CudaLayoutError::RankMismatch {
                shape: 2,
                strides: 1,
            })
        ));
    }

    #[test]
    fn rejects_layout_values_outside_the_cuda_abi() {
        let oversized = usize::try_from(u64::from(u32::MAX) + 1)
            .expect("CUDA ABI boundary test requires a 64-bit host");
        let layout = Layout::from_shape_strides(vec![1].into(), vec![1].into(), oversized);

        assert!(matches!(
            GpuLayoutInfo::try_from(&layout),
            Err(CudaLayoutError::ValueTooLarge {
                field: "offset",
                value,
            }) if value == oversized
        ));
    }
}
