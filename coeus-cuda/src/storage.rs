use crate::driver::{get_cuda_context, CUdeviceptr, CudaDriver};
use coeus_core::{Scalar, Storage, StorageMut};
use half::{bf16, f16};
use std::marker::PhantomData;
use std::sync::Arc;

/// RAII wrapper for raw CUDA memory allocations.
pub struct CudaAllocation {
    pub(crate) ptr: CUdeviceptr,
}

unsafe impl Send for CudaAllocation {}
unsafe impl Sync for CudaAllocation {}

impl Drop for CudaAllocation {
    fn drop(&mut self) {
        if self.ptr != 0 {
            if let Some(drv) = CudaDriver::get() {
                unsafe {
                    let _res = (drv.cu_mem_free)(self.ptr);
                }
            }
        }
    }
}

/// Statically typed enum representing the underlying CUDA memory buffer.
/// Eliminates type erasure and Box/Arc allocation of dynamic traits.
#[derive(Clone)]
pub enum CudaBuffer {
    TensorF32(Arc<cutile::tensor::Tensor<f32>>, Arc<CudaAllocation>),
    TensorF64(Arc<cutile::tensor::Tensor<f64>>, Arc<CudaAllocation>),
    TensorF16(Arc<cutile::tensor::Tensor<f16>>, Arc<CudaAllocation>),
    TensorBF16(Arc<cutile::tensor::Tensor<bf16>>, Arc<CudaAllocation>),
    TensorI32(Arc<cutile::tensor::Tensor<i32>>, Arc<CudaAllocation>),
    Fallback(
        Arc<cuda_async::device_buffer::DeviceBuffer>,
        Arc<CudaAllocation>,
    ),
    Null(Arc<CudaAllocation>),
}

impl CudaBuffer {
    /// Return the enum value as &dyn Any for compatibility downcasting.
    pub fn as_any(&self) -> &dyn std::any::Any {
        match self {
            CudaBuffer::TensorF32(t, _) => t.as_ref() as &dyn std::any::Any,
            CudaBuffer::TensorF64(t, _) => t.as_ref() as &dyn std::any::Any,
            CudaBuffer::TensorF16(t, _) => t.as_ref() as &dyn std::any::Any,
            CudaBuffer::TensorBF16(t, _) => t.as_ref() as &dyn std::any::Any,
            CudaBuffer::TensorI32(t, _) => t.as_ref() as &dyn std::any::Any,
            CudaBuffer::Fallback(b, _) => b.as_ref() as &dyn std::any::Any,
            CudaBuffer::Null(alloc) => alloc.as_ref() as &dyn std::any::Any,
        }
    }
}

/// Device storage allocated on an NVIDIA GPU.
pub struct CudaStorage<T> {
    pub(crate) buffer: CudaBuffer,
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<T>,
}

impl<T> coeus_core::storage::private::Sealed for CudaStorage<T> {}

impl<T> Clone for CudaStorage<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            len: self.len,
            _marker: PhantomData,
        }
    }
}

unsafe impl<T: Send> Send for CudaStorage<T> {}
unsafe impl<T: Sync> Sync for CudaStorage<T> {}

impl<T: Scalar> CudaStorage<T> {
    /// Allocate a new GPU device buffer.
    pub fn new(len: usize) -> Self {
        let size_in_bytes = (len * std::mem::size_of::<T>()).max(4);
        let mut ptr: CUdeviceptr = 0;

        if let Some(drv) = CudaDriver::get() {
            if get_cuda_context().is_some() {
                unsafe {
                    let res = (drv.cu_mem_alloc)(&mut ptr, size_in_bytes);
                    if res != 0 {
                        ptr = 0;
                    }
                }
            }
        }

        let alloc = Arc::new(CudaAllocation { ptr });
        let buffer = if ptr == 0 {
            CudaBuffer::Null(alloc)
        } else {
            let type_id = std::any::TypeId::of::<T>();
            if type_id == std::any::TypeId::of::<f32>() {
                let tensor = unsafe {
                    cutile::tensor::Tensor::<f32>::from_raw_parts(
                        ptr,
                        size_in_bytes,
                        0, // Default device id is 0
                        vec![len as i32],
                        vec![1],
                    )
                };
                CudaBuffer::TensorF32(Arc::new(tensor), alloc)
            } else if type_id == std::any::TypeId::of::<f64>() {
                let tensor = unsafe {
                    cutile::tensor::Tensor::<f64>::from_raw_parts(
                        ptr,
                        size_in_bytes,
                        0,
                        vec![len as i32],
                        vec![1],
                    )
                };
                CudaBuffer::TensorF64(Arc::new(tensor), alloc)
            } else if type_id == std::any::TypeId::of::<f16>() {
                let tensor = unsafe {
                    cutile::tensor::Tensor::<f16>::from_raw_parts(
                        ptr,
                        size_in_bytes,
                        0,
                        vec![len as i32],
                        vec![1],
                    )
                };
                CudaBuffer::TensorF16(Arc::new(tensor), alloc)
            } else if type_id == std::any::TypeId::of::<bf16>() {
                let tensor = unsafe {
                    cutile::tensor::Tensor::<bf16>::from_raw_parts(
                        ptr,
                        size_in_bytes,
                        0,
                        vec![len as i32],
                        vec![1],
                    )
                };
                CudaBuffer::TensorBF16(Arc::new(tensor), alloc)
            } else if type_id == std::any::TypeId::of::<i32>() {
                let tensor = unsafe {
                    cutile::tensor::Tensor::<i32>::from_raw_parts(
                        ptr,
                        size_in_bytes,
                        0,
                        vec![len as i32],
                        vec![1],
                    )
                };
                CudaBuffer::TensorI32(Arc::new(tensor), alloc)
            } else {
                let buffer = unsafe {
                    cuda_async::device_buffer::DeviceBuffer::from_raw_parts(ptr, size_in_bytes, 0)
                };
                CudaBuffer::Fallback(Arc::new(buffer), alloc)
            }
        };

        Self {
            buffer,
            len,
            _marker: PhantomData,
        }
    }

    /// Retrieve the raw CUDA device pointer.
    #[inline]
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        match &self.buffer {
            CudaBuffer::TensorF32(_, alloc) => alloc.ptr,
            CudaBuffer::TensorF64(_, alloc) => alloc.ptr,
            CudaBuffer::TensorF16(_, alloc) => alloc.ptr,
            CudaBuffer::TensorBF16(_, alloc) => alloc.ptr,
            CudaBuffer::TensorI32(_, alloc) => alloc.ptr,
            CudaBuffer::Fallback(_, alloc) => alloc.ptr,
            CudaBuffer::Null(alloc) => alloc.ptr,
        }
    }
}

impl<T: Scalar + cuda_core::DType> CudaStorage<T> {
    /// Access the underlying `cutile` Tensor.
    #[inline]
    pub fn as_cutile_tensor(&self) -> &cutile::tensor::Tensor<T> {
        self.buffer
            .as_any()
            .downcast_ref::<cutile::tensor::Tensor<T>>()
            .expect("Downcast to cutile::tensor::Tensor failed")
    }
}

impl<T: Scalar> Storage<T> for CudaStorage<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn allocate(len: usize) -> Self {
        Self::new(len)
    }

    #[inline]
    fn try_as_slice(&self) -> Option<&[T]> {
        None
    }
}

impl<T: Scalar> StorageMut<T> for CudaStorage<T> {
    #[inline]
    fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        None
    }

    #[inline]
    fn make_unique(&mut self) {
        let is_shared = match &self.buffer {
            CudaBuffer::TensorF32(_, alloc) => Arc::strong_count(alloc) > 1,
            CudaBuffer::TensorF64(_, alloc) => Arc::strong_count(alloc) > 1,
            CudaBuffer::TensorF16(_, alloc) => Arc::strong_count(alloc) > 1,
            CudaBuffer::TensorBF16(_, alloc) => Arc::strong_count(alloc) > 1,
            CudaBuffer::TensorI32(_, alloc) => Arc::strong_count(alloc) > 1,
            CudaBuffer::Fallback(_, alloc) => Arc::strong_count(alloc) > 1,
            CudaBuffer::Null(_) => false,
        };

        if is_shared {
            let size_in_bytes = (self.len * std::mem::size_of::<T>()).max(4);
            let mut new_ptr: CUdeviceptr = 0;
            if let Some(drv) = CudaDriver::get() {
                if get_cuda_context().is_some() {
                    unsafe {
                        let res = (drv.cu_mem_alloc)(&mut new_ptr, size_in_bytes);
                        if res == 0 && new_ptr != 0 {
                            let old_ptr = self.cu_deviceptr();
                            let _copy_res = (drv.cu_memcpy_dtod)(new_ptr, old_ptr, size_in_bytes);
                        } else {
                            new_ptr = 0;
                        }
                    }
                }
            }

            if new_ptr != 0 {
                let alloc = Arc::new(CudaAllocation { ptr: new_ptr });
                let type_id = std::any::TypeId::of::<T>();
                self.buffer = if type_id == std::any::TypeId::of::<f32>() {
                    let tensor = unsafe {
                        cutile::tensor::Tensor::<f32>::from_raw_parts(
                            new_ptr,
                            size_in_bytes,
                            0,
                            vec![self.len as i32],
                            vec![1],
                        )
                    };
                    CudaBuffer::TensorF32(Arc::new(tensor), alloc)
                } else if type_id == std::any::TypeId::of::<f64>() {
                    let tensor = unsafe {
                        cutile::tensor::Tensor::<f64>::from_raw_parts(
                            new_ptr,
                            size_in_bytes,
                            0,
                            vec![self.len as i32],
                            vec![1],
                        )
                    };
                    CudaBuffer::TensorF64(Arc::new(tensor), alloc)
                } else if type_id == std::any::TypeId::of::<f16>() {
                    let tensor = unsafe {
                        cutile::tensor::Tensor::<f16>::from_raw_parts(
                            new_ptr,
                            size_in_bytes,
                            0,
                            vec![self.len as i32],
                            vec![1],
                        )
                    };
                    CudaBuffer::TensorF16(Arc::new(tensor), alloc)
                } else if type_id == std::any::TypeId::of::<bf16>() {
                    let tensor = unsafe {
                        cutile::tensor::Tensor::<bf16>::from_raw_parts(
                            new_ptr,
                            size_in_bytes,
                            0,
                            vec![self.len as i32],
                            vec![1],
                        )
                    };
                    CudaBuffer::TensorBF16(Arc::new(tensor), alloc)
                } else if type_id == std::any::TypeId::of::<i32>() {
                    let tensor = unsafe {
                        cutile::tensor::Tensor::<i32>::from_raw_parts(
                            new_ptr,
                            size_in_bytes,
                            0,
                            vec![self.len as i32],
                            vec![1],
                        )
                    };
                    CudaBuffer::TensorI32(Arc::new(tensor), alloc)
                } else {
                    let buffer = unsafe {
                        cuda_async::device_buffer::DeviceBuffer::from_raw_parts(
                            new_ptr,
                            size_in_bytes,
                            0,
                        )
                    };
                    CudaBuffer::Fallback(Arc::new(buffer), alloc)
                };
            }
        }
    }
}
