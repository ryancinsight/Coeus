use coeus_core::{Scalar, Layout, Backend, CpuAddressableStorage, CpuAddressableStorageMut, Shape, Strides};
use crate::ptr::{Ptr, MutPtr};
use crate::backend_ops::ReductionOp;
use crate::backend_ops::compute_reduction_base_offset;

#[inline]
pub(crate) fn matmul<T: Scalar, B: Backend>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    b: &B::DeviceBuffer<T>,
    b_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    _c_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    assert_eq!(a_layout.shape().len(), 2, "matmul requires 2D input A");
    assert_eq!(b_layout.shape().len(), 2, "matmul requires 2D input B");

    let m = a_layout.shape()[0];
    let k = a_layout.shape()[1];
    let k2 = b_layout.shape()[0];
    let n = b_layout.shape()[1];
    assert_eq!(k, k2, "matmul inner dimension mismatch: {} vs {}", k, k2);

    let a_slice = a.as_slice();
    let b_slice = b.as_slice();
    let c_slice = c.as_mut_slice();

    let a_ptr = Ptr(a_slice.as_ptr());
    let b_ptr = Ptr(b_slice.as_ptr());
    let c_ptr = MutPtr(c_slice.as_mut_ptr());

    let stride_a_row = a_layout.strides()[0];
    let stride_a_col = a_layout.strides()[1];
    let stride_b_row = b_layout.strides()[0];
    let stride_b_col = b_layout.strides()[1];
    let offset_a = a_layout.offset();
    let offset_b = b_layout.offset();

    if stride_a_col == 1 && stride_b_col == 1 {
        // Highly vectorizable fast path
        backend.parallel_for(0, m, move |row| {
            let out_row_offset = row * n;
            let a_row_offset = offset_a + row * stride_a_row;
            for col in 0..n {
                unsafe {
                    c_ptr.write(out_row_offset + col, T::zero());
                }
            }
            for i in 0..k {
                let va = unsafe { a_ptr.read(a_row_offset + i) };
                if va == T::zero() {
                    continue;
                }
                let b_row_offset = offset_b + i * stride_b_row;
                for col in 0..n {
                    let vb = unsafe { b_ptr.read(b_row_offset + col) };
                    unsafe {
                        let out_off = out_row_offset + col;
                        let current = c_ptr.read(out_off);
                        c_ptr.write(out_off, current + va * vb);
                    }
                }
            }
        });
        return;
    }

    backend.parallel_for(0, m, move |row| {
        let out_row_offset = row * n;
        let a_row_offset = offset_a + row * stride_a_row;
        for col in 0..n {
            unsafe {
                c_ptr.write(out_row_offset + col, T::zero());
            }
        }
        for i in 0..k {
            let va = unsafe { a_ptr.read(a_row_offset + i * stride_a_col) };
            if va == T::zero() {
                continue;
            }
            let b_row_offset = offset_b + i * stride_b_row;
            for col in 0..n {
                let vb = unsafe { b_ptr.read(b_row_offset + col * stride_b_col) };
                unsafe {
                    let out_off = out_row_offset + col;
                    let current = c_ptr.read(out_off);
                    c_ptr.write(out_off, current + va * vb);
                }
            }
        }
    });
}

pub trait ReductionKernelOp<T: Scalar> {
    fn combine(x: T, y: T) -> T;
}

pub struct SumOp;
impl<T: Scalar> ReductionKernelOp<T> for SumOp {
    #[inline(always)]
    fn combine(x: T, y: T) -> T { x + y }
}

pub struct MaxOp;
impl<T: Scalar> ReductionKernelOp<T> for MaxOp {
    #[inline(always)]
    fn combine(x: T, y: T) -> T { if x > y { x } else { y } }
}

pub struct MinOp;
impl<T: Scalar> ReductionKernelOp<T> for MinOp {
    #[inline(always)]
    fn combine(x: T, y: T) -> T { if x < y { x } else { y } }
}

#[inline(always)]
fn run_reduction_op<T: Scalar, B: Backend, O: ReductionKernelOp<T>>(
    backend: &B,
    a_ptr: Ptr<T>,
    c_ptr: MutPtr<T>,
    axis_len: usize,
    out_numel: usize,
    out_strides_vec: Strides,
    a_shape_vec: Shape,
    a_strides_vec: Strides,
    a_off: usize,
    axis: usize,
) {
    backend.parallel_for(0, out_numel, move |i| {
        let base_off_a = compute_reduction_base_offset(
            i,
            &out_strides_vec,
            &a_shape_vec,
            &a_strides_vec,
            a_off,
            axis,
        );

        let stride_axis = a_strides_vec[axis];
        let mut acc = unsafe { a_ptr.read(base_off_a) };
        for k in 1..axis_len {
            let val = unsafe { a_ptr.read(base_off_a + k * stride_axis) };
            acc = O::combine(acc, val);
        }

        unsafe {
            c_ptr.write(i, acc);
        }
    });
}

#[inline]
pub(crate) fn reduce<T: Scalar, B: Backend>(
    backend: &B,
    op: ReductionOp,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let a_slice = a.as_slice();
    let c_slice = c.as_mut_slice();

    let a_ptr = Ptr(a_slice.as_ptr());
    let c_ptr = MutPtr(c_slice.as_mut_ptr());

    let a_shape_v = a_layout.shape();
    let axis_len = a_shape_v[axis];

    let out_shape = c_layout.shape();
    let out_numel = out_shape.iter().product::<usize>();

    if axis_len == 0 {
        backend.parallel_for(0, out_numel, move |i| unsafe {
            c_ptr.write(i, T::zero());
        });
        return;
    }

    let a_off = a_layout.offset();

    let out_strides_vec = c_layout.strides_cloned();
    let a_strides_vec = a_layout.strides_cloned();
    let a_shape_vec = a_layout.shape_cloned();

    match op {
        ReductionOp::Sum => {
            run_reduction_op::<T, B, SumOp>(
                backend,
                a_ptr,
                c_ptr,
                axis_len,
                out_numel,
                out_strides_vec,
                a_shape_vec,
                a_strides_vec,
                a_off,
                axis,
            );
        }
        ReductionOp::Max => {
            run_reduction_op::<T, B, MaxOp>(
                backend,
                a_ptr,
                c_ptr,
                axis_len,
                out_numel,
                out_strides_vec,
                a_shape_vec,
                a_strides_vec,
                a_off,
                axis,
            );
        }
        ReductionOp::Min => {
            run_reduction_op::<T, B, MinOp>(
                backend,
                a_ptr,
                c_ptr,
                axis_len,
                out_numel,
                out_strides_vec,
                a_shape_vec,
                a_strides_vec,
                a_off,
                axis,
            );
        }
    }
}
