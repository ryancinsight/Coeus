use crate::fuse::expr_node::{CachedTensor, ExprNode, CPU_EVAL_CACHE};
use crate::ptr::MutPtr;
use coeus_core::{Backend, Layout, Scalar, Storage, StorageMut};
use coeus_tensor::Tensor;
use std::marker::PhantomData;

#[inline(always)]
fn logical_to_coords(mut temp: usize, ndim: usize, out_strides: &[usize], coords: &mut [usize]) {
    match ndim {
        0 => {}
        1 => {
            coords[0] = temp;
        }
        2 => {
            let s0 = out_strides[0];
            coords[0] = temp / s0;
            coords[1] = temp % s0;
        }
        3 => {
            let s0 = out_strides[0];
            let s1 = out_strides[1];
            coords[0] = temp / s0;
            let rem0 = temp % s0;
            coords[1] = rem0 / s1;
            coords[2] = rem0 % s1;
        }
        4 => {
            let s0 = out_strides[0];
            let s1 = out_strides[1];
            let s2 = out_strides[2];
            coords[0] = temp / s0;
            let rem0 = temp % s0;
            coords[1] = rem0 / s1;
            let rem1 = rem0 % s1;
            coords[2] = rem1 / s2;
            coords[3] = rem1 % s2;
        }
        _ => {
            for d in 0..ndim {
                coords[d] = temp / out_strides[d];
                temp %= out_strides[d];
            }
        }
    }
}

// ── CPU evaluator ──

struct CachedInputs<T> {
    addresses: Vec<usize>,
    _marker: PhantomData<T>,
}

impl<T> CachedInputs<T> {
    fn new<E, B>(expr: &E, backend: &B) -> Self
    where
        E: ExprNode<T, B>,
        T: Scalar,
        B: Backend,
    {
        let mut inputs = Vec::new();
        expr.collect_inputs(&mut inputs);

        let mut addresses = Vec::new();
        for &input_ptr in &inputs {
            unsafe {
                let tensor = &*input_ptr;
                if tensor.storage().try_as_slice().is_none() {
                    let contiguous = tensor.to_contiguous_on(backend);
                    let mut host_data = vec![T::zero(); contiguous.numel()];
                    backend.copy_to_host(contiguous.storage(), &mut host_data);
                    let cached_tensor = CachedTensor {
                        data: host_data,
                        layout: contiguous.layout().clone(),
                    };
                    let bytes = Box::new(cached_tensor) as Box<dyn std::any::Any>;
                    let addr = input_ptr as usize;
                    CPU_EVAL_CACHE.with(|cache| {
                        cache.borrow_mut().insert(addr, bytes);
                    });
                    addresses.push(addr);
                }
            }
        }

        Self {
            addresses,
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for CachedInputs<T> {
    fn drop(&mut self) {
        for addr in self.addresses.drain(..) {
            CPU_EVAL_CACHE.with(|cache| {
                cache.borrow_mut().remove(&addr);
            });
        }
    }
}

fn write_fused_values<E, T, B>(
    expr: E,
    out_ptr: MutPtr<T>,
    out_numel: usize,
    out_layout: &Layout,
    contiguous_fast_path: bool,
    backend: &B,
) where
    E: ExprNode<T, B> + Copy + Send,
    T: Scalar,
    B: Backend,
{
    let ndim = out_layout.ndim();
    let out_strides = out_layout.strides_cloned();

    if contiguous_fast_path {
        backend.parallel_for(0, out_numel, move |idx| unsafe {
            let val = expr.eval_cpu_flat(idx);
            out_ptr.write(idx, val);
        });
    } else {
        backend.parallel_for(0, out_numel, move |idx| {
            if ndim <= 8 {
                let mut coords = [0usize; 8];
                logical_to_coords(idx, ndim, &out_strides, &mut coords[..ndim]);
                unsafe {
                    let val = expr.eval_cpu(&coords[..ndim]);
                    out_ptr.write(idx, val);
                }
            } else {
                let mut coords = smallvec::SmallVec::<[usize; 16]>::from_elem(0, ndim);
                logical_to_coords(idx, ndim, &out_strides, &mut coords);
                unsafe {
                    let val = expr.eval_cpu(&coords);
                    out_ptr.write(idx, val);
                }
            }
        });
    }
}

#[inline(always)]
fn reduce_pair<T: Scalar>(op: crate::ReductionOp, acc: T, val: T) -> T {
    match op {
        crate::ReductionOp::Sum | crate::ReductionOp::Mean => acc + val,
        crate::ReductionOp::Max => {
            if val > acc {
                val
            } else {
                acc
            }
        }
        crate::ReductionOp::Min => {
            if val < acc {
                val
            } else {
                acc
            }
        }
    }
}

#[derive(Clone, Copy)]
struct FusedReductionPlan<'a> {
    out_layout: &'a Layout,
    axis: usize,
    axis_len: usize,
    op: crate::ReductionOp,
}

unsafe fn eval_reduction_at<E, T, B>(
    expr: E,
    coords: &mut [usize],
    axis: usize,
    axis_len: usize,
    op: crate::ReductionOp,
) -> T
where
    E: ExprNode<T, B> + Copy,
    T: Scalar,
    B: Backend,
{
    coords[axis] = 0;
    let mut acc = expr.eval_cpu(coords);

    for k in 1..axis_len {
        coords[axis] = k;
        let val = expr.eval_cpu(coords);
        acc = reduce_pair(op, acc, val);
    }

    if matches!(op, crate::ReductionOp::Mean) {
        acc / T::from_f64(axis_len as f64)
    } else {
        acc
    }
}

fn write_fused_reductions<E, T, B>(
    expr: E,
    out_ptr: MutPtr<T>,
    out_numel: usize,
    plan: FusedReductionPlan<'_>,
    backend: &B,
) where
    E: ExprNode<T, B> + Copy + Send,
    T: Scalar,
    B: Backend,
{
    let ndim = plan.out_layout.ndim();
    let out_strides = plan.out_layout.strides_cloned();

    backend.parallel_for(0, out_numel, move |idx| {
        if ndim <= 8 {
            let mut coords = [0usize; 8];
            logical_to_coords(idx, ndim, &out_strides, &mut coords[..ndim]);
            unsafe {
                let acc =
                    eval_reduction_at(expr, &mut coords[..ndim], plan.axis, plan.axis_len, plan.op);
                out_ptr.write(idx, acc);
            }
        } else {
            let mut coords = smallvec::SmallVec::<[usize; 16]>::from_elem(0, ndim);
            logical_to_coords(idx, ndim, &out_strides, &mut coords);
            unsafe {
                let acc = eval_reduction_at(expr, &mut coords, plan.axis, plan.axis_len, plan.op);
                out_ptr.write(idx, acc);
            }
        }
    });
}

/// Evaluate a fused expression DAG on the CPU, returning a new tensor with the result.
pub fn evaluate_fused_cpu<E: ExprNode<T, B> + Copy + Send, T: Scalar, B: Backend>(
    expr: &E,
    backend: &B,
) -> Tensor<T, B> {
    let out_shape = expr
        .shape()
        .expect("Fused expression must have at least one tensor input to determine shape");
    let out_layout = Layout::new(out_shape.clone());
    let mut out = Tensor::zeros_on(out_shape, backend);

    let out_numel = out.numel();
    let _cached_inputs = CachedInputs::<T>::new(expr, backend);

    // 2. Perform parallel evaluation
    let contiguous_fast_path = expr.is_contiguous_and_same_shape(out_layout.shape());
    let slice_result = out.storage_mut().try_as_mut_slice();
    if let Some(slice) = slice_result {
        // CPU output fast path
        let out_ptr = MutPtr(slice.as_mut_ptr());
        write_fused_values(
            *expr,
            out_ptr,
            out_numel,
            &out_layout,
            contiguous_fast_path,
            backend,
        );
    } else {
        // GPU output fallback path: evaluate on a temporary CPU buffer and copy back to GPU
        let mut host_out = vec![T::zero(); out_numel];
        let out_ptr = MutPtr(host_out.as_mut_ptr());
        write_fused_values(
            *expr,
            out_ptr,
            out_numel,
            &out_layout,
            contiguous_fast_path,
            backend,
        );
        backend.copy_to_device(&host_out, out.storage_mut());
    }

    out
}

/// Evaluate a fused expression DAG with a reduction along `axis` on the CPU.
pub fn evaluate_fused_reduce_cpu<E: ExprNode<T, B> + Copy + Send, T: Scalar, B: Backend>(
    expr: &E,
    op: crate::ReductionOp,
    axis: usize,
    backend: &B,
) -> Tensor<T, B> {
    let expr_shape = expr
        .shape()
        .expect("Fused expression must have at least one tensor input to determine shape");
    assert!(
        axis < expr_shape.len(),
        "Axis out of bounds in evaluate_fused_reduce_cpu"
    );

    let mut out_shape = expr_shape.clone();
    out_shape[axis] = 1;
    let out_layout = Layout::new(out_shape.clone());
    let mut out = Tensor::zeros_on(out_shape, backend);

    let out_numel = out.numel();
    let axis_len = expr_shape[axis];

    let _cached_inputs = CachedInputs::<T>::new(expr, backend);

    if axis_len == 0 {
        let slice_result = out.storage_mut().try_as_mut_slice();
        if let Some(slice) = slice_result {
            let out_ptr = MutPtr(slice.as_mut_ptr());
            backend.parallel_for(0, out_numel, move |i| unsafe {
                out_ptr.write(i, T::zero());
            });
        } else {
            let host_out = vec![T::zero(); out_numel];
            backend.copy_to_device(&host_out, out.storage_mut());
        }
        return out;
    }

    // 2. Perform parallel evaluation with reduction
    let plan = FusedReductionPlan {
        out_layout: &out_layout,
        axis,
        axis_len,
        op,
    };
    let slice_result = out.storage_mut().try_as_mut_slice();
    if let Some(slice) = slice_result {
        let out_ptr = MutPtr(slice.as_mut_ptr());
        write_fused_reductions(*expr, out_ptr, out_numel, plan, backend);
    } else {
        // GPU fallback path: evaluate on a temporary CPU buffer and copy back to GPU
        let mut host_out = vec![T::zero(); out_numel];
        let out_ptr = MutPtr(host_out.as_mut_ptr());
        write_fused_reductions(*expr, out_ptr, out_numel, plan, backend);
        backend.copy_to_device(&host_out, out.storage_mut());
    }

    out
}
