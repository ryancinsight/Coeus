use crate::fuse::expr_node::{CachedTensor, ExprNode, CPU_EVAL_CACHE};
use crate::ptr::MutPtr;
use coeus_core::{Backend, BackendError, Layout, Scalar, Storage, StorageMut};
use coeus_tensor::Tensor;
use std::marker::PhantomData;

#[repr(transparent)]
#[derive(Clone, Copy)]
struct ScopedExpression(*const ());

// SAFETY: the wrapper exposes no safe dereference operation. Its only use in
// this module is bounded by Backend::parallel_for, which joins all invocations
// before returning and therefore before the borrowed expression is dropped.
unsafe impl Send for ScopedExpression {}
// SAFETY: expression reads are shared and ExprNode requires Sync; callers of
// read additionally uphold the pointee type and lifetime contract.
unsafe impl Sync for ScopedExpression {}

impl ScopedExpression {
    fn new<E>(expression: &E) -> Self {
        Self(std::ptr::from_ref(expression).cast())
    }

    unsafe fn read<E: Copy>(self) -> E {
        // SAFETY: callers provide the original pointee type and keep it alive
        // until the synchronous parallel_for invocation has joined.
        unsafe { self.0.cast::<E>().read() }
    }
}

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
        for tensor in inputs {
            if tensor.storage().try_as_slice().is_none() {
                let contiguous = tensor.to_contiguous_on(backend);
                let mut host_data = vec![T::zero(); contiguous.numel()];
                backend.copy_to_host(contiguous.storage(), &mut host_data);
                let cached_tensor = CachedTensor {
                    data: host_data,
                    layout: contiguous.layout().clone(),
                };
                let bytes = Box::new(cached_tensor) as Box<dyn std::any::Any>;
                let addr = std::ptr::from_ref(tensor) as usize;
                CPU_EVAL_CACHE.with(|cache| {
                    cache.borrow_mut().insert(addr, bytes);
                });
                addresses.push(addr);
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
    let expression = ScopedExpression::new(&expr);

    if contiguous_fast_path {
        backend.parallel_for(0, out_numel, move |idx| unsafe {
            let expr = expression.read::<E>();
            let val = expr.eval_cpu_flat(idx);
            out_ptr.write(idx, val);
        });
    } else {
        backend.parallel_for(0, out_numel, move |idx| {
            // SAFETY: parallel_for joins before write_fused_values returns,
            // keeping the borrowed expression alive for every invocation.
            let expr = unsafe { expression.read::<E>() };
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
        crate::ReductionOp::Prod => acc * val,
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
    let expression = ScopedExpression::new(&expr);

    backend.parallel_for(0, out_numel, move |idx| {
        // SAFETY: parallel_for joins before write_fused_reductions returns,
        // keeping the borrowed expression alive for every invocation.
        let expr = unsafe { expression.read::<E>() };
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

/// Validate the shared empty-axis contract for fused reductions.
///
/// Sum and product retain their additive and multiplicative identities. Mean,
/// maximum, and minimum require at least one input value.
///
/// # Errors
///
/// Returns [`BackendError::EmptyReduction`] when `axis_len` is zero and `op`
/// has no identity.
pub fn validate_fused_reduction_axis(
    op: crate::ReductionOp,
    axis_len: usize,
) -> Result<(), BackendError> {
    if axis_len == 0
        && matches!(
            op,
            crate::ReductionOp::Mean | crate::ReductionOp::Max | crate::ReductionOp::Min
        )
    {
        return Err(BackendError::EmptyReduction {
            operation: "fused reduction",
            reduction: op,
        });
    }
    Ok(())
}

/// Evaluate a fused expression DAG on the CPU, returning a new tensor with the result.
///
/// # Errors
///
/// Returns [`BackendError`] when the expression has no tensor input or its
/// child shapes cannot be broadcast.
pub fn evaluate_fused_cpu<E: ExprNode<T, B> + Copy + Send, T: Scalar, B: Backend>(
    expr: &E,
    backend: &B,
) -> Result<Tensor<T, B>, BackendError> {
    let out_shape = expr.shape()?.ok_or_else(|| BackendError::Storage {
        operation: "fused expression",
        reason: "expression has no tensor input from which to derive its shape".to_string(),
    })?;
    let out_layout = Layout::new(out_shape.clone());
    let mut out = Tensor::alloc_on(out_shape, backend);

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

    Ok(out)
}

/// Evaluate a fused expression DAG with a reduction along `axis` on the CPU.
///
/// # Errors
///
/// Returns [`BackendError`] when the expression has no tensor input, child
/// shapes cannot be broadcast, `axis` is outside the expression rank, or an
/// empty axis is used with mean, maximum, or minimum.
pub fn evaluate_fused_reduce_cpu<E: ExprNode<T, B> + Copy + Send, T: Scalar, B: Backend>(
    expr: &E,
    op: crate::ReductionOp,
    axis: usize,
    backend: &B,
) -> Result<Tensor<T, B>, BackendError> {
    let expr_shape = expr.shape()?.ok_or_else(|| BackendError::Storage {
        operation: "fused reduction",
        reason: "expression has no tensor input from which to derive its shape".to_string(),
    })?;
    if axis >= expr_shape.len() {
        return Err(BackendError::AxisOutOfRange {
            operation: "fused reduction",
            axis,
            rank: expr_shape.len(),
        });
    }

    let mut out_shape = expr_shape.clone();
    out_shape[axis] = 1;
    let out_layout = Layout::new(out_shape.clone());
    let mut out = Tensor::alloc_on(out_shape, backend);

    let out_numel = out.numel();
    let axis_len = expr_shape[axis];
    validate_fused_reduction_axis(op, axis_len)?;

    let _cached_inputs = CachedInputs::<T>::new(expr, backend);

    if axis_len == 0 {
        let identity = match op {
            crate::ReductionOp::Sum => T::zero(),
            crate::ReductionOp::Prod => T::one(),
            crate::ReductionOp::Mean | crate::ReductionOp::Max | crate::ReductionOp::Min => {
                unreachable!("invariant: undefined empty reductions were rejected")
            }
        };
        backend.fill(out.storage_mut(), identity);
        return Ok(out);
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

    Ok(out)
}
