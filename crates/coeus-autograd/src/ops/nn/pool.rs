use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatElement, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

// ── Backend dispatch helpers ──

struct PoolBackwardInputs<'a, T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    backend: &'a B,
    grad_out_storage: &'a B::DeviceBuffer<T>,
    grad_out_layout: &'a coeus_core::Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &'a mut B::DeviceBuffer<T>,
    grad_input_layout: &'a coeus_core::Layout,
}

#[inline]
fn dispatch_max_pool_backward<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize>(
    request: PoolBackwardInputs<'_, T, B>,
    input_storage: &B::DeviceBuffer<T>,
    input_layout: &coeus_core::Layout,
) -> Result<(), B::Error> {
    let PoolBackwardInputs {
        backend,
        grad_out_storage,
        grad_out_layout,
        kernel_size,
        stride,
        padding,
        dilation,
        grad_input,
        grad_input_layout,
    } = request;

    match DIM {
        1 => backend.max_pool1d_backward(
            grad_out_storage,
            grad_out_layout,
            input_storage,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        ),
        2 => backend.max_pool2d_backward(
            grad_out_storage,
            grad_out_layout,
            input_storage,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        ),
        3 => backend.max_pool3d_backward(
            grad_out_storage,
            grad_out_layout,
            input_storage,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        ),
        _ => panic!("max_pool_backward: unsupported dimension {DIM}"),
    }?;
    Ok(())
}

#[inline]
fn dispatch_avg_pool_backward<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize>(
    request: PoolBackwardInputs<'_, T, B>,
) -> Result<(), B::Error> {
    let PoolBackwardInputs {
        backend,
        grad_out_storage,
        grad_out_layout,
        kernel_size,
        stride,
        padding,
        dilation,
        grad_input,
        grad_input_layout,
    } = request;

    match DIM {
        1 => backend.avg_pool1d_backward(
            grad_out_storage,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        ),
        2 => backend.avg_pool2d_backward(
            grad_out_storage,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        ),
        3 => backend.avg_pool3d_backward(
            grad_out_storage,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        ),
        _ => panic!("avg_pool_backward: unsupported dimension {DIM}"),
    }?;
    Ok(())
}

// ── Max Pool ──

/// Autograd node for N-D max pooling.
pub struct MaxPoolNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Saved input tensor for max-index recomputation during backward.
    pub inp_clone: Tensor<T, B>,
    /// Pooling window size.
    pub kernel_size: usize,
    /// Stride between pooling windows.
    pub stride: usize,
    /// Zero-padding applied to the input.
    pub padding: usize,
    /// Dilation of the pooling window.
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> BackwardNode<T, B>
    for MaxPoolNode<T, B, DIM>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        match DIM {
            1 => "max_pool1d",
            2 => "max_pool2d",
            3 => "max_pool3d",
            _ => "max_poolNd",
        }
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g_in)) = input_grads.first() {
            let mut grad_input = Tensor::zeros_on(self.inp_clone.shape_cloned(), &backend)?;
            let (gi_storage, gi_layout) = grad_input.storage_mut_and_layout()?;
            dispatch_max_pool_backward::<T, B, DIM>(
                PoolBackwardInputs {
                    backend: &backend,
                    grad_out_storage: grad_out.storage(),
                    grad_out_layout: grad_out.layout(),
                    kernel_size: self.kernel_size,
                    stride: self.stride,
                    padding: self.padding,
                    dilation: self.dilation,
                    grad_input: gi_storage,
                    grad_input_layout: gi_layout,
                },
                self.inp_clone.storage(),
                self.inp_clone.layout(),
            )?;
            let gl = g_in.write();
            coeus_ops::add_assign(gl, &grad_input, &backend)?;
        }

        Ok(())
    }
}

fn max_pool_nd_inner<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )?)))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().expect("invariant: requires_grad gates both the Some(grad) construction above and this read").clone();
        let inputs = vec![input.clone()];
        let inp_clone = input.tensor.clone();

        let node = MaxPoolNode::<T, B, DIM> {
            output_grad,
            inputs,
            inp_clone,
            kernel_size,
            stride,
            padding,
            dilation,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}

/// Tracked 1D Max Pooling.
pub fn max_pool1d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Var<T, B>, B::Error> {
    max_pool_nd_inner::<T, B, 1>(input, out_tensor, kernel_size, stride, padding, dilation)
}

/// Tracked 2D Max Pooling.
pub fn max_pool2d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Var<T, B>, B::Error> {
    max_pool_nd_inner::<T, B, 2>(input, out_tensor, kernel_size, stride, padding, dilation)
}

/// Tracked 3D Max Pooling.
pub fn max_pool3d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Var<T, B>, B::Error> {
    max_pool_nd_inner::<T, B, 3>(input, out_tensor, kernel_size, stride, padding, dilation)
}

// ── Adaptive Max Pool 1D ──

/// Adaptive input region for output position `o` over length `in_len`.
///
/// Matches PyTorch's floor/ceil region arithmetic: `start = o*l/o_len`,
/// `end = ceil((o+1)*l/o_len)`. Every region is non-empty for `o_len > 0`.
#[inline]
pub fn adaptive_region(o: usize, in_len: usize, out_len: usize) -> (usize, usize) {
    (o * in_len / out_len, ((o + 1) * in_len).div_ceil(out_len))
}

/// Autograd node for 1D adaptive max pooling.
///
/// Backward scatters each output gradient equally across its region's tied
/// maxima (within the 1e-30 tolerance), matching the `max_axis` convention
/// the broadcast formulation routed through.
pub struct AdaptiveMaxPoolNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Saved `[N, C, L]` input tensor for region-max recomputation.
    pub inp_clone: Tensor<T, B>,
    /// Input length.
    pub in_len: usize,
    /// Output length.
    pub out_len: usize,
}

impl<T: Float + FloatElement, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for AdaptiveMaxPoolNode<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "adaptive_max_pool1d"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g_in)) = input_grads.first() {
            let shape = self.inp_clone.shape_cloned();
            let (n, c) = (shape[0], shape[1]);
            let mut grad_input = Tensor::zeros_on(shape, &backend)?;
            let (gi_storage, gi_layout) = grad_input.storage_mut_and_layout()?;
            let gi_buf = gi_storage.as_mut_slice()?;
            let (gi_off, gi_st) = (gi_layout.offset(), gi_layout.strides_cloned());

            let inp_buf = self.inp_clone.storage().as_slice();
            let inp_layout = self.inp_clone.layout();
            let (in_off, in_st) = (inp_layout.offset(), inp_layout.strides());
            let go_buf = grad_out.storage().as_slice();
            let go_layout = grad_out.layout();
            let (go_off, go_st) = (go_layout.offset(), go_layout.strides());
            let eps = <T as FloatElement>::from_f64(1e-30);

            for n_i in 0..n {
                for c_i in 0..c {
                    let in_base = in_off + n_i * in_st[0] + c_i * in_st[1];
                    let go_base = go_off + n_i * go_st[0] + c_i * go_st[1];
                    let gi_base = gi_off + n_i * gi_st[0] + c_i * gi_st[1];
                    for o in 0..self.out_len {
                        let (start, end) = adaptive_region(o, self.in_len, self.out_len);
                        let mut best = inp_buf[in_base + start * in_st[2]];
                        for i in (start + 1)..end {
                            let v = inp_buf[in_base + i * in_st[2]];
                            if v > best {
                                best = v;
                            }
                        }
                        let mut ties = 0usize;
                        for i in start..end {
                            if Float::abs(inp_buf[in_base + i * in_st[2]] - best) < eps {
                                ties += 1;
                            }
                        }
                        // Exact for every realistic tie count (integer-exact below
                        // 2^24 for f32, 2^53 for f64); regions never approach that.
                        let mut denom = T::ZERO;
                        for _ in 0..ties {
                            denom += T::ONE;
                        }
                        let share = go_buf[go_base + o * go_st[2]] / denom;
                        for i in start..end {
                            if Float::abs(inp_buf[in_base + i * in_st[2]] - best) < eps {
                                let at = gi_base + i * gi_st[2];
                                gi_buf[at] += share;
                            }
                        }
                    }
                }
            }
            let gl = g_in.write();
            coeus_ops::add_assign(gl, &grad_input, &backend)?;
        }

        Ok(())
    }
}

/// Tracked 1D adaptive max pooling over a precomputed output tensor.
///
/// The caller computes the region maxima into `out_tensor`; this wrapper
/// attaches the [`AdaptiveMaxPoolNode`] backward. Region boundaries follow
/// [`adaptive_region`].
///
/// Backward: each output gradient is distributed equally across its region's
/// tied maxima (within 1e-30), matching the `max_axis` convention.
#[inline]
pub fn adaptive_max_pool1d<T: Float + FloatElement, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    in_len: usize,
    out_len: usize,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )?)))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().expect("invariant: requires_grad gates both the Some(grad) construction above and this read").clone();
        let inputs = vec![input.clone()];
        let inp_clone = input.tensor.clone();

        let node = AdaptiveMaxPoolNode::<T, B> {
            output_grad,
            inputs,
            inp_clone,
            in_len,
            out_len,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}

// ── Average Pool ──

/// Autograd node for N-D average pooling.
pub struct AvgPoolNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Shape of the input tensor, used to broadcast gradients on backward.
    pub inp_shape: coeus_core::Shape,
    /// Pooling window size.
    pub kernel_size: usize,
    /// Stride between pooling windows.
    pub stride: usize,
    /// Zero-padding applied to the input.
    pub padding: usize,
    /// Dilation of the pooling window.
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> BackwardNode<T, B>
    for AvgPoolNode<T, B, DIM>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        match DIM {
            1 => "avg_pool1d",
            2 => "avg_pool2d",
            3 => "avg_pool3d",
            _ => "avg_poolNd",
        }
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g_in)) = input_grads.first() {
            let mut grad_input = Tensor::zeros_on(self.inp_shape.clone(), &backend)?;
            let (gi_storage, gi_layout) = grad_input.storage_mut_and_layout()?;
            dispatch_avg_pool_backward::<T, B, DIM>(PoolBackwardInputs {
                backend: &backend,
                grad_out_storage: grad_out.storage(),
                grad_out_layout: grad_out.layout(),
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                dilation: self.dilation,
                grad_input: gi_storage,
                grad_input_layout: gi_layout,
            })?;
            let gl = g_in.write();
            coeus_ops::add_assign(gl, &grad_input, &backend)?;
        }

        Ok(())
    }
}

fn avg_pool_nd_inner<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )?)))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().expect("invariant: requires_grad gates both the Some(grad) construction above and this read").clone();
        let inputs = vec![input.clone()];
        let inp_shape = input.tensor.shape_cloned();

        let node = AvgPoolNode::<T, B, DIM> {
            output_grad,
            inputs,
            inp_shape,
            kernel_size,
            stride,
            padding,
            dilation,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}

/// Tracked 1D Average Pooling.
pub fn avg_pool1d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Var<T, B>, B::Error> {
    avg_pool_nd_inner::<T, B, 1>(input, out_tensor, kernel_size, stride, padding, dilation)
}

/// Tracked 2D Average Pooling.
pub fn avg_pool2d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Var<T, B>, B::Error> {
    avg_pool_nd_inner::<T, B, 2>(input, out_tensor, kernel_size, stride, padding, dilation)
}

/// Tracked 3D Average Pooling.
pub fn avg_pool3d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Var<T, B>, B::Error> {
    avg_pool_nd_inner::<T, B, 3>(input, out_tensor, kernel_size, stride, padding, dilation)
}
