// ── Adaptive Pooling NN modules ──
//
// AdaptiveAvgPool1d / AdaptiveAvgPool2d / AdaptiveMaxPool1d / AdaptiveMaxPool2d:
//   Stateless modules where the user specifies the output spatial size rather
//   than the kernel/stride.  The pooling regions are computed dynamically to
//   evenly cover the input spatial extent.
//
//   Matches PyTorch `nn.AdaptiveAvgPool1d(output_size)` etc.
//
// ZST phantom markers provide zero-overhead type safety without any allocation.

use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;
use std::marker::PhantomData;

/// Transposed averaging matrix `P_T` `[in_len, out_len]` for adaptive average
/// pooling along one axis. Output column `o` carries `1/region` in the rows of
/// its input region `[floor(o*in/out), ceil((o+1)*in/out))` — PyTorch's adaptive
/// region convention — and zero elsewhere. `input @ P_T` then averages each
/// region; being a constant matmul it is differentiable (the backward scatters
/// the per-region-averaged gradient back to every input position in the region).
fn avg_pool_matrix_t<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    in_len: usize,
    out_len: usize,
    backend: &B,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let mut pt = vec![T::zero(); in_len * out_len];
    for o in 0..out_len {
        let start = o * in_len / out_len;
        let end = ((o + 1) * in_len).div_ceil(out_len);
        let inv = T::from_f64(1.0 / (end - start) as f64);
        // Column `o`, rows `start..end`: flat index `l * out_len + o`.
        for slot in pt
            .iter_mut()
            .skip(start * out_len + o)
            .step_by(out_len)
            .take(end - start)
        {
            *slot = inv;
        }
    }
    Ok(Var::new(
        Tensor::from_slice_on([in_len, out_len], &pt, backend)?,
        false,
    )?)
}

/// Adaptive **max** pool over the last axis of a 2D view `[rows, in_len]` →
/// `[rows, out_len]`. Each output takes the max over its adaptive input region
/// (PyTorch's region convention). Differentiable: non-region positions are
/// filled with `-inf` and `max_axis` routes the gradient to each region's
/// argmax. The transient `[rows, out_len, in_len]` mask is `out_len`× the slice
/// — modest for typical pools; an argmax-scatter kernel would avoid it.
fn masked_adaptive_max<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x2: &Var<T, B>,
    rows: usize,
    in_len: usize,
    out_len: usize,
    backend: &B,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    // outside[o, l] = 1 where l is NOT in region o → filled with -inf below.
    let mut outside = vec![T::one(); out_len * in_len];
    for o in 0..out_len {
        let start = o * in_len / out_len;
        let end = ((o + 1) * in_len).div_ceil(out_len);
        outside[o * in_len + start..o * in_len + end].fill(T::zero());
    }
    let outside_var = Var::new(
        Tensor::from_slice_on([1, out_len, in_len], &outside, backend)?,
        false,
    )?;
    let reshaped = coeus_autograd::reshape(x2, [rows, 1, in_len])?;
    let xb = coeus_autograd::broadcast_to(
        &reshaped,
        vec![rows, out_len, in_len],
    )?;
    let ob = coeus_autograd::broadcast_to(&outside_var, vec![rows, out_len, in_len])?;
    let masked = coeus_autograd::masked_fill(&xb, &ob, T::from_f64(f64::NEG_INFINITY))?;
    coeus_autograd::max_axis(&masked, 2)
}

// ── AdaptiveAvgPool1d ─────────────────────────────────────────────────────────

/// Adaptive average pooling for `[N, C, L]` → `[N, C, output_size]`.
///
/// Matches PyTorch `nn.AdaptiveAvgPool1d(output_size)`.  Each output position
/// pools over a region of the input whose size is determined by the ratio
/// `L / output_size`, with ceiling/floor arithmetic for non-divisible sizes.
///
/// # Examples
///
/// ```
/// use coeus_nn::{AdaptiveAvgPool1d, Module};
/// use coeus_autograd::Var;
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
///
/// let m = AdaptiveAvgPool1d::<f32, SequentialBackend>::new(2);
/// let x = Var::new(
///     Tensor::<f32, SequentialBackend>::ones([1, 3, 8]).expect("construct tensor"),
///     false,
/// )
/// .expect("construct variable");
/// let y = m.forward(&x).expect("run forward");
/// assert_eq!(y.tensor.shape(), &[1, 3, 2]);
/// ```
#[derive(Clone, Debug)]
pub struct AdaptiveAvgPool1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Target spatial output length.
    pub output_size: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> AdaptiveAvgPool1d<T, B> {
    /// Create an `AdaptiveAvgPool1d` with the given output size.
    pub const fn new(output_size: usize) -> Self {
        Self {
            output_size,
            _marker: PhantomData,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AdaptiveAvgPool1d<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        // Differentiable: average each adaptive region via a constant averaging
        // matmul over the length axis.  [N, C, L] -> [N*C, L] @ P_T[L, O].
        let shape = input.tensor.shape_cloned();
        let (n, c, l) = (shape[0], shape[1], shape[2]);
        let o = self.output_size;
        let backend = B::default();
        let p_t = avg_pool_matrix_t::<T, B>(l, o, &backend)?;
        let x2 = coeus_autograd::reshape(input, [n * c, l])?;
        let out2 = coeus_autograd::matmul(&x2, &p_t)?;
        coeus_autograd::reshape(&out2, [n, c, o])
    }
}

// ── AdaptiveAvgPool2d ─────────────────────────────────────────────────────────

/// Adaptive average pooling for `[N, C, H, W]` → `[N, C, out_h, out_w]`.
///
/// Matches PyTorch `nn.AdaptiveAvgPool2d((out_h, out_w))`.  Commonly used as
/// the final spatial pooling stage in CNN classifiers (e.g., ResNet).
///
/// # Examples
///
/// ```
/// use coeus_nn::{AdaptiveAvgPool2d, Module};
/// use coeus_autograd::Var;
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
///
/// let m = AdaptiveAvgPool2d::<f32, SequentialBackend>::new(1, 1);
/// let x = Var::new(
///     Tensor::<f32, SequentialBackend>::ones([2, 4, 8, 8]).expect("construct tensor"),
///     false,
/// )
/// .expect("construct variable");
/// let y = m.forward(&x).expect("run forward");
/// assert_eq!(y.tensor.shape(), &[2, 4, 1, 1]);
/// ```
#[derive(Clone, Debug)]
pub struct AdaptiveAvgPool2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Target output height.
    pub out_h: usize,
    /// Target output width.
    pub out_w: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> AdaptiveAvgPool2d<T, B> {
    /// Create an `AdaptiveAvgPool2d` pooling to `(out_h, out_w)` output size.
    pub const fn new(out_h: usize, out_w: usize) -> Self {
        Self {
            out_h,
            out_w,
            _marker: PhantomData,
        }
    }

    /// Create an `AdaptiveAvgPool2d` pooling to a square `(size, size)` output.
    pub const fn square(size: usize) -> Self {
        Self::new(size, size)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AdaptiveAvgPool2d<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        // Differentiable separable pooling: average over W, then over H, each a
        // constant averaging matmul. Averaging is separable so this equals the
        // joint 2D adaptive average.
        let shape = input.tensor.shape_cloned();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (oh, ow) = (self.out_h, self.out_w);
        let backend = B::default();

        // Fast path for global (1×1) pooling: sequential mean_axis reductions
        // avoid allocating the O(H*W) averaging matrix.
        if oh == 1 && ow == 1 {
            let after_h = coeus_autograd::mean_axis(input, 2)?; // [N, C, 1, W]
            return coeus_autograd::mean_axis(&after_h, 3); // [N, C, 1, 1]
        }

        // Pool W: [N, C, H, W] -> [N*C*H, W] @ PW_T[W, OW] -> [N, C, H, OW].
        let pw_t = avg_pool_matrix_t::<T, B>(w, ow, &backend)?;
        let xw = coeus_autograd::reshape(input, [n * c * h, w])?;
        let yw = coeus_autograd::matmul(&xw, &pw_t)?;
        let yw = coeus_autograd::reshape(&yw, [n, c, h, ow])?;

        // Pool H: bring H last, [N, C, OW, H] -> [N*C*OW, H] @ PH_T[H, OH].
        let ph_t = avg_pool_matrix_t::<T, B>(h, oh, &backend)?;
        let yw_p = coeus_autograd::permute(&yw, &[0, 1, 3, 2])?;
        let yh_input = coeus_autograd::reshape(&yw_p, [n * c * ow, h])?;
        let yh = coeus_autograd::matmul(&yh_input, &ph_t)?;
        let yh = coeus_autograd::reshape(&yh, [n, c, ow, oh])?;
        // Final transpose to [N, C, OH, OW]; reshape materializes it contiguous.
        let out = coeus_autograd::permute(&yh, &[0, 1, 3, 2])?;
        coeus_autograd::reshape(&out, [n, c, oh, ow])
    }
}

// ── AdaptiveMaxPool1d ─────────────────────────────────────────────────────────

/// Adaptive max pooling for `[N, C, L]` → `[N, C, output_size]`.
///
/// Matches PyTorch `nn.AdaptiveMaxPool1d(output_size)`.
#[derive(Clone, Debug)]
pub struct AdaptiveMaxPool1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Target spatial output length.
    pub output_size: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> AdaptiveMaxPool1d<T, B> {
    /// Create an `AdaptiveMaxPool1d` with the given output size.
    pub const fn new(output_size: usize) -> Self {
        Self {
            output_size,
            _marker: PhantomData,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AdaptiveMaxPool1d<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        // Differentiable: masked max over each adaptive region along the length.
        let shape = input.tensor.shape_cloned();
        let (n, c, l) = (shape[0], shape[1], shape[2]);
        let o = self.output_size;
        let backend = B::default();
        let x2 = coeus_autograd::reshape(input, [n * c, l])?;
        let pooled = masked_adaptive_max::<T, B>(&x2, n * c, l, o, &backend)?;
        coeus_autograd::reshape(&pooled, [n, c, o])
    }
}

// ── AdaptiveMaxPool2d ─────────────────────────────────────────────────────────

/// Adaptive max pooling for `[N, C, H, W]` → `[N, C, out_h, out_w]`.
///
/// Matches PyTorch `nn.AdaptiveMaxPool2d((out_h, out_w))`.
#[derive(Clone, Debug)]
pub struct AdaptiveMaxPool2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Target output height.
    pub out_h: usize,
    /// Target output width.
    pub out_w: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> AdaptiveMaxPool2d<T, B> {
    /// Create an `AdaptiveMaxPool2d` pooling to `(out_h, out_w)` output size.
    pub const fn new(out_h: usize, out_w: usize) -> Self {
        Self {
            out_h,
            out_w,
            _marker: PhantomData,
        }
    }

    /// Create an `AdaptiveMaxPool2d` pooling to a square `(size, size)` output.
    pub const fn square(size: usize) -> Self {
        Self::new(size, size)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AdaptiveMaxPool2d<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        // Differentiable separable max pooling: max over W, then over H (max of a
        // 2D region equals the max of per-axis maxes), each a masked max_axis.
        let shape = input.tensor.shape_cloned();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (oh, ow) = (self.out_h, self.out_w);
        let backend = B::default();

        // Fast path for global (1×1) max pooling: sequential max_axis reductions.
        if oh == 1 && ow == 1 {
            let after_h = coeus_autograd::max_axis(input, 2)?; // [N, C, 1, W]
            return coeus_autograd::max_axis(&after_h, 3); // [N, C, 1, 1]
        }

        // Pool W: [N, C, H, W] -> [N*C*H, W] -> [N*C*H, OW] -> [N, C, H, OW].
        let xw = coeus_autograd::reshape(input, [n * c * h, w])?;
        let pw = masked_adaptive_max::<T, B>(&xw, n * c * h, w, ow, &backend)?;
        let yw = coeus_autograd::reshape(&pw, [n, c, h, ow])?;

        // Pool H: bring H last, [N, C, OW, H] -> [N*C*OW, H] -> [N*C*OW, OH].
        let yw_p = coeus_autograd::permute(&yw, &[0, 1, 3, 2])?;
        let yh_input = coeus_autograd::reshape(&yw_p, [n * c * ow, h])?;
        let ph = masked_adaptive_max::<T, B>(
            &yh_input,
            n * c * ow,
            h,
            oh,
            &backend,
        )?;
        let yh = coeus_autograd::reshape(&ph, [n, c, ow, oh])?;
        let out = coeus_autograd::permute(&yh, &[0, 1, 3, 2])?;
        coeus_autograd::reshape(&out, [n, c, oh, ow])
    }
}
