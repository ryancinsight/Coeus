// ── conv_transpose — transposed (fractional stride) convolution ──
//
// Provides public-API entry points that delegate to the `BackendOps`
// default implementations.  These forward-only functions are the analogue
// of `conv1d` / `conv2d` in this crate.

use crate::backend_ops::BackendOps;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// Compute the output length of a transposed 1-D convolution.
///
/// `l_out = (l - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1`
#[inline]
pub fn conv_transpose1d_output_len(
    l: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> usize {
    (l - 1) * stride + dilation * (kernel - 1) + output_padding + 1 - 2 * padding
}

/// Compute the output spatial dimensions of a transposed 2-D convolution.
///
/// `h_out = (h - 1) * stride - 2 * padding + dilation * (kh - 1) + output_padding + 1`
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn conv_transpose2d_output_dims(
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> (usize, usize) {
    let h_out = (h - 1) * stride + dilation * (kh - 1) + output_padding + 1 - 2 * padding;
    let w_out = (w - 1) * stride + dilation * (kw - 1) + output_padding + 1 - 2 * padding;
    (h_out, w_out)
}

/// 1-D Transposed Convolution.
///
/// - `input`:  `[N, C_in, L]`
/// - `weight`: `[C_in, C_out, K]`  (transposed weight convention)
/// - `bias`:   optional `[C_out]`
/// - `output`: `[N, C_out, L_out]`
///
/// `L_out = (L - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1`
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn conv_transpose1d<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    weight: &Tensor<T, B>,
    bias: Option<&Tensor<T, B>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    backend: &B,
) -> Tensor<T, B> {
    let n = input.shape()[0];
    let c_out = weight.shape()[1];
    let l = input.shape()[2];
    let k = weight.shape()[2];
    let l_out = conv_transpose1d_output_len(l, k, stride, padding, output_padding, dilation);

    let mut output = Tensor::zeros_on([n, c_out, l_out], backend);
    let (out_storage, out_layout) = output.storage_mut_and_layout();
    backend.conv_transpose1d(
        input.storage(),
        input.layout(),
        weight.storage(),
        weight.layout(),
        bias.map(|b| b.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_storage,
        out_layout,
    );
    output
}

/// 2-D Transposed Convolution.
///
/// - `input`:  `[N, C_in, H, W]`
/// - `weight`: `[C_in, C_out, KH, KW]`  (transposed weight convention)
/// - `bias`:   optional `[C_out]`
/// - `output`: `[N, C_out, H_out, W_out]`
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn conv_transpose2d<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    weight: &Tensor<T, B>,
    bias: Option<&Tensor<T, B>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    backend: &B,
) -> Tensor<T, B> {
    let n = input.shape()[0];
    let c_out = weight.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];
    let kh = weight.shape()[2];
    let kw = weight.shape()[3];
    let (h_out, w_out) =
        conv_transpose2d_output_dims(h, w, kh, kw, stride, padding, output_padding, dilation);

    let mut output = Tensor::zeros_on([n, c_out, h_out, w_out], backend);
    let (out_storage, out_layout) = output.storage_mut_and_layout();
    backend.conv_transpose2d(
        input.storage(),
        input.layout(),
        weight.storage(),
        weight.layout(),
        bias.map(|b| b.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_storage,
        out_layout,
    );
    output
}

/// Compute the output spatial dimensions of a transposed 3-D convolution.
///
/// Mirrors `torch.nn.ConvTranspose3d`'s output-shape formula per spatial axis:
/// `d_out = (d - 1) * stride - 2 * padding + dilation * (kd - 1) + output_padding + 1`
/// (analogous for `h_out`, `w_out`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn conv_transpose3d_output_dims(
    d: usize,
    h: usize,
    w: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> (usize, usize, usize) {
    let d_out = (d - 1) * stride + dilation * (kd - 1) + output_padding + 1 - 2 * padding;
    let h_out = (h - 1) * stride + dilation * (kh - 1) + output_padding + 1 - 2 * padding;
    let w_out = (w - 1) * stride + dilation * (kw - 1) + output_padding + 1 - 2 * padding;
    (d_out, h_out, w_out)
}

/// 3-D Transposed Convolution.
///
/// - `input`:  `[N, C_in, D, H, W]`
/// - `weight`: `[C_in, C_out, KD, KH, KW]`  (transposed weight convention)
/// - `bias`:   optional `[C_out]`
/// - `output`: `[N, C_out, D_out, H_out, W_out]`
///
/// `D_out = (D - 1) * stride - 2 * padding + dilation * (KD - 1) + output_padding + 1`
/// (analogous for `H_out`, `W_out`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn conv_transpose3d<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    weight: &Tensor<T, B>,
    bias: Option<&Tensor<T, B>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    backend: &B,
) -> Tensor<T, B> {
    let n = input.shape()[0];
    let c_out = weight.shape()[1];
    let d = input.shape()[2];
    let h = input.shape()[3];
    let w = input.shape()[4];
    let kd = weight.shape()[2];
    let kh = weight.shape()[3];
    let kw = weight.shape()[4];
    let (d_out, h_out, w_out) = conv_transpose3d_output_dims(
        d,
        h,
        w,
        kd,
        kh,
        kw,
        stride,
        padding,
        output_padding,
        dilation,
    );

    let mut output = Tensor::zeros_on([n, c_out, d_out, h_out, w_out], backend);
    let (out_storage, out_layout) = output.storage_mut_and_layout();
    backend.conv_transpose3d(
        input.storage(),
        input.layout(),
        weight.storage(),
        weight.layout(),
        bias.map(|b| b.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_storage,
        out_layout,
    );
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn conv_transpose1d_identity_kernel() {
        // stride=1, padding=0, dilation=1, kernel=[1]: output = input
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![1, 1, 4], &[1.0f32, 2.0, 3.0, 4.0]);
        // weight [C_in=1, C_out=1, K=1] = [[[1]]]
        let weight = Tensor::from_slice(vec![1, 1, 1], &[1.0f32]);
        let out = conv_transpose1d(&input, &weight, None, 1, 0, 0, 1, &b);
        assert_eq!(out.shape(), &[1, 1, 4]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn conv_transpose1d_stride2_upsample() {
        // stride=2: each input position emits to alternating output positions.
        // input [1,1,3]=[1,2,3], weight [1,1,1]=[1] → output [1,1,5]=[1,0,2,0,3]
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![1, 1, 3], &[1.0f32, 2.0, 3.0]);
        let weight = Tensor::from_slice(vec![1, 1, 1], &[1.0f32]);
        let out = conv_transpose1d(&input, &weight, None, 2, 0, 0, 1, &b);
        assert_eq!(out.shape(), &[1, 1, 5]);
        assert_eq!(out.as_slice(), &[1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn conv_transpose2d_identity_kernel() {
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![1, 1, 2, 2], &[1.0f32, 2.0, 3.0, 4.0]);
        let weight = Tensor::from_slice(vec![1, 1, 1, 1], &[1.0f32]);
        let out = conv_transpose2d(&input, &weight, None, 1, 0, 0, 1, &b);
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
