use crate::backend_ops::CpuBackend;
use coeus_core::{ComputeBackend, Float, Layout};

/// Default host-side 1-D transposed convolution.
///
/// Copies input/weight/bias to host, scatters via stride loop, copies back.
pub fn conv_transpose1d<T: Float, B: ComputeBackend>(
    backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    weight: &B::DeviceBuffer<T>,
    weight_layout: &Layout,
    bias: Option<&B::DeviceBuffer<T>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) {
    let n = input_layout.shape()[0];
    let c_in = input_layout.shape()[1];
    let l = input_layout.shape()[2];
    let c_out = weight_layout.shape()[1];
    let k = weight_layout.shape()[2];
    let l_out = output_layout.shape()[2];

    let in_numel = n * c_in * l;
    let w_numel = c_in * c_out * k;
    let out_numel = n * c_out * l_out;

    let mut in_h = vec![T::zero(); in_numel];
    let mut w_h = vec![T::zero(); w_numel];
    let mut out_h = vec![T::zero(); out_numel];

    backend.copy_to_host(input, &mut in_h);
    backend.copy_to_host(weight, &mut w_h);

    for ni in 0..n {
        for ic in 0..c_in {
            for ti in 0..l {
                let in_val = in_h[ni * c_in * l + ic * l + ti];
                for oc in 0..c_out {
                    for ki in 0..k {
                        let t_out = ti * stride + ki * dilation;
                        if t_out < padding {
                            continue;
                        }
                        let t_out = t_out - padding;
                        if t_out >= l_out {
                            continue;
                        }
                        let w_val = w_h[ic * c_out * k + oc * k + ki];
                        out_h[ni * c_out * l_out + oc * l_out + t_out] += in_val * w_val;
                    }
                }
            }
        }
    }

    if let Some(b) = bias {
        let mut b_h = vec![T::zero(); c_out];
        backend.copy_to_host(b, &mut b_h);
        for ni in 0..n {
            for oc in 0..c_out {
                for t in 0..l_out {
                    out_h[ni * c_out * l_out + oc * l_out + t] += b_h[oc];
                }
            }
        }
    }

    let _ = output_padding;
    backend.copy_to_device(&out_h, output);
}

/// Default host-side 2-D transposed convolution.
pub fn conv_transpose2d<T: Float, B: ComputeBackend>(
    backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    weight: &B::DeviceBuffer<T>,
    weight_layout: &Layout,
    bias: Option<&B::DeviceBuffer<T>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) {
    let n = input_layout.shape()[0];
    let c_in = input_layout.shape()[1];
    let h = input_layout.shape()[2];
    let w = input_layout.shape()[3];
    let c_out = weight_layout.shape()[1];
    let kh = weight_layout.shape()[2];
    let kw = weight_layout.shape()[3];
    let h_out = output_layout.shape()[2];
    let w_out = output_layout.shape()[3];

    let in_numel = n * c_in * h * w;
    let weight_numel = c_in * c_out * kh * kw;
    let out_numel = n * c_out * h_out * w_out;

    let mut in_h = vec![T::zero(); in_numel];
    let mut wt_h = vec![T::zero(); weight_numel];
    let mut out_h = vec![T::zero(); out_numel];

    backend.copy_to_host(input, &mut in_h);
    backend.copy_to_host(weight, &mut wt_h);

    for ni in 0..n {
        for ic in 0..c_in {
            for hi in 0..h {
                for wi in 0..w {
                    let in_val = in_h[ni * c_in * h * w + ic * h * w + hi * w + wi];
                    for oc in 0..c_out {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let h_pos = hi * stride + ki * dilation;
                                let w_pos = wi * stride + kj * dilation;
                                if h_pos < padding || w_pos < padding {
                                    continue;
                                }
                                let h_out_idx = h_pos - padding;
                                let w_out_idx = w_pos - padding;
                                if h_out_idx >= h_out || w_out_idx >= w_out {
                                    continue;
                                }
                                let wt_val =
                                    wt_h[ic * c_out * kh * kw + oc * kh * kw + ki * kw + kj];
                                let out_idx = ni * c_out * h_out * w_out
                                    + oc * h_out * w_out
                                    + h_out_idx * w_out
                                    + w_out_idx;
                                out_h[out_idx] += in_val * wt_val;
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(b) = bias {
        let mut b_h = vec![T::zero(); c_out];
        backend.copy_to_host(b, &mut b_h);
        for ni in 0..n {
            for oc in 0..c_out {
                for hi in 0..h_out {
                    for wi in 0..w_out {
                        let idx = ni * c_out * h_out * w_out + oc * h_out * w_out + hi * w_out + wi;
                        out_h[idx] += b_h[oc];
                    }
                }
            }
        }
    }

    let _ = output_padding;
    backend.copy_to_device(&out_h, output);
}

/// Default CPU-side 3-D transposed convolution.
///
/// Inputs are copied to host, scattered via the canonical transposed-conv
/// 5-D loop (`out[ni, oc, do, ho, wo] += in[ni, ic, di, hi, wi] *
/// weight[ic, oc, kd, kh, kw]` with `q = p * stride + k * dilation -
/// padding`), optionally biased, then written back.
///
/// `output_padding` is constrained by the PyTorch output-shape formula
/// (`out = (in - 1)*stride - 2*padding + dilation*(k-1) + output_padding + 1`);
/// it merely controls the allocation size computed upstream, so the
/// forward kernel does not need to consult it once the output layout is
/// supplied.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose3d<T: Float, B: CpuBackend>(
    backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    weight: &B::DeviceBuffer<T>,
    weight_layout: &Layout,
    bias: Option<&B::DeviceBuffer<T>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) {
    let n = input_layout.shape()[0];
    let c_in = input_layout.shape()[1];
    let d = input_layout.shape()[2];
    let h = input_layout.shape()[3];
    let w_in_coord = input_layout.shape()[4];
    let c_out = weight_layout.shape()[1];
    let kd = weight_layout.shape()[2];
    let kh = weight_layout.shape()[3];
    let kw = weight_layout.shape()[4];
    let d_out = output_layout.shape()[2];
    let h_out = output_layout.shape()[3];
    let w_out = output_layout.shape()[4];

    let in_numel = n * c_in * d * h * w_in_coord;
    let weight_numel = c_in * c_out * kd * kh * kw;
    let out_numel = n * c_out * d_out * h_out * w_out;

    let mut in_h = vec![T::zero(); in_numel];
    let mut wt_h = vec![T::zero(); weight_numel];
    let mut out_h = vec![T::zero(); out_numel];

    backend.copy_to_host(input, &mut in_h);
    backend.copy_to_host(weight, &mut wt_h);

    for ni in 0..n {
        for ic in 0..c_in {
            for di in 0..d {
                for hi in 0..h {
                    for wi in 0..w_in_coord {
                        let in_val = in_h[ni * c_in * d * h * w_in_coord
                            + ic * d * h * w_in_coord
                            + di * h * w_in_coord
                            + hi * w_in_coord
                            + wi];
                        for oc in 0..c_out {
                            for k_i in 0..kd {
                                for kj in 0..kh {
                                    for kk in 0..kw {
                                        let d_pos = di * stride + k_i * dilation;
                                        let h_pos = hi * stride + kj * dilation;
                                        let w_pos = wi * stride + kk * dilation;
                                        if d_pos < padding || h_pos < padding || w_pos < padding {
                                            continue;
                                        }
                                        let d_out_idx = d_pos - padding;
                                        let h_out_idx = h_pos - padding;
                                        let w_out_idx = w_pos - padding;
                                        if d_out_idx >= d_out
                                            || h_out_idx >= h_out
                                            || w_out_idx >= w_out
                                        {
                                            continue;
                                        }
                                        let wt_val = wt_h[ic * c_out * kd * kh * kw
                                            + oc * kd * kh * kw
                                            + k_i * kh * kw
                                            + kj * kw
                                            + kk];
                                        let out_idx = ni * c_out * d_out * h_out * w_out
                                            + oc * d_out * h_out * w_out
                                            + d_out_idx * h_out * w_out
                                            + h_out_idx * w_out
                                            + w_out_idx;
                                        out_h[out_idx] += in_val * wt_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(b) = bias {
        let mut b_h = vec![T::zero(); c_out];
        backend.copy_to_host(b, &mut b_h);
        for ni in 0..n {
            for oc in 0..c_out {
                for di in 0..d_out {
                    for hi in 0..h_out {
                        for wi in 0..w_out {
                            let idx = ni * c_out * d_out * h_out * w_out
                                + oc * d_out * h_out * w_out
                                + di * h_out * w_out
                                + hi * w_out
                                + wi;
                            out_h[idx] += b_h[oc];
                        }
                    }
                }
            }
        }
    }

    let _ = output_padding;
    backend.copy_to_device(&out_h, output);
}
