use crate::backend_ops::trait_def::BackendOps;
use coeus_core::{Float, Layout};

/// Default host-side 1-D transposed convolution.
///
/// Copies input/weight/bias to host, scatters via stride loop, copies back.
pub fn conv_transpose1d<T: Float, B: BackendOps<T>>(
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
                        out_h[ni * c_out * l_out + oc * l_out + t_out] =
                            out_h[ni * c_out * l_out + oc * l_out + t_out] + in_val * w_val;
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
                    out_h[ni * c_out * l_out + oc * l_out + t] =
                        out_h[ni * c_out * l_out + oc * l_out + t] + b_h[oc];
                }
            }
        }
    }

    let _ = output_padding;
    backend.copy_to_device(&out_h, output);
}

/// Default host-side 2-D transposed convolution.
pub fn conv_transpose2d<T: Float, B: BackendOps<T>>(
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
                                out_h[out_idx] = out_h[out_idx] + in_val * wt_val;
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
                        out_h[idx] = out_h[idx] + b_h[oc];
                    }
                }
            }
        }
    }

    let _ = output_padding;
    backend.copy_to_device(&out_h, output);
}
