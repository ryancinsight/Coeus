use crate::ImageTensor;
use dtype::float::Float32;
use rand::Rng;
use tensor::Tensor;
use utils::TransformError;

use super::Transform;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorJitter {
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
    probability: f32,
}

impl ColorJitter {
    pub fn new(
        brightness: f32,
        contrast: f32,
        saturation: f32,
        hue: f32,
        probability: f32,
    ) -> Result<Self, TransformError> {
        if brightness < 0.0 || contrast < 0.0 || saturation < 0.0 || hue < 0.0 {
            return Err(TransformError::InvalidInput {
                message: "brightness/contrast/saturation/hue must be non-negative".to_string(),
            });
        }
        if hue > 0.5 {
            return Err(TransformError::InvalidInput {
                message: "hue must be in [0, 0.5] (fraction of full rotation)".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&probability) {
            return Err(TransformError::InvalidInput {
                message: "probability must be in [0,1]".to_string(),
            });
        }
        Ok(Self {
            brightness,
            contrast,
            saturation,
            hue,
            probability,
        })
    }

    pub fn apply_with_rng(
        &self,
        input: &ImageTensor,
        rng: &mut impl Rng,
    ) -> Result<ImageTensor, TransformError> {
        if rng.gen::<f32>() >= self.probability {
            return Ok(input.clone());
        }

        let shape = input.shape().dims();
        match shape.len() {
            3 => color_jitter_chw(input, shape[0], shape[1], shape[2], self, rng),
            4 => color_jitter_nchw(input, shape[0], shape[1], shape[2], shape[3], self, rng),
            _ => Err(TransformError::UnsupportedType {
                type_name: format!("{}-dimensional tensor", shape.len()),
            }),
        }
    }
}

impl<'a> Transform<&'a ImageTensor, ImageTensor> for ColorJitter {
    fn apply(&self, input: &'a ImageTensor) -> Result<ImageTensor, TransformError> {
        let mut rng = rand::thread_rng();
        self.apply_with_rng(input, &mut rng)
    }
}

fn sample_factor(rng: &mut impl Rng, amount: f32) -> f32 {
    if amount == 0.0 {
        return 1.0;
    }
    let low = (1.0 - amount).max(0.0);
    let high = 1.0 + amount;
    rng.gen_range(low..=high)
}

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let v = max;
    let delta = max - min;

    if delta == 0.0 {
        return (0.0, 0.0, v);
    }

    let s = if max == 0.0 { 0.0 } else { delta / max };

    let mut h = if max == r {
        (g - b) / delta
    } else if max == g {
        2.0 + (b - r) / delta
    } else {
        4.0 + (r - g) / delta
    };
    h /= 6.0;
    if h < 0.0 {
        h += 1.0;
    }
    (h, s, v)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (v, v, v);
    }
    let h = (h - h.floor()) * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

fn apply_color_ops(
    r: f32,
    g: f32,
    b: f32,
    brightness_factor: f32,
    contrast_factor: f32,
    saturation_factor: f32,
    hue_delta: f32,
) -> (f32, f32, f32) {
    let mut r = clamp01(r * brightness_factor);
    let mut g = clamp01(g * brightness_factor);
    let mut b = clamp01(b * brightness_factor);

    let mean = (r + g + b) / 3.0;
    r = clamp01((r - mean) * contrast_factor + mean);
    g = clamp01((g - mean) * contrast_factor + mean);
    b = clamp01((b - mean) * contrast_factor + mean);

    let (h, s, v) = rgb_to_hsv(r, g, b);
    let s = clamp01(s * saturation_factor);
    let h = h + hue_delta;
    let (r, g, b) = hsv_to_rgb(h, s, v);
    (clamp01(r), clamp01(g), clamp01(b))
}

fn color_jitter_chw(
    input: &ImageTensor,
    channels: usize,
    height: usize,
    width: usize,
    params: &ColorJitter,
    rng: &mut impl Rng,
) -> Result<ImageTensor, TransformError> {
    if channels != 3 {
        return Err(TransformError::ShapeMismatch {
            expected: "CHW tensor with C=3".to_string(),
            actual: format!("C={channels}"),
        });
    }

    let b = sample_factor(rng, params.brightness);
    let c = sample_factor(rng, params.contrast);
    let s = sample_factor(rng, params.saturation);
    let hue_delta = if params.hue == 0.0 {
        0.0
    } else {
        rng.gen_range(-params.hue..=params.hue)
    };

    let slice = input.as_slice();
    let hw = height * width;
    let mut out = Vec::with_capacity(channels * hw);

    for c_idx in 0..3 {
        let base = c_idx * hw;
        out.extend_from_slice(&slice[base..base + hw]);
    }

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let r = out[idx].get();
            let g = out[hw + idx].get();
            let b0 = out[2 * hw + idx].get();
            let (r, g, b1) = apply_color_ops(r, g, b0, b, c, s, hue_delta);
            out[idx] = Float32::new(r);
            out[hw + idx] = Float32::new(g);
            out[2 * hw + idx] = Float32::new(b1);
        }
    }

    Ok(Tensor::from_vec(out, &[3, height, width])?)
}

fn color_jitter_nchw(
    input: &ImageTensor,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    params: &ColorJitter,
    rng: &mut impl Rng,
) -> Result<ImageTensor, TransformError> {
    if channels != 3 {
        return Err(TransformError::ShapeMismatch {
            expected: "NCHW tensor with C=3".to_string(),
            actual: format!("C={channels}"),
        });
    }

    let b = sample_factor(rng, params.brightness);
    let c = sample_factor(rng, params.contrast);
    let s = sample_factor(rng, params.saturation);
    let hue_delta = if params.hue == 0.0 {
        0.0
    } else {
        rng.gen_range(-params.hue..=params.hue)
    };

    let slice = input.as_slice();
    let hw = height * width;
    let image_stride = channels * hw;
    let mut out = Vec::with_capacity(batch * image_stride);
    out.extend_from_slice(slice);

    for n in 0..batch {
        let base = n * image_stride;
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let r_idx = base + idx;
                let g_idx = base + hw + idx;
                let b_idx = base + 2 * hw + idx;

                let r0 = out[r_idx].get();
                let g0 = out[g_idx].get();
                let b0 = out[b_idx].get();
                let (r1, g1, b1) = apply_color_ops(r0, g0, b0, b, c, s, hue_delta);
                out[r_idx] = Float32::new(r1);
                out[g_idx] = Float32::new(g1);
                out[b_idx] = Float32::new(b1);
            }
        }
    }

    Ok(Tensor::from_vec(out, &[batch, 3, height, width])?)
}
