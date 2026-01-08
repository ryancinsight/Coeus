use crate::transforms::Resize;
use crate::ImageTensor;
use rand::Rng;
use tensor::Tensor;
use utils::TransformError;

use super::Transform;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RandomResizedCrop {
    size: (usize, usize),
    scale: (f32, f32),
    ratio: (f32, f32),
}

impl RandomResizedCrop {
    pub fn new(
        size: (usize, usize),
        scale: (f32, f32),
        ratio: (f32, f32),
    ) -> Result<Self, TransformError> {
        if size.0 == 0 || size.1 == 0 {
            return Err(TransformError::InvalidInput {
                message: "size must be non-zero".to_string(),
            });
        }
        if scale.0 <= 0.0 || scale.1 <= 0.0 || scale.0 > scale.1 {
            return Err(TransformError::InvalidInput {
                message: "scale must satisfy 0 < min <= max".to_string(),
            });
        }
        if ratio.0 <= 0.0 || ratio.1 <= 0.0 || ratio.0 > ratio.1 {
            return Err(TransformError::InvalidInput {
                message: "ratio must satisfy 0 < min <= max".to_string(),
            });
        }
        Ok(Self { size, scale, ratio })
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    pub fn apply_with_rng(
        &self,
        input: &ImageTensor,
        rng: &mut impl Rng,
    ) -> Result<ImageTensor, TransformError> {
        let shape = input.shape().dims();
        match shape.len() {
            3 => {
                let (c, h, w) = (shape[0], shape[1], shape[2]);
                let (top, left, crop_h, crop_w) = sample_params(h, w, self.scale, self.ratio, rng)?;
                let cropped = crop_chw(input, c, h, w, top, left, crop_h, crop_w)?;
                let resized = Resize::new(self.size).apply_tensor(&cropped)?;
                Ok(resized)
            }
            4 => {
                let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
                let (top, left, crop_h, crop_w) = sample_params(h, w, self.scale, self.ratio, rng)?;
                let cropped = crop_nchw(input, n, c, h, w, top, left, crop_h, crop_w)?;
                let resized = Resize::new(self.size).apply_tensor(&cropped)?;
                Ok(resized)
            }
            _ => Err(TransformError::UnsupportedType {
                type_name: format!("{}-dimensional tensor", shape.len()),
            }),
        }
    }
}

impl<'a> Transform<&'a ImageTensor, ImageTensor> for RandomResizedCrop {
    fn apply(&self, input: &'a ImageTensor) -> Result<ImageTensor, TransformError> {
        let mut rng = rand::thread_rng();
        self.apply_with_rng(input, &mut rng)
    }
}

fn sample_params(
    height: usize,
    width: usize,
    scale: (f32, f32),
    ratio: (f32, f32),
    rng: &mut impl Rng,
) -> Result<(usize, usize, usize, usize), TransformError> {
    let area = (height * width) as f32;

    for _ in 0..10 {
        let target_area = area * rng.gen_range(scale.0..=scale.1);
        let log_ratio_min = ratio.0.ln();
        let log_ratio_max = ratio.1.ln();
        let aspect = (rng.gen_range(log_ratio_min..=log_ratio_max)).exp();

        let crop_w = (target_area * aspect).sqrt().round() as isize;
        let crop_h = (target_area / aspect).sqrt().round() as isize;

        if crop_h <= 0 || crop_w <= 0 {
            continue;
        }
        let crop_h = crop_h as usize;
        let crop_w = crop_w as usize;

        if crop_h <= height && crop_w <= width {
            let max_top = height - crop_h;
            let max_left = width - crop_w;
            let top = if max_top == 0 {
                0
            } else {
                rng.gen_range(0..=max_top)
            };
            let left = if max_left == 0 {
                0
            } else {
                rng.gen_range(0..=max_left)
            };
            return Ok((top, left, crop_h, crop_w));
        }
    }

    let in_ratio = width as f32 / height as f32;
    let (crop_h, crop_w) = if in_ratio < ratio.0 {
        let crop_w = width;
        let crop_h = ((crop_w as f32) / ratio.0).round() as usize;
        (crop_h.min(height), crop_w)
    } else if in_ratio > ratio.1 {
        let crop_h = height;
        let crop_w = ((crop_h as f32) * ratio.1).round() as usize;
        (crop_h, crop_w.min(width))
    } else {
        (height, width)
    };

    let top = (height - crop_h) / 2;
    let left = (width - crop_w) / 2;
    Ok((top, left, crop_h, crop_w))
}

fn crop_chw(
    input: &ImageTensor,
    channels: usize,
    height: usize,
    width: usize,
    top: usize,
    left: usize,
    crop_h: usize,
    crop_w: usize,
) -> Result<ImageTensor, TransformError> {
    let slice = input.as_slice();
    let mut out = Vec::with_capacity(channels * crop_h * crop_w);
    for c in 0..channels {
        let channel_base = c * height * width;
        for y in 0..crop_h {
            let src_row = channel_base + (top + y) * width + left;
            out.extend_from_slice(&slice[src_row..src_row + crop_w]);
        }
    }
    Ok(Tensor::from_vec(out, &[channels, crop_h, crop_w])?)
}

fn crop_nchw(
    input: &ImageTensor,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    top: usize,
    left: usize,
    crop_h: usize,
    crop_w: usize,
) -> Result<ImageTensor, TransformError> {
    let slice = input.as_slice();
    let mut out = Vec::with_capacity(batch * channels * crop_h * crop_w);
    let image_stride = channels * height * width;
    let channel_stride = height * width;

    for n in 0..batch {
        let img_base = n * image_stride;
        for c in 0..channels {
            let channel_base = img_base + c * channel_stride;
            for y in 0..crop_h {
                let src_row = channel_base + (top + y) * width + left;
                out.extend_from_slice(&slice[src_row..src_row + crop_w]);
            }
        }
    }

    Ok(Tensor::from_vec(out, &[batch, channels, crop_h, crop_w])?)
}
