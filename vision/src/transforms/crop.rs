use crate::ImageTensor;
use rand::Rng;
use tensor::Tensor;
use utils::TransformError;

use super::Transform;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CenterCrop {
    size: (usize, usize),
}

impl CenterCrop {
    pub fn new(size: (usize, usize)) -> Self {
        Self { size }
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }
}

impl<'a> Transform<&'a ImageTensor, ImageTensor> for CenterCrop {
    fn apply(&self, input: &'a ImageTensor) -> Result<ImageTensor, TransformError> {
        let (target_h, target_w) = self.size;
        if target_h == 0 || target_w == 0 {
            return Err(TransformError::InvalidInput {
                message: "crop size must be non-zero".to_string(),
            });
        }

        let shape = input.shape().dims();
        match shape.len() {
            3 => {
                let (c, h, w) = (shape[0], shape[1], shape[2]);
                if target_h > h || target_w > w {
                    return Err(TransformError::ShapeMismatch {
                        expected: format!("H,W >= {target_h},{target_w}"),
                        actual: format!("H,W = {h},{w}"),
                    });
                }
                let top = (h - target_h) / 2;
                let left = (w - target_w) / 2;
                crop_chw(input, c, h, w, top, left, target_h, target_w)
            }
            4 => {
                let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
                if target_h > h || target_w > w {
                    return Err(TransformError::ShapeMismatch {
                        expected: format!("H,W >= {target_h},{target_w}"),
                        actual: format!("H,W = {h},{w}"),
                    });
                }
                let top = (h - target_h) / 2;
                let left = (w - target_w) / 2;
                crop_nchw(input, n, c, h, w, top, left, target_h, target_w)
            }
            _ => Err(TransformError::UnsupportedType {
                type_name: format!("{}-dimensional tensor", shape.len()),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomCrop {
    size: (usize, usize),
}

impl RandomCrop {
    pub fn new(size: (usize, usize)) -> Self {
        Self { size }
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    pub fn apply_with_rng(
        &self,
        input: &ImageTensor,
        rng: &mut impl Rng,
    ) -> Result<ImageTensor, TransformError> {
        let (target_h, target_w) = self.size;
        if target_h == 0 || target_w == 0 {
            return Err(TransformError::InvalidInput {
                message: "crop size must be non-zero".to_string(),
            });
        }

        let shape = input.shape().dims();
        match shape.len() {
            3 => {
                let (c, h, w) = (shape[0], shape[1], shape[2]);
                if target_h > h || target_w > w {
                    return Err(TransformError::ShapeMismatch {
                        expected: format!("H,W >= {target_h},{target_w}"),
                        actual: format!("H,W = {h},{w}"),
                    });
                }
                let max_top = h - target_h;
                let max_left = w - target_w;
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
                crop_chw(input, c, h, w, top, left, target_h, target_w)
            }
            4 => {
                let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
                if target_h > h || target_w > w {
                    return Err(TransformError::ShapeMismatch {
                        expected: format!("H,W >= {target_h},{target_w}"),
                        actual: format!("H,W = {h},{w}"),
                    });
                }
                let max_top = h - target_h;
                let max_left = w - target_w;
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
                crop_nchw(input, n, c, h, w, top, left, target_h, target_w)
            }
            _ => Err(TransformError::UnsupportedType {
                type_name: format!("{}-dimensional tensor", shape.len()),
            }),
        }
    }
}

impl<'a> Transform<&'a ImageTensor, ImageTensor> for RandomCrop {
    fn apply(&self, input: &'a ImageTensor) -> Result<ImageTensor, TransformError> {
        let mut rng = rand::thread_rng();
        self.apply_with_rng(input, &mut rng)
    }
}

fn crop_chw(
    input: &ImageTensor,
    channels: usize,
    height: usize,
    width: usize,
    top: usize,
    left: usize,
    target_height: usize,
    target_width: usize,
) -> Result<ImageTensor, TransformError> {
    let slice = input.as_slice();
    let mut out = Vec::with_capacity(channels * target_height * target_width);
    for c in 0..channels {
        let channel_base = c * height * width;
        for y in 0..target_height {
            let src_row = channel_base + (top + y) * width + left;
            out.extend_from_slice(&slice[src_row..src_row + target_width]);
        }
    }
    Ok(Tensor::from_vec(
        out,
        &[channels, target_height, target_width],
    )?)
}

fn crop_nchw(
    input: &ImageTensor,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    top: usize,
    left: usize,
    target_height: usize,
    target_width: usize,
) -> Result<ImageTensor, TransformError> {
    let slice = input.as_slice();
    let mut out = Vec::with_capacity(batch * channels * target_height * target_width);
    let image_stride = channels * height * width;
    let channel_stride = height * width;

    for n in 0..batch {
        let img_base = n * image_stride;
        for c in 0..channels {
            let channel_base = img_base + c * channel_stride;
            for y in 0..target_height {
                let src_row = channel_base + (top + y) * width + left;
                out.extend_from_slice(&slice[src_row..src_row + target_width]);
            }
        }
    }

    Ok(Tensor::from_vec(
        out,
        &[batch, channels, target_height, target_width],
    )?)
}
