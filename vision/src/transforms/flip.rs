use crate::ImageTensor;
use rand::Rng;
use tensor::Tensor;
use utils::TransformError;

use super::Transform;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RandomHorizontalFlip {
    probability: f32,
}

impl RandomHorizontalFlip {
    pub fn new(probability: f32) -> Result<Self, TransformError> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(TransformError::InvalidInput {
                message: "probability must be in [0,1]".to_string(),
            });
        }
        Ok(Self { probability })
    }

    pub fn probability(&self) -> f32 {
        self.probability
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
            3 => flip_chw(input, shape[0], shape[1], shape[2]),
            4 => flip_nchw(input, shape[0], shape[1], shape[2], shape[3]),
            _ => Err(TransformError::UnsupportedType {
                type_name: format!("{}-dimensional tensor", shape.len()),
            }),
        }
    }
}

impl<'a> Transform<&'a ImageTensor, ImageTensor> for RandomHorizontalFlip {
    fn apply(&self, input: &'a ImageTensor) -> Result<ImageTensor, TransformError> {
        let mut rng = rand::thread_rng();
        self.apply_with_rng(input, &mut rng)
    }
}

fn flip_chw(
    input: &ImageTensor,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<ImageTensor, TransformError> {
    let slice = input.as_slice();
    let mut out = Vec::with_capacity(channels * height * width);

    for c in 0..channels {
        let channel_base = c * height * width;
        for y in 0..height {
            let row_base = channel_base + y * width;
            for x in 0..width {
                out.push(slice[row_base + (width - 1 - x)]);
            }
        }
    }

    Ok(Tensor::from_vec(out, &[channels, height, width])?)
}

fn flip_nchw(
    input: &ImageTensor,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<ImageTensor, TransformError> {
    let slice = input.as_slice();
    let mut out = Vec::with_capacity(batch * channels * height * width);

    let image_stride = channels * height * width;
    let channel_stride = height * width;

    for n in 0..batch {
        let img_base = n * image_stride;
        for c in 0..channels {
            let channel_base = img_base + c * channel_stride;
            for y in 0..height {
                let row_base = channel_base + y * width;
                for x in 0..width {
                    out.push(slice[row_base + (width - 1 - x)]);
                }
            }
        }
    }

    Ok(Tensor::from_vec(out, &[batch, channels, height, width])?)
}
