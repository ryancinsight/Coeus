use crate::{Error, ImageTensor, Result};
use dtype::float::Float32;
use tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    Rgb,
    Luma,
}

pub fn decode_image(bytes: &[u8], mode: ColorMode) -> Result<ImageTensor> {
    let dyn_img = image::load_from_memory(bytes)?;
    match mode {
        ColorMode::Rgb => {
            let rgb = dyn_img.to_rgb8();
            let (w, h) = rgb.dimensions();
            let raw = rgb.into_raw();
            let (w, h) = (w as usize, h as usize);

            let mut data = Vec::with_capacity(3 * h * w);
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        let idx = (y * w + x) * 3 + c;
                        let v = raw[idx] as f32 / 255.0;
                        data.push(Float32::new(v));
                    }
                }
            }

            Ok(Tensor::from_vec(data, &[3, h, w])?)
        }
        ColorMode::Luma => {
            let gray = dyn_img.to_luma8();
            let (w, h) = gray.dimensions();
            let raw = gray.into_raw();
            let (w, h) = (w as usize, h as usize);

            let mut data = Vec::with_capacity(h * w);
            for b in raw {
                data.push(Float32::new(b as f32 / 255.0));
            }

            Ok(Tensor::from_vec(data, &[1, h, w])?)
        }
    }
}

pub fn decode_rgb_image(bytes: &[u8]) -> Result<ImageTensor> {
    decode_image(bytes, ColorMode::Rgb)
}

pub fn decode_luma_image(bytes: &[u8]) -> Result<ImageTensor> {
    decode_image(bytes, ColorMode::Luma)
}

pub fn validate_chw_image(tensor: &ImageTensor) -> Result<(usize, usize, usize)> {
    let shape = tensor.shape().dims();
    if shape.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "expected CHW tensor with 3 dimensions, got shape {shape:?}"
        )));
    }
    Ok((shape[0], shape[1], shape[2]))
}
