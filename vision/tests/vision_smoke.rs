use std::io::Cursor;

use dtype::float::Float32;
use image::{DynamicImage, ImageBuffer, ImageOutputFormat, Rgb};
use rand::{rngs::StdRng, SeedableRng};
use tensor::Tensor;

use coeus_vision::io::{decode_rgb_image, validate_chw_image};
use coeus_vision::transforms::{
    CenterCrop, ColorJitter, RandomHorizontalFlip, RandomResizedCrop, Transform,
};

fn encode_png_rgb8(
    pixels: &[(u8, u8, u8)],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, image::ImageError> {
    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let (r, g, b) = pixels[idx];
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    let dyn_img = DynamicImage::ImageRgb8(img);
    let mut out = Vec::new();
    let mut cursor = Cursor::new(&mut out);
    dyn_img.write_to(&mut cursor, ImageOutputFormat::Png)?;
    Ok(out)
}

#[test]
fn decode_rgb_produces_chw_float32() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = encode_png_rgb8(
        &[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)],
        2,
        2,
    )?;
    let t = decode_rgb_image(&bytes)?;
    let (c, h, w) = validate_chw_image(&t)?;
    assert_eq!((c, h, w), (3, 2, 2));
    assert_eq!(t.as_slice().len(), 12);
    Ok(())
}

#[test]
fn center_crop_extracts_expected_region() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<Float32> = (0..(3 * 4 * 4)).map(|i| Float32::new(i as f32)).collect();
    let t = Tensor::from_vec(data, &[3, 4, 4])?;

    let crop = CenterCrop::new((2, 2));
    let out = crop.apply(&t)?;
    assert_eq!(out.shape().dims(), &[3, 2, 2]);

    let in_slice = t.as_slice();
    let out_slice = out.as_slice();
    for c in 0..3 {
        for y in 0..2 {
            for x in 0..2 {
                let in_y = 1 + y;
                let in_x = 1 + x;
                let in_idx = c * 16 + in_y * 4 + in_x;
                let out_idx = c * 4 + y * 2 + x;
                assert_eq!(out_slice[out_idx], in_slice[in_idx]);
            }
        }
    }
    Ok(())
}

#[test]
fn random_horizontal_flip_is_deterministic_with_seed() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<Float32> = (0..(3 * 2 * 3)).map(|i| Float32::new(i as f32)).collect();
    let t = Tensor::from_vec(data, &[3, 2, 3])?;

    let flip = RandomHorizontalFlip::new(1.0)?;
    let mut rng = StdRng::seed_from_u64(0);
    let out = flip.apply_with_rng(&t, &mut rng)?;

    let in_slice = t.as_slice();
    let out_slice = out.as_slice();
    for c in 0..3 {
        for y in 0..2 {
            for x in 0..3 {
                let in_idx = c * 6 + y * 3 + x;
                let out_idx = c * 6 + y * 3 + (2 - x);
                assert_eq!(out_slice[out_idx], in_slice[in_idx]);
            }
        }
    }
    Ok(())
}

#[test]
fn color_jitter_preserves_shape_and_range() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<Float32> = (0..(3 * 4 * 4))
        .map(|i| Float32::new((i as f32) / 100.0))
        .collect();
    let t = Tensor::from_vec(data, &[3, 4, 4])?;

    let jitter = ColorJitter::new(0.2, 0.2, 0.2, 0.1, 1.0)?;
    let mut rng = StdRng::seed_from_u64(123);
    let out = jitter.apply_with_rng(&t, &mut rng)?;
    assert_eq!(out.shape().dims(), &[3, 4, 4]);
    for v in out.as_slice() {
        let x = v.get();
        assert!(x.is_finite());
        assert!((0.0..=1.0).contains(&x));
    }
    Ok(())
}

#[test]
fn random_resized_crop_produces_target_size() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<Float32> = (0..(3 * 10 * 12))
        .map(|i| Float32::new((i as f32) / 255.0))
        .collect();
    let t = Tensor::from_vec(data, &[3, 10, 12])?;

    let rrc = RandomResizedCrop::new((4, 5), (0.5, 1.0), (0.75, 1.33))?;
    let mut rng = StdRng::seed_from_u64(42);
    let out = rrc.apply_with_rng(&t, &mut rng)?;
    assert_eq!(out.shape().dims(), &[3, 4, 5]);
    Ok(())
}
