#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{Backend, dispatch::{BackendKind, select_backend}};

    // FIXME: Half-precision operations not yet implemented
    // proptest! {
    //     #[test]
    //     fn prop_fp16_approx(n in 1..=100usize) {
    //         let data_a = vec![1.0f32; n];
    //         let data_b = vec![1.0f32; n];
    //         let backend = select_backend(BackendKind::Cpu).unwrap();
    //         // let half_a = backend.to_half(&data_a);
    //         // let half_b = backend.to_half(&data_b);
    //         // let half_res = backend.half_mul(&half_a, &half_b);
    //         // let f32_res = backend.mul(&data_a, &data_b);
    //         // let dequant_half = backend.from_half(&half_res);
    //         // prop_assert!(dequant_half.iter().zip(f32_res.iter()).all(|(h, f)| (h - f).abs() < 1e-3)); // Approx check
    //         // Temporarily disabled until half-precision methods are implemented
    //     }
    // }
}
