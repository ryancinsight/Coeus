#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // FIXME: ONNX export/import not yet implemented
    // proptest! {
    //     #[test]
    //     fn prop_onnx_roundtrip(n in 1..=10usize) {
    //         let backend = select_backend(BackendKind::Cpu);
    //         let t = Tensor::new(vec![1.0f32; n], vec![n], backend);
    //         let exported = backend.export_onnx(&[t.clone()]);
    //         let imported = /* parse ONNX */ exported; // Assume import fn
    //         prop_assert_eq!(imported[0], t); // Roundtrip check
    //     }
    // }
}
