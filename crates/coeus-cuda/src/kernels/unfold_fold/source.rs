pub(super) const SOURCE: &str = include_str!("source.cu");

#[cfg(test)]
mod tests {
    use super::SOURCE;

    #[test]
    fn unfold_fold_source_compiles_for_native_float() {
        let source = SOURCE.replace("{TYPE}", "float");
        crate::kernels::fuse::compile_cuda_to_ptx(&source)
            .expect("unfold/fold CUDA source must compile through NVRTC");
    }
}
