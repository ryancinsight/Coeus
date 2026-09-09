#[path = "../../../coeus-ops/tests/ops/ownership/accumulated_outputs.rs"]
mod accumulated_outputs;
#[path = "../../../coeus-ops/tests/ops/ownership/optimizer.rs"]
mod optimizer;

use accumulated_outputs::{
    attention_preserves_output_clones, convolution_preserves_output_clones,
    pooling_preserves_output_clones, windows_preserve_output_clones,
};
use coeus_cuda::CudaBackend;
use coeus_hephaestus::HephaestusBackend;
use optimizer::adam_preserves_all_state_clones;

#[test]
fn adam_preserves_parameter_and_moment_clones() {
    if !crate::availability::device_available() {
        return;
    }
    adam_preserves_all_state_clones(&CudaBackend::new(), 1.0_f32);
    adam_preserves_all_state_clones(&HephaestusBackend::<CudaBackend>::new(), 1.0_f32);
}

#[test]
fn convolution_preserves_forward_and_accumulated_clones() {
    if !crate::availability::device_available() {
        return;
    }
    convolution_preserves_output_clones(&CudaBackend::new(), 1.0_f32);
    convolution_preserves_output_clones(&CudaBackend::new(), 1.0_f64);
}

#[test]
fn attention_preserves_forward_and_accumulated_clones() {
    if !crate::availability::device_available() {
        return;
    }
    attention_preserves_output_clones(&CudaBackend::new(), 1.0_f32);
    attention_preserves_output_clones(&CudaBackend::new(), 1.0_f64);
    attention_preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), 1.0_f32);
    attention_preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), 1.0_f64);
}

#[test]
fn pooling_preserves_forward_and_accumulated_clones() {
    if !crate::availability::device_available() {
        return;
    }
    pooling_preserves_output_clones(&CudaBackend::new(), 1.0_f32);
    pooling_preserves_output_clones(&CudaBackend::new(), 1.0_f64);
    pooling_preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), 1.0_f32);
    pooling_preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), 1.0_f64);
}

#[test]
fn windows_preserve_extracted_and_folded_output_clones() {
    if !crate::availability::device_available() {
        return;
    }
    windows_preserve_output_clones(&CudaBackend::new(), 1.0_f32);
    windows_preserve_output_clones(&CudaBackend::new(), 1.0_f64);
    windows_preserve_output_clones(&HephaestusBackend::<CudaBackend>::new(), 1.0_f32);
    windows_preserve_output_clones(&HephaestusBackend::<CudaBackend>::new(), 1.0_f64);
}
