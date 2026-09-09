#[path = "../../../coeus-ops/tests/ops/ownership/accumulated_outputs.rs"]
mod accumulated_outputs;
#[path = "../../../coeus-ops/tests/ops/ownership/optimizer.rs"]
mod optimizer;
#[path = "../../../coeus-ops/tests/ops/ownership/staggered.rs"]
mod staggered;

use accumulated_outputs::{
    attention_preserves_output_clones, convolution_preserves_output_clones,
    pooling_preserves_output_clones, windows_preserve_output_clones,
};
use coeus_hephaestus::HephaestusBackend;
use coeus_wgpu::WgpuBackend;
use optimizer::adam_preserves_all_state_clones;
use staggered::staggered_preserves_clones;

#[test]
fn adam_preserves_parameter_and_moment_clones() {
    if !crate::availability::device_available("state update ownership") {
        return;
    }
    adam_preserves_all_state_clones(&WgpuBackend::new(), 1.0_f32);
    adam_preserves_all_state_clones(&HephaestusBackend::<WgpuBackend>::new(), 1.0_f32);
}

#[test]
fn convolution_preserves_forward_and_accumulated_clones() {
    if !crate::availability::device_available("convolution ownership") {
        return;
    }
    convolution_preserves_output_clones(&WgpuBackend::new(), 1.0_f32);
}

#[test]
fn attention_preserves_forward_and_accumulated_clones() {
    if !crate::availability::device_available("attention ownership") {
        return;
    }
    attention_preserves_output_clones(&WgpuBackend::new(), 1.0_f32);
    attention_preserves_output_clones(&HephaestusBackend::<WgpuBackend>::new(), 1.0_f32);
}

#[test]
fn pooling_preserves_forward_and_accumulated_clones() {
    if !crate::availability::device_available("pooling ownership") {
        return;
    }
    pooling_preserves_output_clones(&WgpuBackend::new(), 1.0_f32);
    pooling_preserves_output_clones(&HephaestusBackend::<WgpuBackend>::new(), 1.0_f32);
}

#[test]
fn windows_preserve_extracted_and_folded_output_clones() {
    if !crate::availability::device_available("window ownership") {
        return;
    }
    windows_preserve_output_clones(&WgpuBackend::new(), 1.0_f32);
    windows_preserve_output_clones(&HephaestusBackend::<WgpuBackend>::new(), 1.0_f32);
}

#[test]
fn staggered_preserves_output_clones() {
    if !crate::availability::device_available("staggered ownership") {
        return;
    }
    staggered_preserves_clones(&WgpuBackend::new(), 1.0_f32);
}
