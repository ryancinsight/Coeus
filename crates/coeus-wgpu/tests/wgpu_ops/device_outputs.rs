#[path = "../../../coeus-ops/tests/ops/ownership/device_outputs.rs"]
mod shared;

use coeus_hephaestus::HephaestusBackend;
use coeus_wgpu::WgpuBackend;
use shared::{
    preserves_output_clones, rejects_invalid_output_write, scans_preserve_output_clones, Add,
    Negate, Product, Square, Sum,
};

#[test]
fn backend_negate_preserves_output_clones() {
    if !crate::availability::device_available("device output ownership") {
        return;
    }
    preserves_output_clones(&WgpuBackend::new(), Negate, 1.0_f32);
    preserves_output_clones(&WgpuBackend::new(), Negate, 1_i32);
}

#[test]
fn backend_add_preserves_output_clones() {
    if !crate::availability::device_available("device output ownership") {
        return;
    }
    preserves_output_clones(&WgpuBackend::new(), Add, 1.0_f32);
    preserves_output_clones(&WgpuBackend::new(), Add, 1_i32);
    preserves_output_clones(&WgpuBackend::new(), Add, 1_u32);
}

#[test]
fn backend_sum_preserves_output_clones() {
    if !crate::availability::device_available("device output ownership") {
        return;
    }
    preserves_output_clones(&WgpuBackend::new(), Sum, 1.0_f32);
    preserves_output_clones(&WgpuBackend::new(), Sum, 1_i32);
    preserves_output_clones(&WgpuBackend::new(), Sum, 1_u32);
}

#[test]
fn backend_product_preserves_output_clones() {
    if !crate::availability::device_available("device output ownership") {
        return;
    }
    preserves_output_clones(&WgpuBackend::new(), Product, 1.0_f32);
    preserves_output_clones(&WgpuBackend::new(), Product, 1_i32);
    preserves_output_clones(&WgpuBackend::new(), Product, 1_u32);
}

#[test]
fn backend_invalid_output_requests_preserve_values() {
    if !crate::availability::device_available("device output ownership") {
        return;
    }
    rejects_invalid_output_write(&WgpuBackend::new(), Negate, 1.0_f32);
    rejects_invalid_output_write(&WgpuBackend::new(), Negate, 1_i32);
    rejects_invalid_output_write(&WgpuBackend::new(), Add, 1.0_f32);
    rejects_invalid_output_write(&WgpuBackend::new(), Add, 1_i32);
    rejects_invalid_output_write(&WgpuBackend::new(), Add, 1_u32);
    rejects_invalid_output_write(&WgpuBackend::new(), Sum, 1.0_f32);
    rejects_invalid_output_write(&WgpuBackend::new(), Sum, 1_i32);
    rejects_invalid_output_write(&WgpuBackend::new(), Sum, 1_u32);
    rejects_invalid_output_write(&WgpuBackend::new(), Product, 1.0_f32);
    rejects_invalid_output_write(&WgpuBackend::new(), Product, 1_i32);
    rejects_invalid_output_write(&WgpuBackend::new(), Product, 1_u32);
}

#[test]
fn provider_backend_negate_preserves_output_clones() {
    if !crate::availability::device_available("device output ownership") {
        return;
    }
    preserves_output_clones(&HephaestusBackend::<WgpuBackend>::new(), Negate, 1.0_f32);
    preserves_output_clones(&HephaestusBackend::<WgpuBackend>::new(), Negate, 1_i32);
}

#[test]
fn provider_backend_add_preserves_output_clones() {
    if !crate::availability::device_available("device output ownership") {
        return;
    }
    preserves_output_clones(&HephaestusBackend::<WgpuBackend>::new(), Add, 1.0_f32);
    preserves_output_clones(&HephaestusBackend::<WgpuBackend>::new(), Add, 1_i32);
    preserves_output_clones(&HephaestusBackend::<WgpuBackend>::new(), Add, 1_u32);
}

#[test]
fn provider_backend_sum_preserves_output_clones() {
    if !crate::availability::device_available("device output ownership") {
        return;
    }
    preserves_output_clones(&HephaestusBackend::<WgpuBackend>::new(), Sum, 1.0_f32);
    preserves_output_clones(&HephaestusBackend::<WgpuBackend>::new(), Sum, 1_i32);
    preserves_output_clones(&HephaestusBackend::<WgpuBackend>::new(), Sum, 1_u32);
}

#[test]
fn provider_backend_invalid_output_requests_preserve_values() {
    if !crate::availability::device_available("device output ownership") {
        return;
    }
    rejects_invalid_output_write(&HephaestusBackend::<WgpuBackend>::new(), Negate, 1.0_f32);
    rejects_invalid_output_write(&HephaestusBackend::<WgpuBackend>::new(), Negate, 1_i32);
    rejects_invalid_output_write(&HephaestusBackend::<WgpuBackend>::new(), Add, 1.0_f32);
    rejects_invalid_output_write(&HephaestusBackend::<WgpuBackend>::new(), Add, 1_i32);
    rejects_invalid_output_write(&HephaestusBackend::<WgpuBackend>::new(), Add, 1_u32);
    rejects_invalid_output_write(&HephaestusBackend::<WgpuBackend>::new(), Sum, 1.0_f32);
    rejects_invalid_output_write(&HephaestusBackend::<WgpuBackend>::new(), Sum, 1_i32);
    rejects_invalid_output_write(&HephaestusBackend::<WgpuBackend>::new(), Sum, 1_u32);
}

#[test]
fn backend_square_preserves_output_clones() {
    if !crate::availability::device_available("device output ownership") {
        return;
    }
    preserves_output_clones(&WgpuBackend::new(), Square, 1.0_f32);
    rejects_invalid_output_write(&WgpuBackend::new(), Square, 1.0_f32);
    preserves_output_clones(&HephaestusBackend::<WgpuBackend>::new(), Square, 1.0_f32);
    rejects_invalid_output_write(&HephaestusBackend::<WgpuBackend>::new(), Square, 1.0_f32);
}

#[test]
fn backend_scans_preserves_output_clones() {
    if !crate::availability::device_available("device output ownership") {
        return;
    }
    scans_preserve_output_clones(&WgpuBackend::new(), 1.0_f32);
    scans_preserve_output_clones(&WgpuBackend::new(), 1_i32);
    scans_preserve_output_clones(&WgpuBackend::new(), 1_u32);
    scans_preserve_output_clones(&HephaestusBackend::<WgpuBackend>::new(), 1.0_f32);
    scans_preserve_output_clones(&HephaestusBackend::<WgpuBackend>::new(), 1_i32);
    scans_preserve_output_clones(&HephaestusBackend::<WgpuBackend>::new(), 1_u32);
}
