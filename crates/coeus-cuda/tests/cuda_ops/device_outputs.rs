#[path = "../../../coeus-ops/tests/ops/ownership/device_outputs.rs"]
mod shared;

use coeus_cuda::CudaBackend;
use coeus_hephaestus::HephaestusBackend;
use shared::{
    preserves_output_clones, rejects_invalid_output_write, scans_preserve_output_clones, Add,
    Negate, Product, Square, Sum,
};

#[test]
fn backend_negate_preserves_output_clones() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_output_clones(&CudaBackend::new(), Negate, 1.0_f32);
    preserves_output_clones(&CudaBackend::new(), Negate, 1.0_f64);
    preserves_output_clones(&CudaBackend::new(), Negate, 1_i32);
}

#[test]
fn backend_add_preserves_output_clones() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_output_clones(&CudaBackend::new(), Add, 1.0_f32);
    preserves_output_clones(&CudaBackend::new(), Add, 1.0_f64);
    preserves_output_clones(&CudaBackend::new(), Add, 1_i32);
}

#[test]
fn backend_sum_preserves_output_clones() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_output_clones(&CudaBackend::new(), Sum, 1.0_f32);
    preserves_output_clones(&CudaBackend::new(), Sum, 1_i32);
}

#[test]
fn backend_product_preserves_output_clones() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_output_clones(&CudaBackend::new(), Product, 1.0_f32);
    preserves_output_clones(&CudaBackend::new(), Product, 1.0_f64);
    preserves_output_clones(&CudaBackend::new(), Product, 1_i32);
    preserves_output_clones(
        &CudaBackend::new(),
        Product,
        eunomia::F16::from_bits(0x3c00),
    );
    preserves_output_clones(
        &CudaBackend::new(),
        Product,
        eunomia::Bf16::from_bits(0x3f80),
    );
}

#[test]
fn backend_invalid_output_requests_preserve_values() {
    if !crate::availability::device_available() {
        return;
    }
    rejects_invalid_output_write(&CudaBackend::new(), Negate, 1.0_f32);
    rejects_invalid_output_write(&CudaBackend::new(), Negate, 1.0_f64);
    rejects_invalid_output_write(&CudaBackend::new(), Negate, 1_i32);
    rejects_invalid_output_write(&CudaBackend::new(), Add, 1.0_f32);
    rejects_invalid_output_write(&CudaBackend::new(), Add, 1.0_f64);
    rejects_invalid_output_write(&CudaBackend::new(), Add, 1_i32);
    rejects_invalid_output_write(&CudaBackend::new(), Sum, 1.0_f32);
    rejects_invalid_output_write(&CudaBackend::new(), Sum, 1_i32);
    rejects_invalid_output_write(&CudaBackend::new(), Product, 1.0_f32);
    rejects_invalid_output_write(&CudaBackend::new(), Product, 1.0_f64);
    rejects_invalid_output_write(&CudaBackend::new(), Product, 1_i32);
    rejects_invalid_output_write(
        &CudaBackend::new(),
        Product,
        eunomia::F16::from_bits(0x3c00),
    );
    rejects_invalid_output_write(
        &CudaBackend::new(),
        Product,
        eunomia::Bf16::from_bits(0x3f80),
    );
}

#[test]
fn provider_backend_negate_preserves_output_clones() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), Negate, 1.0_f32);
    preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), Negate, 1.0_f64);
    preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), Negate, 1_i32);
}

#[test]
fn provider_backend_add_preserves_output_clones() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), Add, 1.0_f32);
    preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), Add, 1.0_f64);
    preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), Add, 1_i32);
}

#[test]
fn provider_backend_sum_preserves_output_clones() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), Sum, 1.0_f32);
    preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), Sum, 1_i32);
}

#[test]
fn provider_backend_invalid_output_requests_preserve_values() {
    if !crate::availability::device_available() {
        return;
    }
    rejects_invalid_output_write(&HephaestusBackend::<CudaBackend>::new(), Negate, 1.0_f32);
    rejects_invalid_output_write(&HephaestusBackend::<CudaBackend>::new(), Negate, 1.0_f64);
    rejects_invalid_output_write(&HephaestusBackend::<CudaBackend>::new(), Negate, 1_i32);
    rejects_invalid_output_write(&HephaestusBackend::<CudaBackend>::new(), Add, 1.0_f32);
    rejects_invalid_output_write(&HephaestusBackend::<CudaBackend>::new(), Add, 1.0_f64);
    rejects_invalid_output_write(&HephaestusBackend::<CudaBackend>::new(), Add, 1_i32);
    rejects_invalid_output_write(&HephaestusBackend::<CudaBackend>::new(), Sum, 1.0_f32);
    rejects_invalid_output_write(&HephaestusBackend::<CudaBackend>::new(), Sum, 1_i32);
}

#[test]
fn backend_square_preserves_output_clones() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_output_clones(&CudaBackend::new(), Square, 1.0_f32);
    rejects_invalid_output_write(&CudaBackend::new(), Square, 1.0_f32);
    preserves_output_clones(&CudaBackend::new(), Square, 1.0_f64);
    rejects_invalid_output_write(&CudaBackend::new(), Square, 1.0_f64);
    preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), Square, 1.0_f32);
    rejects_invalid_output_write(&HephaestusBackend::<CudaBackend>::new(), Square, 1.0_f32);
    preserves_output_clones(&HephaestusBackend::<CudaBackend>::new(), Square, 1.0_f64);
    rejects_invalid_output_write(&HephaestusBackend::<CudaBackend>::new(), Square, 1.0_f64);
}

#[test]
fn backend_scans_preserves_output_clones() {
    if !crate::availability::device_available() {
        return;
    }
    scans_preserve_output_clones(&CudaBackend::new(), 1.0_f32);
    scans_preserve_output_clones(&CudaBackend::new(), 1_i32);
    scans_preserve_output_clones(&HephaestusBackend::<CudaBackend>::new(), 1.0_f32);
    scans_preserve_output_clones(&HephaestusBackend::<CudaBackend>::new(), 1_i32);
}
