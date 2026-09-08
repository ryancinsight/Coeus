#[path = "../../../coeus-core/tests/core_ops/storage/backend_writes.rs"]
mod shared;

use coeus_cuda::CudaBackend;
use coeus_hephaestus::HephaestusBackend;
use shared::{preserves_cloned_storage, Write};

#[test]
fn backend_fill_preserves_cloned_storage() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_cloned_storage(&CudaBackend::new(), Write::Fill);
}

#[test]
fn provider_backend_fill_preserves_cloned_storage() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_cloned_storage(&HephaestusBackend::<CudaBackend>::new(), Write::Fill);
}

#[test]
fn backend_zero_fill_preserves_cloned_storage() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_cloned_storage(&CudaBackend::new(), Write::FillZero);
}

#[test]
fn provider_backend_zero_fill_preserves_cloned_storage() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_cloned_storage(&HephaestusBackend::<CudaBackend>::new(), Write::FillZero);
}

#[test]
fn backend_upload_preserves_cloned_storage() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_cloned_storage(&CudaBackend::new(), Write::CopyToDevice);
}

#[test]
fn provider_backend_upload_preserves_cloned_storage() {
    if !crate::availability::device_available() {
        return;
    }
    preserves_cloned_storage(
        &HephaestusBackend::<CudaBackend>::new(),
        Write::CopyToDevice,
    );
}
