#![cfg(not(feature = "cuda"))]

use coeus_core::ComputeBackend;
use coeus_cuda::{CudaProvider, HephaestusBackend};
use coeus_hephaestus::HephaestusProvider;

#[test]
fn disabled_provider_keeps_cuda_backend_identity() {
    let backend = HephaestusBackend::<CudaProvider>::new();
    assert_eq!(backend.name(), "cuda");
}

#[test]
fn disabled_provider_acquisition_fails_honestly() {
    let error = CudaProvider::try_device().expect_err("stub CUDA device acquisition must fail");
    assert!(
        matches!(
            error,
            hephaestus_core::HephaestusError::AdapterUnavailable { .. }
        ),
        "unexpected stub acquisition failure: {error:?}"
    );
}
