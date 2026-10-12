//! CUDA provider wiring for the generic Hephaestus backend.

mod provider;

pub use coeus_hephaestus::HephaestusBackend;
#[cfg(all(test, feature = "cuda"))]
pub(crate) use provider::get_cuda_device;
pub use provider::{CudaProvider, CudaScalar};
