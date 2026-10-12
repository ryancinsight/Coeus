//! WGPU provider wiring for the generic Hephaestus backend.

mod provider;

pub use coeus_hephaestus::HephaestusBackend;
#[cfg(test)]
pub(crate) use provider::get_wgpu_context;
pub use provider::WgpuProvider;
