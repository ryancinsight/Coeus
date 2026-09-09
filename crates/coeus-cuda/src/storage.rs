use crate::backend::get_cuda_device;
use coeus_core::StorageMut;
use coeus_hephaestus::HephaestusStorage;
use hephaestus_core::{ComputeDevice, DeviceBuffer};
use themis::{MemoryTier, PlacementHint};

#[test]
fn storage_allocates_device_tier() {
    let storage = HephaestusStorage::<crate::CudaBackend, f32>::new(8);
    assert_eq!(storage.buffer().tier(), MemoryTier::Device);
}

#[test]
fn host_pinned_hint_uses_truthful_device_tier() {
    let device = get_cuda_device();
    let input = vec![1.0f32, -2.5, 3.25, 8.0];
    let staging = device
        .upload_with_hint(&input, PlacementHint::Tier(MemoryTier::HostPinned))
        .expect("failed to upload into host-pinned tier");
    // CUDA's ComputeDevice buffer contract represents device allocations;
    // host-pinned transfer memory is transient and is not exposed as this
    // persistent buffer type. The provider therefore reports Device.
    assert_eq!(staging.tier(), MemoryTier::Device);
    let mut roundtrip = vec![0.0f32; input.len()];
    device
        .download(&staging, &mut roundtrip)
        .expect("failed to download from host-pinned tier");
    assert_eq!(roundtrip, input);
}

#[test]
fn copy_on_write_preserves_values_in_both_device_buffers() {
    let device = get_cuda_device();
    let input = vec![1.0f32, -2.5, 3.25, 8.0];
    let source = device
        .upload_with_hint(&input, PlacementHint::Tier(MemoryTier::Device))
        .expect("failed to upload COW source");
    let mut writable = HephaestusStorage::<crate::CudaBackend, _>::from_buffer(source);
    let retained = writable.clone();

    writable.make_unique();

    assert_ne!(writable.allocation_id(), retained.allocation_id());
    let mut writable_values = vec![0.0f32; input.len()];
    let mut retained_values = vec![0.0f32; input.len()];
    device
        .download(writable.buffer(), &mut writable_values)
        .expect("failed to download detached COW buffer");
    device
        .download(retained.buffer(), &mut retained_values)
        .expect("failed to download retained COW buffer");

    assert_eq!(writable_values, input);
    assert_eq!(retained_values, input);
}
