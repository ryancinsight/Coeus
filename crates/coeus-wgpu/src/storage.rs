use crate::backend::get_wgpu_context;
use coeus_core::ComputeBackend;
use coeus_core::StorageMut;
use coeus_hephaestus::HephaestusStorage;
use hephaestus_core::{ComputeDevice, DeviceBuffer};
use themis::{MemoryTier, PlacementHint};

#[test]
fn storage_allocates_device_tier() {
    let storage = HephaestusStorage::<crate::WgpuBackend, f32>::new(16);
    assert_eq!(storage.buffer().tier(), MemoryTier::Device);
}

#[test]
fn device_upload_roundtrip_preserves_values() {
    let ctx = get_wgpu_context();
    let input = vec![1.0f32, -2.5, 3.25, 8.0];
    let device_buf = ctx
        .hephaestus_device
        .upload_with_hint(&input, PlacementHint::Tier(MemoryTier::Device))
        .expect("failed to upload into device tier");
    assert_eq!(device_buf.tier(), MemoryTier::Device);
    let mut out = vec![0.0f32; input.len()];
    ctx.hephaestus_device
        .download(&device_buf, &mut out)
        .expect("failed to download from device tier");
    assert_eq!(out, input);
}

#[test]
fn backend_zero_memory_operations_preserve_exact_values() {
    let backend = crate::backend::WgpuBackend::new();
    let mut storage = backend.allocate_zeroed::<u32>(4);
    let mut values = [u32::MAX; 4];
    backend.copy_to_host(&storage, &mut values);
    assert_eq!(values, [0; 4]);

    backend.fill(&mut storage, 0xdead_beef);
    backend.fill(&mut storage, 0);
    backend.copy_to_host(&storage, &mut values);
    assert_eq!(values, [0; 4]);
}

#[test]
fn host_pinned_upload_is_rejected_without_false_tier() {
    let ctx = get_wgpu_context();
    let input = vec![1.0f32, -2.5, 3.25, 8.0];
    let error = ctx
        .hephaestus_device
        .upload_with_hint(&input, PlacementHint::Tier(MemoryTier::HostPinned))
        .expect_err("WGPU cannot guarantee persistent host-pinned placement");
    match error {
        hephaestus_core::HephaestusError::AllocationFailed { message } => assert_eq!(
            message,
            "WGPU cannot guarantee requested memory tier HostPinned; use Device placement"
        ),
        other => panic!("expected allocation failure, got {other:?}"),
    }
}

#[test]
fn copy_on_write_preserves_values_in_both_device_buffers() {
    let ctx = get_wgpu_context();
    let input = vec![1.0f32, -2.5, 3.25, 8.0];
    let source = ctx
        .hephaestus_device
        .upload_with_hint(&input, PlacementHint::Tier(MemoryTier::Device))
        .expect("failed to upload COW source");
    let mut writable = HephaestusStorage::<crate::WgpuBackend, _>::from_buffer(source);
    let retained = writable.clone();

    writable.make_unique();

    assert_ne!(writable.allocation_id(), retained.allocation_id());
    let mut writable_values = vec![0.0f32; input.len()];
    let mut retained_values = vec![0.0f32; input.len()];
    ctx.hephaestus_device
        .download(writable.buffer(), &mut writable_values)
        .expect("failed to download detached COW buffer");
    ctx.hephaestus_device
        .download(retained.buffer(), &mut retained_values)
        .expect("failed to download retained COW buffer");

    assert_eq!(writable_values, input);
    assert_eq!(retained_values, input);
}
