use coeus_core::{ComputeBackend, Layout};
use coeus_hephaestus::HephaestusBackend;
use coeus_metal::MetalProvider;
use coeus_ops::CrossOps;

type Backend = HephaestusBackend<MetalProvider>;

fn require_device() {
    if hephaestus_metal::MetalDevice::try_default().is_err() {
        assert_ne!(
            std::env::var("HEPHAESTUS_METAL_REQUIRE_DEVICE").as_deref(),
            Ok("1"),
            "Metal CI requires an acquired device"
        );
    }
}

fn check_cross(a: &[f32], b: &[f32], shape: [usize; 2], dim: usize) {
    let backend = Backend::new();
    let layout = Layout::new(shape.into());
    let numel = a.len();
    // SAFETY: the following `copy_to_device` calls initialize every element
    // before any read.
    let mut a_device = unsafe { backend.allocate::<f32>(numel) }
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(a, &mut a_device)
        .expect("invariant: test backend operation succeeds");
    let mut b_device = unsafe { backend.allocate::<f32>(numel) }
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(b, &mut b_device)
        .expect("invariant: test backend operation succeeds");
    let out_device = CrossOps::cross_storage(&backend, &a_device, &layout, &b_device, dim)
        .expect("Metal cross failed");
    let mut actual = vec![0.0_f32; numel];
    backend
        .copy_to_host(&out_device, &mut actual)
        .expect("invariant: test backend operation succeeds");
    let mut expected = vec![0.0_f32; numel];
    coeus_ops::cross_fold(a, b, &layout, dim, &mut expected);
    assert_eq!(actual, expected, "Metal cross must match the host fold");
}

#[test]
fn metal_cross_last_axis_matches_host_fold() {
    require_device();
    if hephaestus_metal::MetalDevice::try_default().is_err() {
        return;
    }
    // dim=1 is the last axis and both operands are contiguous: exercises the
    // on-device `CrossProductOps` bridge.
    check_cross(
        &[1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[0.0_f32, 1.0, 0.0, 0.0, 0.0, 1.0],
        [2, 3],
        1,
    );
}

#[test]
fn metal_cross_non_last_axis_falls_back() {
    require_device();
    if hephaestus_metal::MetalDevice::try_default().is_err() {
        return;
    }
    // dim=0 on a [3, 3] tensor is not the last axis: the on-device seam
    // cannot express this layout, so the backend falls back to the shared
    // host-fold.
    check_cross(
        &[1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0],
        &[0.0_f32, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
        [3, 3],
        0,
    );
}
