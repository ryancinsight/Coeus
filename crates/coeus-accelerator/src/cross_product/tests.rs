use super::HostCrossProductOps;
use crate::storage::tests::fixtures::{TestDevice, TestProvider};
use crate::{CrossProductProvider, HephaestusBackend, HephaestusProvider, HephaestusStorage};
use coeus_core::{ComputeBackend, Layout, Scalar};
use coeus_ops::CrossOps;
use hephaestus_core::{ComputeDevice, CrossProductOps};

impl<T> CrossProductProvider<T> for TestProvider
where
    T: Scalar,
    HostCrossProductOps: CrossProductOps<TestDevice, T>,
{
    type Operations = HostCrossProductOps;
}

type Backend = HephaestusBackend<TestProvider>;

fn upload(backend: &Backend, data: &[f32]) -> HephaestusStorage<TestProvider, f32> {
    // SAFETY: the upload below writes every element before any read.
    let mut storage = unsafe { backend.allocate(data.len()) }
        .expect("invariant: test backend allocation succeeds");
    backend
        .copy_to_device(data, &mut storage)
        .expect("invariant: test backend upload succeeds");
    storage
}

fn download(
    backend: &Backend,
    storage: &HephaestusStorage<TestProvider, f32>,
    len: usize,
) -> Vec<f32> {
    let mut host = vec![0.0; len];
    backend
        .copy_to_host(storage, &mut host)
        .expect("invariant: test backend download succeeds");
    host
}

#[test]
fn seam_bundle_matches_host_fold_on_last_axis() {
    let backend = Backend::new();
    let layout = Layout::new(vec![2, 3].into());
    let a = upload(&backend, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let b = upload(&backend, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    // dim=1 is the last axis and both operands are contiguous: routes through
    // the `HostCrossProductOps` seam bundle (download/fold/upload/copy).
    let out = backend
        .cross_storage(&a, &layout, &b, 1)
        .expect("invariant: test cross product succeeds");
    assert_eq!(
        download(&backend, &out, 6),
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    );
}

#[test]
fn fallback_matches_host_fold_off_last_axis() {
    let backend = Backend::new();
    let layout = Layout::new(vec![3, 3].into());
    let a = upload(&backend, &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0]);
    let b = upload(&backend, &[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0]);
    // dim=0 is not the last axis: the seam cannot express this layout, so the
    // blanket falls back to the shared host-fold free function.
    let out = backend
        .cross_storage(&a, &layout, &b, 0)
        .expect("invariant: test cross product succeeds");
    assert_eq!(
        download(&backend, &out, 9),
        vec![0.0, 0.0, 0.0, -5.0, 0.0, 20.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn seam_bundle_rejects_length_mismatch() {
    let device = TestProvider::try_device().expect("invariant: test device acquisition succeeds");
    let a = device
        .upload(&[0.0_f32; 6])
        .expect("invariant: test buffer upload succeeds");
    let b = device
        .upload(&[0.0_f32; 3])
        .expect("invariant: test buffer upload succeeds");
    let out = device
        .upload(&[0.0_f32; 6])
        .expect("invariant: test buffer upload succeeds");
    let error = HostCrossProductOps
        .cross_into(device, &a, &b, &out)
        .expect_err("length mismatch must be rejected");
    assert!(matches!(
        error,
        hephaestus_core::HephaestusError::InvalidConfiguration { .. }
    ));
}

#[test]
fn seam_bundle_accepts_empty() {
    let device = TestProvider::try_device().expect("invariant: test device acquisition succeeds");
    let empty: &[f32] = &[];
    let a = device
        .upload(empty)
        .expect("invariant: test buffer upload succeeds");
    let b = device
        .upload(empty)
        .expect("invariant: test buffer upload succeeds");
    let out = device
        .upload(empty)
        .expect("invariant: test buffer upload succeeds");
    HostCrossProductOps
        .cross_into(device, &a, &b, &out)
        .expect("invariant: empty cross product succeeds");
    let mut host = Vec::new();
    device
        .download(&out, &mut host)
        .expect("invariant: test buffer download succeeds");
    assert!(host.is_empty());
}
