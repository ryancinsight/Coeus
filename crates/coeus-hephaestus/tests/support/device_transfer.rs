use coeus_core::ComputeBackend;
use coeus_hephaestus::{HephaestusProvider, HephaestusStorage};

pub(crate) fn rejects_invalid_device_transfer_without_detaching<B, Check>(
    backend: &B,
    is_length_mismatch: Check,
) where
    B: ComputeBackend<DeviceBuffer<u32> = HephaestusStorage<B, u32>> + HephaestusProvider,
    <B as ComputeBackend>::Error: core::fmt::Debug,
    Check: Fn(&<B as ComputeBackend>::Error) -> bool,
{
    let mut destination = backend
        .allocate_zeroed::<u32>(2)
        .expect("invariant: two-element device buffer allocation succeeds");
    backend
        .copy_to_device(&[7_u32, 11], &mut destination)
        .expect("invariant: matching device transfer succeeds");
    let retained = destination.clone();
    let allocation_id = retained.allocation_id();
    assert_eq!(destination.allocation_id(), allocation_id);

    let error = match backend.copy_to_device(&[13_u32], &mut destination) {
        Err(error) => error,
        Ok(()) => panic!("short host input must be rejected"),
    };
    assert!(is_length_mismatch(&error));
    assert_eq!(destination.allocation_id(), allocation_id);

    let mut short_output = [19_u32];
    let error = match backend.copy_to_host(&destination, &mut short_output) {
        Err(error) => error,
        Ok(()) => panic!("short host output must be rejected"),
    };
    assert!(is_length_mismatch(&error));
    assert_eq!(short_output, [19_u32]);

    let mut actual = [0_u32; 2];
    backend
        .copy_to_host(&destination, &mut actual)
        .expect("invariant: matching host transfer succeeds");
    assert_eq!(actual, [7, 11]);
}
