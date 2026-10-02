use super::backend_writes::{preserves_cloned_storage, Write};
use coeus_core::{BackendError, ComputeBackend, MoiraiBackend, SequentialBackend};

#[test]
fn backend_fill_preserves_cloned_storage() {
    preserves_cloned_storage(&SequentialBackend::new(), Write::Fill);
    preserves_cloned_storage(&MoiraiBackend::new(), Write::Fill);
}

#[test]
fn backend_zero_fill_preserves_cloned_storage() {
    preserves_cloned_storage(&SequentialBackend::new(), Write::FillZero);
    preserves_cloned_storage(&MoiraiBackend::new(), Write::FillZero);
}

#[test]
fn backend_upload_preserves_cloned_storage() {
    preserves_cloned_storage(&SequentialBackend::new(), Write::CopyToDevice);
    preserves_cloned_storage(&MoiraiBackend::new(), Write::CopyToDevice);
}

fn assert_transfer_length_errors<B>()
where
    B: ComputeBackend<Error = BackendError> + Default,
{
    let backend = B::default();
    let mut storage = backend
        .allocate_zeroed::<u32>(2)
        .expect("invariant: two-element CPU buffer allocation succeeds");
    backend
        .copy_to_device(&[7, 11], &mut storage)
        .expect("invariant: matching host-to-device copy succeeds");

    let error = match backend.copy_to_device(&[13], &mut storage) {
        Err(error) => error,
        Ok(()) => panic!("short host input must be rejected"),
    };
    assert!(matches!(
        error,
        BackendError::BufferLengthMismatch {
            operation: "copy_to_device",
            source_len: 1,
            destination_len: 2,
        }
    ));

    let mut retained = [0; 2];
    backend
        .copy_to_host(&storage, &mut retained)
        .expect("invariant: matching device-to-host copy succeeds");
    assert_eq!(retained, [7, 11]);

    let mut short_output = [19];
    let error = match backend.copy_to_host(&storage, &mut short_output) {
        Err(error) => error,
        Ok(()) => panic!("short host output must be rejected"),
    };
    assert!(matches!(
        error,
        BackendError::BufferLengthMismatch {
            operation: "copy_to_host",
            source_len: 2,
            destination_len: 1,
        }
    ));
    assert_eq!(short_output, [19]);
}

#[test]
fn sequential_backend_rejects_mismatched_transfer_lengths_without_writes() {
    assert_transfer_length_errors::<SequentialBackend>();
}

#[test]
fn moirai_backend_rejects_mismatched_transfer_lengths_without_writes() {
    assert_transfer_length_errors::<MoiraiBackend>();
}
