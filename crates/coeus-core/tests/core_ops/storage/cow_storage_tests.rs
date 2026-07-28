use coeus_core::{
    BackendError, CowStorage, CpuAddressableStorage, CpuAddressableStorageMut, CpuStorage, Storage,
};

#[test]
fn cpu_storage_reports_uniqueness_and_detaches_on_mutation() {
    let original = CpuStorage::try_from_slice(&[1_i32, 2, 3, 4]).expect("allocation succeeds");
    assert!(original.is_unique());

    let mut shared = original.clone();
    assert!(!original.is_unique());
    assert!(!shared.is_unique());
    assert_eq!(original.as_slice().as_ptr(), shared.as_slice().as_ptr());

    shared.as_mut_slice().expect("COW allocation")[1] = 20;

    assert!(shared.is_unique());
    assert!(original.is_unique());
    assert_ne!(original.as_slice().as_ptr(), shared.as_slice().as_ptr());
    assert_eq!(original.as_slice(), &[1, 2, 3, 4]);
    assert_eq!(shared.as_slice(), &[1, 20, 3, 4]);
}

#[test]
fn cow_storage_exposes_cpu_uniqueness_without_unwrapping() {
    let original =
        CowStorage::new(CpuStorage::try_from_slice(&[5_i32, 6, 7]).expect("allocation succeeds"));
    assert!(original.is_unique());

    let mut shared = original.clone();
    assert!(!original.is_unique());
    assert!(!shared.is_unique());
    assert_eq!(original.as_slice().as_ptr(), shared.as_slice().as_ptr());

    shared.as_mut_slice().expect("COW allocation")[2] = 70;

    assert!(original.is_unique());
    assert!(shared.is_unique());
    assert_eq!(original.as_slice(), &[5, 6, 7]);
    assert_eq!(shared.as_slice(), &[5, 6, 70]);
}

#[test]
fn empty_cpu_storage_exposes_valid_zero_length_slices() {
    let mut storage = CpuStorage::<u128>::try_new(0).expect("allocation succeeds");

    assert_eq!(storage.len(), 0);
    assert_eq!(storage.as_slice(), &[]);
    assert_eq!(
        storage.as_mut_slice().expect("empty COW allocation"),
        &mut []
    );
}

#[test]
fn cpu_storage_reports_byte_size_overflow() {
    let error = CpuStorage::<u128>::try_new(usize::MAX)
        .err()
        .expect("overflow must be fallible");
    assert!(
        matches!(error, BackendError::Storage { operation, .. } if operation == "CpuStorage::try_new")
    );
}
