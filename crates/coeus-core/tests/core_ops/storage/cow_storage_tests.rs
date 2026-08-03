use coeus_core::{
    CowStorage, CpuAddressableStorage, CpuAddressableStorageMut, CpuStorage, Storage,
};

#[test]
fn cpu_storage_reports_uniqueness_and_detaches_on_mutation() {
    let original = CpuStorage::from_slice(&[1_i32, 2, 3, 4]);
    assert!(original.is_unique());

    let mut shared = original.clone();
    assert!(!original.is_unique());
    assert!(!shared.is_unique());
    assert_eq!(original.as_slice().as_ptr(), shared.as_slice().as_ptr());

    shared.as_mut_slice()[1] = 20;

    assert!(shared.is_unique());
    assert!(original.is_unique());
    assert_ne!(original.as_slice().as_ptr(), shared.as_slice().as_ptr());
    assert_eq!(original.as_slice(), &[1, 2, 3, 4]);
    assert_eq!(shared.as_slice(), &[1, 20, 3, 4]);
}

#[test]
fn cow_storage_exposes_cpu_uniqueness_without_unwrapping() {
    let original = CowStorage::new(CpuStorage::from_slice(&[5_i32, 6, 7]));
    assert!(original.is_unique());

    let mut shared = original.clone();
    assert!(!original.is_unique());
    assert!(!shared.is_unique());
    assert_eq!(original.as_slice().as_ptr(), shared.as_slice().as_ptr());

    shared.as_mut_slice()[2] = 70;

    assert!(original.is_unique());
    assert!(shared.is_unique());
    assert_eq!(original.as_slice(), &[5, 6, 7]);
    assert_eq!(shared.as_slice(), &[5, 6, 70]);
}

#[test]
fn empty_cpu_storage_exposes_valid_zero_length_slices() {
    let mut storage = CpuStorage::<u64>::new(0);

    assert_eq!(storage.len(), 0);
    assert_eq!(storage.as_slice(), &[]);
    assert_eq!(storage.as_mut_slice(), &mut []);
}

#[test]
fn cpu_storage_is_initialized_before_readable_slices_exist() {
    let zeros = CpuStorage::<f32>::new(4);
    assert_eq!(zeros.as_slice(), &[0.0; 4]);

    let filled = CpuStorage::filled(4, 3_i32);
    assert_eq!(filled.as_slice(), &[3; 4]);
}
