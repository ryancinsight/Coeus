use coeus_core::{CowStorage, CpuAddressableStorage, CpuAddressableStorageMut, CpuStorage};

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
