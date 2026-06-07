use coeus_core::MoiraiBackend;
use coeus_tensor::{Tensor, StateDict};

#[test]
fn test_state_dict_save_load() {
    let backend = MoiraiBackend::new();
    let t1 = Tensor::from_slice_on(vec![2, 3], &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &backend);
    let t2 = Tensor::from_slice_on(vec![4], &[-1.0f64, -2.0, -3.0, -4.0], &backend);

    let mut sd = StateDict::new();
    sd.insert("layer1.weight", t1);
    sd.insert("layer1.bias", t2);

    let mut buffer = Vec::new();
    sd.save(&mut buffer).unwrap();

    let loaded = StateDict::<f64, MoiraiBackend>::load(&mut &buffer[..]).unwrap();
    assert_eq!(loaded.len(), 2);

    let lt1 = loaded.get("layer1.weight").unwrap();
    assert_eq!(lt1.shape(), &[2, 3]);
    assert_eq!(lt1.to_contiguous().as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let lt2 = loaded.get("layer1.bias").unwrap();
    assert_eq!(lt2.shape(), &[4]);
    assert_eq!(lt2.to_contiguous().as_slice(), &[-1.0, -2.0, -3.0, -4.0]);
}

#[test]
fn test_state_dict_alignment_fallback() {
    let backend = MoiraiBackend::new();
    let t = Tensor::from_slice_on(vec![2], &[1.5f64, 2.5], &backend);

    let mut sd = StateDict::new();
    sd.insert("x", t);

    let mut buffer = Vec::new();
    sd.save(&mut buffer).unwrap();

    let loaded = StateDict::<f64, MoiraiBackend>::load(&mut &buffer[..]).unwrap();
    let lx = loaded.get("x").unwrap();
    assert_eq!(lx.to_contiguous().as_slice(), &[1.5, 2.5]);
}
