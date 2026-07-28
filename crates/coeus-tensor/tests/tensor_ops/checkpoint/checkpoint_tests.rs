use coeus_core::MoiraiBackend;
use coeus_tensor::{StateDict, StateLimits, Tensor};

fn state_error<T>(result: std::io::Result<T>) -> std::io::Error {
    match result {
        Ok(_) => panic!("state operation must fail"),
        Err(error) => error,
    }
}

#[test]
fn test_state_dict_save_load() {
    let backend = MoiraiBackend::new();
    let t1 = Tensor::from_slice_on(vec![2, 3], &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &backend).expect("construct tensor");
    let t2 = Tensor::from_slice_on(vec![4], &[-1.0f64, -2.0, -3.0, -4.0], &backend).expect("construct tensor");

    let mut sd = StateDict::new();
    sd.insert("layer1.weight", t1);
    sd.insert("layer1.bias", t2);

    let mut buffer = Vec::new();
    sd.save(&mut buffer).unwrap();

    let loaded = StateDict::<f64, MoiraiBackend>::load(&mut &buffer[..]).unwrap();
    assert_eq!(loaded.len(), 2);

    let lt1 = loaded.get("layer1.weight").unwrap();
    assert_eq!(lt1.shape(), &[2, 3]);
    assert_eq!(
        lt1.to_contiguous()
            .expect("materialize contiguous tensor")
            .as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let lt2 = loaded.get("layer1.bias").unwrap();
    assert_eq!(lt2.shape(), &[4]);
    assert_eq!(
        lt2.to_contiguous()
            .expect("materialize contiguous tensor")
            .as_slice(),
        &[-1.0, -2.0, -3.0, -4.0]
    );
}

#[test]
fn test_state_dict_alignment_fallback() {
    let backend = MoiraiBackend::new();
    let t = Tensor::from_slice_on(vec![2], &[1.5f64, 2.5], &backend).expect("construct tensor");

    let mut sd = StateDict::new();
    sd.insert("x", t);

    let mut buffer = Vec::new();
    sd.save(&mut buffer).unwrap();

    let loaded = StateDict::<f64, MoiraiBackend>::load(&mut &buffer[..]).unwrap();
    let lx = loaded.get("x").unwrap();
    assert_eq!(
        lx.to_contiguous()
            .expect("materialize contiguous tensor")
            .as_slice(),
        &[1.5, 2.5]
    );
}

#[test]
fn archived_tensor_view_borrows_payload_and_shape() {
    let backend = MoiraiBackend::new();
    let tensor = Tensor::from_slice_on([2], &[1.5_f64, 2.5], &backend).expect("construct tensor");
    let mut state = StateDict::new();
    state.insert("field.x", tensor);
    let mut bytes = rkyv::util::AlignedVec::<16>::new();
    state.save(&mut bytes).unwrap();

    let archive = StateDict::<f64, MoiraiBackend>::archive(&bytes, StateLimits::default()).unwrap();
    assert_eq!(archive.scalar_type(), "f64");
    let archived = archive.tensor("field.x").unwrap();
    assert_eq!(archived.shape().collect::<Vec<_>>(), [2]);
    assert_eq!(archived.bytes(), bytemuck::cast_slice(&[1.5_f64, 2.5]));
    let archive_range = bytes.as_ptr_range();
    assert!(archive_range.contains(&archived.bytes().as_ptr()));
}

#[test]
fn archive_bytes_are_deterministic_across_insertion_order() {
    fn save(order: [&str; 2]) -> Vec<u8> {
        let backend = MoiraiBackend::new();
        let mut state = StateDict::new();
        for name in order {
            let value = if name == "a" { 1.0_f64 } else { 2.0 };
            state.insert(name, Tensor::from_slice_on([1], &[value], &backend).expect("construct tensor"));
        }
        let mut bytes = Vec::new();
        state.save(&mut bytes).unwrap();
        bytes
    }

    assert_eq!(save(["a", "b"]), save(["b", "a"]));
}

#[test]
fn load_rejects_archive_and_tensor_count_limits() {
    let backend = MoiraiBackend::new();
    let mut state = StateDict::new();
    state.insert("a", Tensor::from_slice_on([1], &[1.0_f64], &backend).expect("construct tensor"));
    state.insert("b", Tensor::from_slice_on([1], &[2.0_f64], &backend).expect("construct tensor"));
    let mut bytes = Vec::new();
    state.save(&mut bytes).unwrap();

    let byte_limits = StateLimits {
        archive_bytes: bytes.len() - 1,
        ..StateLimits::default()
    };
    let byte_error = state_error(StateDict::<f64, MoiraiBackend>::load_with_limits(
        &mut bytes.as_slice(),
        byte_limits,
    ));
    assert!(byte_error
        .to_string()
        .contains("archive bytes exceed limit"));

    let count_limits = StateLimits {
        tensors: 1,
        ..StateLimits::default()
    };
    let count_error = state_error(StateDict::<f64, MoiraiBackend>::load_with_limits(
        &mut bytes.as_slice(),
        count_limits,
    ));
    assert!(count_error
        .to_string()
        .contains("tensor count exceed limit"));

    let rank_limits = StateLimits {
        rank: 0,
        ..StateLimits::default()
    };
    let rank_error = state_error(StateDict::<f64, MoiraiBackend>::load_with_limits(
        &mut bytes.as_slice(),
        rank_limits,
    ));
    assert!(rank_error.to_string().contains("tensor rank exceed limit"));
}

#[test]
fn load_rejects_truncation_and_scalar_mismatch() {
    let backend = MoiraiBackend::new();
    let mut state = StateDict::new();
    state.insert("x", Tensor::from_slice_on([2], &[1.0_f64, 2.0], &backend).expect("construct tensor"));
    let mut bytes = Vec::new();
    state.save(&mut bytes).unwrap();

    let truncated = &bytes[..bytes.len() - 1];
    let truncation = state_error(StateDict::<f64, MoiraiBackend>::load(&mut &*truncated));
    assert!(truncation.to_string().contains("invalid state archive"));

    let mismatch = state_error(StateDict::<f32, MoiraiBackend>::load(&mut bytes.as_slice()));
    assert!(mismatch.to_string().contains("state scalar mismatch"));
}
