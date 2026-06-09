use coeus_core::{ComputeBackend, SequentialBackend};
use coeus_tensor::Tensor;
use coeus_wgpu::{add, WgpuBackend};

#[test]
fn test_wgpu_transfers_and_addition() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];

    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &a_data);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &b_data);

    let a_wgpu = a_seq.to_backend_on(&seq, &wgpu_b);
    let b_wgpu = b_seq.to_backend_on(&seq, &wgpu_b);

    assert_eq!(a_wgpu.shape(), &[2, 3]);
    assert_eq!(b_wgpu.shape(), &[2, 3]);

    let c_wgpu = add(&a_wgpu, &b_wgpu);
    let c_seq = c_wgpu.to_backend_on(&wgpu_b, &seq);

    let expected = vec![11.0f32, 22.0, 33.0, 44.0, 55.0, 66.0];
    assert_eq!(c_seq.as_slice(), &expected);
}

#[test]
fn test_wgpu_backend_ops_unified() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let a = Tensor::<f32, WgpuBackend>::from_slice_on(
        vec![2, 3],
        &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
        &wgpu_b,
    );
    let b = Tensor::<f32, WgpuBackend>::from_slice_on(
        vec![2, 3],
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        &wgpu_b,
    );

    let c = coeus_ops::add(&a, &b, &wgpu_b);
    let c_cpu = c.to_backend_on(&wgpu_b, &seq);
    assert_eq!(c_cpu.as_slice(), &[11.0, 18.0, 33.0, 36.0, 55.0, 54.0]);

    let d = coeus_ops::relu(&a, &wgpu_b);
    let d_cpu = d.to_backend_on(&wgpu_b, &seq);
    assert_eq!(d_cpu.as_slice(), &[1.0, 0.0, 3.0, 0.0, 5.0, 0.0]);

    let m1 = Tensor::<f32, WgpuBackend>::from_slice_on(vec![2, 2], &[1.0, 2.0, 3.0, 4.0], &wgpu_b);
    let m2 = Tensor::<f32, WgpuBackend>::from_slice_on(vec![2, 2], &[5.0, 6.0, 7.0, 8.0], &wgpu_b);
    let mr = coeus_ops::matmul(&m1, &m2, &wgpu_b);
    let mr_cpu = mr.to_backend_on(&wgpu_b, &seq);
    assert_eq!(mr_cpu.as_slice(), &[19.0, 22.0, 43.0, 50.0]);

    let s0 = coeus_ops::sum_axis(&a, 0, &wgpu_b);
    let s0_cpu = s0.to_backend_on(&wgpu_b, &seq);
    assert_eq!(s0_cpu.as_slice(), &[-3.0, 3.0, -3.0]);
}

#[test]
fn test_wgpu_tiled_matmul() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let m = 20;
    let k = 24;
    let n = 18;

    let a_data: Vec<f32> = (0..m * k).map(|x| (x as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..k * n).map(|x| (x as f32) * 0.02).collect();

    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![m, k], &a_data);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![k, n], &b_data);

    let a_wgpu = a_seq.to_backend_on(&seq, &wgpu_b);
    let b_wgpu = b_seq.to_backend_on(&seq, &wgpu_b);

    let c_wgpu = coeus_wgpu::matmul(&a_wgpu, &b_wgpu);
    let c_seq_res = c_wgpu.to_backend_on(&wgpu_b, &seq);

    let c_seq_expected = coeus_ops::matmul(&a_seq, &b_seq, &seq);

    assert_eq!(c_seq_res.shape(), c_seq_expected.shape());
    let slice_res = c_seq_res.as_slice();
    let slice_expected = c_seq_expected.as_slice();
    for i in 0..slice_res.len() {
        let diff = (slice_res[i] - slice_expected[i]).abs();
        assert!(
            diff < 1e-4,
            "Mismatch at {}: {} vs {} (diff {})",
            i,
            slice_res[i],
            slice_expected[i],
            diff
        );
    }
}

#[test]
fn test_wgpu_cow_semantics() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let a = Tensor::<f32, WgpuBackend>::from_slice_on(vec![3], &[1.0, 2.0, 3.0], &wgpu_b);
    let mut b = a.clone();

    wgpu_b.fill(b.storage_mut(), 10.0);

    let a_cpu = a.to_backend_on(&wgpu_b, &seq);
    let b_cpu = b.to_backend_on(&wgpu_b, &seq);

    assert_eq!(a_cpu.as_slice(), &[1.0, 2.0, 3.0]);
    assert_eq!(b_cpu.as_slice(), &[10.0, 10.0, 10.0]);
}
