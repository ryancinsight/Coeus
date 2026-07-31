use coeus_tensor::Tensor;

use super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

#[test]
fn test_wgpu_parity_sum_axis0() {
    let s = seq();
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 0, &s).expect("valid CPU sum axis");
    let gpu = to_cpu(&coeus_ops::sum_axis(&to_gpu(&x), 0, &wgpu()).expect("valid WGPU sum axis"));
    assert_parity("sum_axis0", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_sum_axis1() {
    let s = seq();
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 1, &s).expect("valid CPU sum axis");
    let gpu = to_cpu(&coeus_ops::sum_axis(&to_gpu(&x), 1, &wgpu()).expect("valid WGPU sum axis"));
    assert_parity("sum_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_mean_axis() {
    let s = seq();
    let data = (0..12).map(|x| x as f32 * 0.5).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::mean_axis(&x, 1, &s).expect("valid CPU mean axis");
    let gpu = to_cpu(&coeus_ops::mean_axis(&to_gpu(&x), 1, &wgpu()).expect("valid WGPU mean axis"));
    assert_parity("mean_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_max_axis() {
    let s = seq();
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 2.0, 0.5, 7.0, 3.0, 5.0, 9.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::max_axis(&x, 1, &s).expect("valid CPU max axis");
    let gpu = to_cpu(&coeus_ops::max_axis(&to_gpu(&x), 1, &wgpu()).expect("valid WGPU max axis"));
    assert_parity("max_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_min_axis() {
    let s = seq();
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 0.2, 0.5, 7.0, 3.0, 5.0, -1.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::min_axis(&x, 0, &s).expect("valid CPU min axis");
    let gpu = to_cpu(&coeus_ops::min_axis(&to_gpu(&x), 0, &wgpu()).expect("valid WGPU min axis"));
    assert_parity("min_axis0", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_prod_axis() {
    let s = seq();
    let data = vec![1.0f32, -2.0, 3.0, 4.0, 0.5, 6.0];
    let x = Tensor::from_slice(vec![2, 3], &data);
    let cpu = coeus_ops::prod_axis(&x, 1, &s).expect("valid CPU product axis");
    let gpu =
        to_cpu(&coeus_ops::prod_axis(&to_gpu(&x), 1, &wgpu()).expect("valid WGPU product axis"));
    assert_parity("prod_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_rank_one_sum() {
    let s = seq();
    let input = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0]);
    let cpu = coeus_ops::sum_axis(&input, 0, &s).expect("valid CPU rank-one sum");
    let gpu =
        to_cpu(&coeus_ops::sum_axis(&to_gpu(&input), 0, &wgpu()).expect("valid WGPU rank-one sum"));

    assert_eq!(gpu.shape(), &[1]);
    assert_parity("rank-one-sum", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_rank_one_scan() {
    let input = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0]);
    let cpu = coeus_ops::cumsum(&input, 0);
    let gpu = to_cpu(&coeus_ops::cumsum(&to_gpu(&input), 0));

    assert_eq!(gpu.shape(), &[4]);
    assert_parity("rank-one-scan", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_reduction_rejects_unsupported_rank() {
    let input = Tensor::from_slice(vec![2, 2, 2], &[1.0f32; 8]);
    let gpu_input = to_gpu(&input);

    let error = match coeus_ops::sum_axis(&gpu_input, 1, &wgpu()) {
        Ok(_) => panic!("rank-three WGPU reduction unexpectedly succeeded"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        coeus_wgpu::WgpuBackendError::Validation(coeus_core::BackendError::UnsupportedRank {
            operation: "reduction",
            rank: 3,
            max_rank: 2,
        })
    ));
}
