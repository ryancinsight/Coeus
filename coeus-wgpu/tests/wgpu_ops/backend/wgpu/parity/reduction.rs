use coeus_tensor::Tensor;

use super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

#[test]
fn test_wgpu_parity_sum_axis0() {
    let s = seq();
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 0, &s);
    let gpu = to_cpu(&coeus_ops::sum_axis(&to_gpu(&x), 0, &wgpu()));
    assert_parity("sum_axis0", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_sum_axis1() {
    let s = seq();
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 1, &s);
    let gpu = to_cpu(&coeus_ops::sum_axis(&to_gpu(&x), 1, &wgpu()));
    assert_parity("sum_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_mean_axis() {
    let s = seq();
    let data = (0..12).map(|x| x as f32 * 0.5).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::mean_axis(&x, 1, &s);
    let gpu = to_cpu(&coeus_ops::mean_axis(&to_gpu(&x), 1, &wgpu()));
    assert_parity("mean_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_max_axis() {
    let s = seq();
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 2.0, 0.5, 7.0, 3.0, 5.0, 9.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::max_axis(&x, 1, &s);
    let gpu = to_cpu(&coeus_ops::max_axis(&to_gpu(&x), 1, &wgpu()));
    assert_parity("max_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_min_axis() {
    let s = seq();
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 0.2, 0.5, 7.0, 3.0, 5.0, -1.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::min_axis(&x, 0, &s);
    let gpu = to_cpu(&coeus_ops::min_axis(&to_gpu(&x), 0, &wgpu()));
    assert_parity("min_axis0", cpu.as_slice(), gpu.as_slice());
}
