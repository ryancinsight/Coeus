use super::*;

#[test]
fn test_cuda_parity_sum_axis0() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 0, &s).expect("valid CPU sum axis");
    let gpu = to_cpu(
        &coeus_ops::sum_axis(&to_gpu(&x, &s, &c), 0, &c).expect("valid CUDA sum axis"),
        &c,
        &s,
    );
    assert_parity_tol("sum_axis0", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_sum_axis1() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 1, &s).expect("valid CPU sum axis");
    let gpu = to_cpu(
        &coeus_ops::sum_axis(&to_gpu(&x, &s, &c), 1, &c).expect("valid CUDA sum axis"),
        &c,
        &s,
    );
    assert_parity_tol("sum_axis1", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_mean_axis() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = (0..12).map(|x| x as f32 * 0.5).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::mean_axis(&x, 1, &s).expect("valid CPU mean axis");
    let gpu = to_cpu(
        &coeus_ops::mean_axis(&to_gpu(&x, &s, &c), 1, &c).expect("valid CUDA mean axis"),
        &c,
        &s,
    );
    assert_parity_tol("mean_axis1", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_max_axis() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 2.0, 0.5, 7.0, 3.0, 5.0, 9.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::max_axis(&x, 1, &s).expect("valid CPU max axis");
    let gpu = to_cpu(
        &coeus_ops::max_axis(&to_gpu(&x, &s, &c), 1, &c).expect("valid CUDA max axis"),
        &c,
        &s,
    );
    assert_parity_tol("max_axis1", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_min_axis() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 0.2, 0.5, 7.0, 3.0, 5.0, -1.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::min_axis(&x, 0, &s).expect("valid CPU min axis");
    let gpu = to_cpu(
        &coeus_ops::min_axis(&to_gpu(&x, &s, &c), 0, &c).expect("valid CUDA min axis"),
        &c,
        &s,
    );
    assert_parity_tol("min_axis0", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_cumulative_scans() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = (1..=6).map(|value| value as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![2, 3], &data);
    let gpu_input = to_gpu(&x, &s, &c);

    let cpu_prefix = coeus_ops::cumsum(&x, 1);
    let gpu_prefix = to_cpu(&coeus_ops::cumsum(&gpu_input, 1), &c, &s);
    assert_parity_tol(
        "cumsum-axis1",
        cpu_prefix.as_slice(),
        gpu_prefix.as_slice(),
        CUDA_TOL,
    );

    let cpu_suffix = coeus_ops::suffix_sum(&x, 0);
    let gpu_suffix = to_cpu(&coeus_ops::suffix_sum(&gpu_input, 0), &c, &s);
    assert_parity_tol(
        "suffix-sum-axis0",
        cpu_suffix.as_slice(),
        gpu_suffix.as_slice(),
        CUDA_TOL,
    );
}

// Matmul.
