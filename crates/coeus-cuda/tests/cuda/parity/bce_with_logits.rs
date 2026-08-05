use super::{assert_parity_tol, backends, to_cpu, to_gpu, Tensor, Var, CUDA_TOL};

#[test]
fn bce_with_logits_dispatches_with_cuda_value_and_gradient_parity() {
    let Some((cpu, cuda)) = backends() else {
        return;
    };
    let logits = Tensor::from_slice_on([2, 2], &[100.0_f32, -100.0, 1.5, -0.5], &cpu);
    let target = Tensor::from_slice_on([2, 2], &[1.0_f32, 0.0, 1.0, 0.0], &cpu);
    let cpu_logits = Var::new(logits.clone().permute(&[1, 0]), true);
    let cpu_target = Var::new(target.clone().permute(&[1, 0]), true);
    let cuda_logits = Var::new(to_gpu(&logits, &cpu, &cuda).permute(&[1, 0]), true);
    let cuda_target = Var::new(to_gpu(&target, &cpu, &cuda).permute(&[1, 0]), true);

    let cpu_loss = coeus_nn::bce_with_logits(&cpu_logits, &cpu_target);
    let cuda_loss = coeus_nn::bce_with_logits(&cuda_logits, &cuda_target);
    cpu_loss
        .backward()
        .expect("CPU BCE-with-logits backward must succeed");
    cuda_loss
        .backward()
        .expect("CUDA BCE-with-logits backward must succeed");

    let cuda_loss = to_cpu(&cuda_loss.tensor, &cuda, &cpu);
    let cpu_logits_gradient = cpu_logits.grad().expect("tracked CPU logits gradient");
    let cuda_logits_gradient = to_cpu(
        &cuda_logits.grad().expect("tracked CUDA logits gradient"),
        &cuda,
        &cpu,
    );
    let cpu_target_gradient = cpu_target.grad().expect("tracked CPU target gradient");
    let cuda_target_gradient = to_cpu(
        &cuda_target.grad().expect("tracked CUDA target gradient"),
        &cuda,
        &cpu,
    );

    assert!(cpu_loss.tensor.as_slice()[0].is_finite());
    assert!(cuda_loss.as_slice()[0].is_finite());

    assert_parity_tol(
        "BCE-with-logits loss",
        cpu_loss.tensor.as_slice(),
        cuda_loss.as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "BCE-with-logits logits gradient",
        cpu_logits_gradient.as_slice(),
        cuda_logits_gradient.as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "BCE-with-logits target gradient",
        cpu_target_gradient.as_slice(),
        cuda_target_gradient.as_slice(),
        CUDA_TOL,
    );
}
