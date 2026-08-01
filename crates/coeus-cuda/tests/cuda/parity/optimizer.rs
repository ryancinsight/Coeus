use super::*;

#[test]
fn test_cuda_parity_adamw_step() {
    let Some((s, c)) = backends() else {
        return;
    };
    let n = 16;
    let param: Vec<f32> = (0..n).map(|x| x as f32 * 0.01).collect();
    let grad: Vec<f32> = (0..n).map(|x| -(x as f32 * 0.05 - 0.4)).collect();
    let m1_init: Vec<f32> = vec![0.0; n];
    let m2_init: Vec<f32> = vec![0.0; n];

    let p_c = Tensor::from_slice(vec![n], &param);
    let g_c = Tensor::from_slice(vec![n], &grad);
    let mut m1_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &m1_init);
    let mut m2_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &m2_init);
    let mut p_c_mut = p_c.clone();
    let p_c_layout = p_c_mut.layout().clone();
    let g_c_layout = g_c.layout().clone();
    let m1_c_layout = m1_c.layout().clone();
    let m2_c_layout = m2_c.layout().clone();
    s.adamw_step(
        p_c_mut.storage_mut(),
        &p_c_layout,
        g_c.storage(),
        &g_c_layout,
        m1_c.storage_mut(),
        &m1_c_layout,
        m2_c.storage_mut(),
        &m2_c_layout,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.01,
        1,
    )
    .expect("CPU AdamW step");

    let p_g = to_gpu(&p_c, &s, &c);
    let g_g = to_gpu(&g_c, &s, &c);
    let mut m1_g = Tensor::from_slice_on(vec![n], &m1_init, &c);
    let mut m2_g = Tensor::from_slice_on(vec![n], &m2_init, &c);
    let mut p_g_mut = p_g.clone();
    let p_g_layout = p_g_mut.layout().clone();
    let g_g_layout = g_g.layout().clone();
    let m1_g_layout = m1_g.layout().clone();
    let m2_g_layout = m2_g.layout().clone();
    c.adamw_step(
        p_g_mut.storage_mut(),
        &p_g_layout,
        g_g.storage(),
        &g_g_layout,
        m1_g.storage_mut(),
        &m1_g_layout,
        m2_g.storage_mut(),
        &m2_g_layout,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.01,
        1,
    )
    .expect("CUDA AdamW step");

    assert_parity_tol(
        "adamw_step",
        p_c_mut.as_slice(),
        to_cpu(&p_g_mut, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

// Full round-trip: CPU to GPU to CPU identity.

#[test]
fn test_cuda_parity_roundtrip_identity() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data: Vec<f32> = (0..100).map(|x| x as f32 * 0.123 - 6.15).collect();
    let x = Tensor::<f32, SequentialBackend>::from_slice(vec![10, 10], &data);
    let back = to_gpu(&x, &s, &c).to_backend_on(&c, &s);
    assert_parity_tol("roundtrip", x.as_slice(), back.as_slice(), CUDA_TOL);
}

// ── Provider-owned optimizer step parity (sgd / adam / rmsprop / adagrad) ──
//
// adamw is covered by `test_cuda_parity_adamw_step` above. These cover the
// remaining four Hephaestus stateful updates, checking both the updated
// parameter and the optimizer state against the CPU reference.

#[test]
fn test_cuda_parity_sgd_step() {
    let Some((s, c)) = backends() else {
        return;
    };
    let n = 16;
    let param: Vec<f32> = (0..n).map(|x| x as f32 * 0.01).collect();
    let grad: Vec<f32> = (0..n).map(|x| -(x as f32 * 0.05 - 0.4)).collect();
    let vel: Vec<f32> = (0..n).map(|x| x as f32 * 0.002).collect();
    let (lr, momentum) = (0.05f32, 0.9f32);

    let g_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &grad);
    let mut p_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &param);
    let mut vel_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &vel);
    let pl = p_c.layout().clone();
    let gl = g_c.layout().clone();
    let vl = vel_c.layout().clone();
    s.sgd_step(
        p_c.storage_mut(),
        &pl,
        g_c.storage(),
        &gl,
        vel_c.storage_mut(),
        &vl,
        lr,
        momentum,
    )
    .expect("CPU SGD step");

    let g_g = to_gpu(&g_c, &s, &c);
    let mut p_g = Tensor::from_slice_on(vec![n], &param, &c);
    let mut vel_g = Tensor::from_slice_on(vec![n], &vel, &c);
    c.sgd_step(
        p_g.storage_mut(),
        &pl,
        g_g.storage(),
        &gl,
        vel_g.storage_mut(),
        &vl,
        lr,
        momentum,
    )
    .expect("CUDA SGD step");

    assert_parity_tol(
        "sgd_p",
        p_c.as_slice(),
        to_cpu(&p_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "sgd_velocity",
        vel_c.as_slice(),
        to_cpu(&vel_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_adam_step() {
    let Some((s, c)) = backends() else {
        return;
    };
    let n = 16;
    let param: Vec<f32> = (0..n).map(|x| x as f32 * 0.01).collect();
    let grad: Vec<f32> = (0..n).map(|x| -(x as f32 * 0.05 - 0.4)).collect();
    let m_init: Vec<f32> = (0..n).map(|x| x as f32 * 0.001).collect();
    let v_init: Vec<f32> = (0..n).map(|x| x as f32 * 0.002).collect();
    let (lr, beta1, beta2, eps, t) = (0.05f32, 0.9f32, 0.99f32, 1e-6f32, 3usize);

    let g_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &grad);
    let mut p_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &param);
    let mut m_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &m_init);
    let mut v_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &v_init);
    let pl = p_c.layout().clone();
    let gl = g_c.layout().clone();
    let ml = m_c.layout().clone();
    let vl = v_c.layout().clone();
    s.adam_step(
        p_c.storage_mut(),
        &pl,
        g_c.storage(),
        &gl,
        m_c.storage_mut(),
        &ml,
        v_c.storage_mut(),
        &vl,
        lr,
        beta1,
        beta2,
        eps,
        t,
    )
    .expect("CPU Adam step");

    let g_g = to_gpu(&g_c, &s, &c);
    let mut p_g = Tensor::from_slice_on(vec![n], &param, &c);
    let mut m_g = Tensor::from_slice_on(vec![n], &m_init, &c);
    let mut v_g = Tensor::from_slice_on(vec![n], &v_init, &c);
    c.adam_step(
        p_g.storage_mut(),
        &pl,
        g_g.storage(),
        &gl,
        m_g.storage_mut(),
        &ml,
        v_g.storage_mut(),
        &vl,
        lr,
        beta1,
        beta2,
        eps,
        t,
    )
    .expect("CUDA Adam step");

    assert_parity_tol(
        "adam_p",
        p_c.as_slice(),
        to_cpu(&p_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "adam_m",
        m_c.as_slice(),
        to_cpu(&m_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "adam_v",
        v_c.as_slice(),
        to_cpu(&v_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_rmsprop_step() {
    let Some((s, c)) = backends() else {
        return;
    };
    let n = 16;
    let param: Vec<f32> = (0..n).map(|x| x as f32 * 0.01).collect();
    let grad: Vec<f32> = (0..n).map(|x| -(x as f32 * 0.05 - 0.4)).collect();
    let v_init: Vec<f32> = (0..n).map(|x| x as f32 * 0.002).collect();
    let (lr, alpha, eps) = (0.05f32, 0.99f32, 1e-6f32);

    let g_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &grad);
    let mut p_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &param);
    let mut v_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &v_init);
    let pl = p_c.layout().clone();
    let gl = g_c.layout().clone();
    let vl = v_c.layout().clone();
    s.rmsprop_step(
        p_c.storage_mut(),
        &pl,
        g_c.storage(),
        &gl,
        v_c.storage_mut(),
        &vl,
        lr,
        alpha,
        eps,
    )
    .expect("CPU RMSProp step");

    let g_g = to_gpu(&g_c, &s, &c);
    let mut p_g = Tensor::from_slice_on(vec![n], &param, &c);
    let mut v_g = Tensor::from_slice_on(vec![n], &v_init, &c);
    c.rmsprop_step(
        p_g.storage_mut(),
        &pl,
        g_g.storage(),
        &gl,
        v_g.storage_mut(),
        &vl,
        lr,
        alpha,
        eps,
    )
    .expect("CUDA RMSProp step");

    assert_parity_tol(
        "rmsprop_p",
        p_c.as_slice(),
        to_cpu(&p_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "rmsprop_v",
        v_c.as_slice(),
        to_cpu(&v_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_adagrad_step() {
    let Some((s, c)) = backends() else {
        return;
    };
    let n = 16;
    let param: Vec<f32> = (0..n).map(|x| x as f32 * 0.01).collect();
    let grad: Vec<f32> = (0..n).map(|x| -(x as f32 * 0.05 - 0.4)).collect();
    let h_init: Vec<f32> = (0..n).map(|x| x as f32 * 0.002).collect();
    let (lr, eps) = (0.05f32, 1e-6f32);

    let g_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &grad);
    let mut p_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &param);
    let mut h_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &h_init);
    let pl = p_c.layout().clone();
    let gl = g_c.layout().clone();
    let hl = h_c.layout().clone();
    s.adagrad_step(
        p_c.storage_mut(),
        &pl,
        g_c.storage(),
        &gl,
        h_c.storage_mut(),
        &hl,
        lr,
        eps,
    )
    .expect("CPU AdaGrad step");

    let g_g = to_gpu(&g_c, &s, &c);
    let mut p_g = Tensor::from_slice_on(vec![n], &param, &c);
    let mut h_g = Tensor::from_slice_on(vec![n], &h_init, &c);
    c.adagrad_step(
        p_g.storage_mut(),
        &pl,
        g_g.storage(),
        &gl,
        h_g.storage_mut(),
        &hl,
        lr,
        eps,
    )
    .expect("CUDA AdaGrad step");

    assert_parity_tol(
        "adagrad_p",
        p_c.as_slice(),
        to_cpu(&p_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "adagrad_history",
        h_c.as_slice(),
        to_cpu(&h_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

// ── Transposed convolution forward parity (on-device gather vs CPU scatter) ──
