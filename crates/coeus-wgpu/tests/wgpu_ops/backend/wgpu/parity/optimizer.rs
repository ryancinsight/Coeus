use coeus_core::SequentialBackend;
use coeus_ops::OptimizerOps;
use coeus_tensor::Tensor;

use super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

#[test]
fn test_wgpu_parity_adamw_step() {
    let s = seq();
    let w = wgpu();
    let n = 16;
    let param: Vec<f32> = (0..n).map(|x| x as f32 * 0.01).collect();
    let grad: Vec<f32> = (0..n).map(|x| -(x as f32 * 0.05 - 0.4)).collect();
    let m1_init: Vec<f32> = vec![0.0; n];
    let m2_init: Vec<f32> = vec![0.0; n];

    let p_c = Tensor::from_slice(vec![n], &param).expect("construct tensor");
    let g_c = Tensor::from_slice(vec![n], &grad).expect("construct tensor");
    let mut m1_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &m1_init)
        .expect("construct tensor");
    let mut m2_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &m2_init)
        .expect("construct tensor");
    let mut p_c_mut = p_c.clone();
    let p_c_layout = p_c_mut.layout().clone();
    let g_c_layout = g_c.layout().clone();
    let m1_c_layout = m1_c.layout().clone();
    let m2_c_layout = m2_c.layout().clone();
    s.adamw_step(
        p_c_mut.storage_mut().expect("access tensor storage"),
        &p_c_layout,
        g_c.storage(),
        &g_c_layout,
        m1_c.storage_mut().expect("access tensor storage"),
        &m1_c_layout,
        m2_c.storage_mut().expect("access tensor storage"),
        &m2_c_layout,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.01,
        1,
    )
    .expect("execute CPU AdamW");

    let p_g = to_gpu(&p_c);
    let g_g = to_gpu(&g_c);
    let mut m1_g = Tensor::from_slice_on(vec![n], &m1_init, &w).expect("construct tensor");
    let mut m2_g = Tensor::from_slice_on(vec![n], &m2_init, &w).expect("construct tensor");
    let mut p_g_mut = p_g.clone();
    let p_g_layout = p_g_mut.layout().clone();
    let g_g_layout = g_g.layout().clone();
    let m1_g_layout = m1_g.layout().clone();
    let m2_g_layout = m2_g.layout().clone();
    w.adamw_step(
        p_g_mut.storage_mut().expect("access tensor storage"),
        &p_g_layout,
        g_g.storage(),
        &g_g_layout,
        m1_g.storage_mut().expect("access tensor storage"),
        &m1_g_layout,
        m2_g.storage_mut().expect("access tensor storage"),
        &m2_g_layout,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.01,
        1,
    )
    .expect("execute WGPU AdamW");

    assert_parity(
        "adamw_step",
        p_c_mut.as_slice(),
        to_cpu(&p_g_mut).as_slice(),
    );
}
