//! Differential parity for embedding lookup and gradient operations.
//!
//! Functions exercised:
//!   `embedding`                           - index lookup into weight matrix
//!   `embedding_backward`                  - gradient w.r.t. weight (accumulate)
//!   `embedding_backward_with_padding_idx` - same but skip a designated row
//!
//! All reference values are integer-valued (exact in f64) so assertions use
//! `assert_eq!` without an epsilon band.  SequentialBackend and MoiraiBackend
//! receive identical inputs and must return identical results.

use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_tensor::Tensor;

fn weights<B>(backend: &B) -> Tensor<f64, B>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    // weight [3, 4]: row 0=[1,2,3,4], row 1=[5,6,7,8], row 2=[9,10,11,12]
    Tensor::from_slice_on(
        vec![3, 4],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        backend,
    )
}

// EMBEDDING

fn check_embedding<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    let w = weights(backend);

    // indices [2] = [0, 2] -> output [2, 4] = [row0, row2]
    // = [[1,2,3,4],[9,10,11,12]]
    let idx = Tensor::from_slice_on(vec![2], &[0.0_f64, 2.0], backend);
    let out = coeus_ops::embedding(&w, &idx, backend);
    assert_eq!(out.shape(), &[2, 4], "embedding 1d-index shape");
    assert_eq!(
        out.as_slice(),
        &[1.0_f64, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0],
        "embedding 1d-index"
    );

    // indices [1] = [1] -> output [1, 4] = [[5,6,7,8]]
    let idx_single = Tensor::from_slice_on(vec![1], &[1.0_f64], backend);
    let out_single = coeus_ops::embedding(&w, &idx_single, backend);
    assert_eq!(out_single.shape(), &[1, 4], "embedding single-index shape");
    assert_eq!(
        out_single.as_slice(),
        &[5.0_f64, 6.0, 7.0, 8.0],
        "embedding single-index"
    );

    // 2D indices [2, 2] = [[0,1],[2,0]] -> output [2, 2, 4]
    // out[0,0]=row0, out[0,1]=row1, out[1,0]=row2, out[1,1]=row0
    let idx_2d = Tensor::from_slice_on(vec![2, 2], &[0.0_f64, 1.0, 2.0, 0.0], backend);
    let out_2d = coeus_ops::embedding(&w, &idx_2d, backend);
    assert_eq!(out_2d.shape(), &[2, 2, 4], "embedding 2d-index shape");
    assert_eq!(
        out_2d.as_slice(),
        &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0],
        "embedding 2d-index"
    );
}

// EMBEDDING_BACKWARD

fn check_embedding_backward<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // grad_out [2, 4] = all-ones; indices [2] = [0, 2]
    // grad_weight[0] += grad_out[0] = [1,1,1,1]
    // grad_weight[2] += grad_out[1] = [1,1,1,1]
    // grad_weight[1] unchanged = [0,0,0,0]
    // -> [[1,1,1,1],[0,0,0,0],[1,1,1,1]]
    let grad_out = Tensor::from_slice_on(vec![2, 4], &[1.0_f64; 8], backend);
    let idx = Tensor::from_slice_on(vec![2], &[0.0_f64, 2.0], backend);
    let gw = coeus_ops::embedding_backward(&grad_out, &idx, 3, backend);
    assert_eq!(gw.shape(), &[3, 4], "embedding_backward shape");
    assert_eq!(
        gw.as_slice(),
        &[1.0_f64, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "embedding_backward"
    );

    // Repeated index: indices=[0,0]; both grad rows scatter onto weight row 0.
    // grad_weight[0] = [1,1,1,1] + [1,1,1,1] = [2,2,2,2]
    let grad_out2 = Tensor::from_slice_on(vec![2, 4], &[1.0_f64; 8], backend);
    let idx_rep = Tensor::from_slice_on(vec![2], &[0.0_f64, 0.0], backend);
    let gw2 = coeus_ops::embedding_backward(&grad_out2, &idx_rep, 3, backend);
    assert_eq!(gw2.shape(), &[3, 4], "embedding_backward repeated shape");
    assert_eq!(
        gw2.as_slice(),
        &[2.0_f64, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "embedding_backward repeated"
    );
}

// EMBEDDING_BACKWARD_WITH_PADDING_IDX

fn check_embedding_backward_padding<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // Same as above but with padding_idx=Some(0): grad at index 0 is suppressed.
    // indices=[0,2], grad_out=all-ones
    // weight[0] is padding: zero gradient even though index 0 appears.
    // weight[1]: not referenced -> 0
    // weight[2]: grad_out[1] = [1,1,1,1]
    // -> [[0,0,0,0],[0,0,0,0],[1,1,1,1]]
    let grad_out = Tensor::from_slice_on(vec![2, 4], &[1.0_f64; 8], backend);
    let idx = Tensor::from_slice_on(vec![2], &[0.0_f64, 2.0], backend);
    let gw = coeus_ops::embedding_backward_with_padding_idx(&grad_out, &idx, 3, Some(0), backend);
    assert_eq!(gw.shape(), &[3, 4], "embedding_backward_padding shape");
    assert_eq!(
        gw.as_slice(),
        &[0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "embedding_backward_padding"
    );

    // No padding (None): identical to embedding_backward.
    let gw_none = coeus_ops::embedding_backward_with_padding_idx(&grad_out, &idx, 3, None, backend);
    assert_eq!(
        gw_none.as_slice(),
        &[1.0_f64, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "embedding_backward_padding None"
    );
}

// wrappers

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_embedding(backend);
    check_embedding_backward(backend);
    check_embedding_backward_padding(backend);
}

#[test]
fn sequential_embedding_ops_match_reference() {
    let backend = SequentialBackend;
    check_all(&backend);
}

#[test]
fn moirai_embedding_ops_match_reference() {
    let backend = MoiraiBackend;
    check_all(&backend);
}
