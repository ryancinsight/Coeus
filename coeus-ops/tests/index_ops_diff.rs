//! Differential parity for index-based selection and scatter operations.
//!
//! Functions exercised:
//!   `gather`        - element-wise index look-up along one dim (torch.gather semantics)
//!   `index_select`  - slice selection along one dim with a 1-D index tensor
//!   `index_put`     - scatter-assign values at 1-D row indices (first-dim only)
//!   `scatter_add`   - accumulate source into output at gather-compatible indices
//!   `masked_select` - extract elements where mask is non-zero (1-D result)
//!   `bmm`           - batched matrix multiply [B,M,K] x [B,K,N] -> [B,M,N]
//!
//! All reference values are integer-valued, so assertions use `assert_eq!`
//! without an epsilon band.  SequentialBackend and MoiraiBackend receive
//! identical inputs and must return identical results; divergence signals a
//! dispatch bug.

use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_tensor::Tensor;

fn t<B>(shape: &[usize], vals: &[f64], backend: &B) -> Tensor<f64, B>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Tensor::from_slice_on(shape.to_vec(), vals, backend)
}

// GATHER

fn check_gather<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // input [[1,2,3],[4,5,6]] shape [2,3], dim=1
    // index [[0,2],[1,0]] shape [2,2]
    // out[0,0]=input[0,index[0,0]]=input[0,0]=1
    // out[0,1]=input[0,index[0,1]]=input[0,2]=3
    // out[1,0]=input[1,index[1,0]]=input[1,1]=5
    // out[1,1]=input[1,index[1,1]]=input[1,0]=4
    // -> [[1,3],[5,4]]
    let inp = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend);
    let idx = t(&[2, 2], &[0.0, 2.0, 1.0, 0.0], backend);
    let out = coeus_ops::gather(&inp, 1, &idx, backend);
    assert_eq!(out.shape(), &[2, 2], "gather shape");
    assert_eq!(out.as_slice(), &[1.0_f64, 3.0, 5.0, 4.0], "gather dim=1");

    // gather along dim=0: input [[1,2,3],[4,5,6]] shape [2,3], dim=0
    // index [[0,1,0]] shape [1,3]
    // out[0,j]=input[index[0,j],j]: out=[input[0,0],input[1,1],input[0,2]]=[1,5,3]
    let idx0 = t(&[1, 3], &[0.0, 1.0, 0.0], backend);
    let out0 = coeus_ops::gather(&inp, 0, &idx0, backend);
    assert_eq!(out0.shape(), &[1, 3], "gather dim=0 shape");
    assert_eq!(out0.as_slice(), &[1.0_f64, 5.0, 3.0], "gather dim=0");
}

// INDEX_SELECT

fn check_index_select<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // input [[1,2,3],[4,5,6]] shape [2,3], dim=1, index=[2,0] (1-D)
    // selects columns 2 and 0: [[3,1],[6,4]], shape [2,2]
    let inp = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend);
    let idx = t(&[2], &[2.0, 0.0], backend);
    let out = coeus_ops::index_select(&inp, 1, &idx, backend);
    assert_eq!(out.shape(), &[2, 2], "index_select dim=1 shape");
    assert_eq!(
        out.as_slice(),
        &[3.0_f64, 1.0, 6.0, 4.0],
        "index_select dim=1"
    );

    // dim=0, index=[1] selects row 1: [[4,5,6]], shape [1,3]
    let idx0 = t(&[1], &[1.0], backend);
    let out0 = coeus_ops::index_select(&inp, 0, &idx0, backend);
    assert_eq!(out0.shape(), &[1, 3], "index_select dim=0 shape");
    assert_eq!(out0.as_slice(), &[4.0_f64, 5.0, 6.0], "index_select dim=0");

    // Repeated index: [0, 0] duplicates row 0 twice.
    let idx_rep = t(&[2], &[0.0, 0.0], backend);
    let out_rep = coeus_ops::index_select(&inp, 0, &idx_rep, backend);
    assert_eq!(out_rep.shape(), &[2, 3], "index_select repeat shape");
    assert_eq!(
        out_rep.as_slice(),
        &[1.0_f64, 2.0, 3.0, 1.0, 2.0, 3.0],
        "index_select repeat"
    );
}

// INDEX_PUT

fn check_index_put<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // Non-accumulate: write [10,20] at row 1 of [[1,2],[3,4],[5,6]].
    // output: [[1,2],[10,20],[5,6]]
    let inp = t(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend);
    let indices = t(&[1], &[1.0], backend);
    let values = t(&[1, 2], &[10.0, 20.0], backend);
    let out = coeus_ops::index_put(&inp, &indices, &values, false, backend);
    assert_eq!(out.shape(), &[3, 2], "index_put shape");
    assert_eq!(
        out.as_slice(),
        &[1.0_f64, 2.0, 10.0, 20.0, 5.0, 6.0],
        "index_put replace"
    );

    // Accumulate: add [1,1] at row 0 of [[1,2],[3,4]].
    // output[0] = [1+1, 2+1] = [2, 3]; row 1 unchanged.
    let inp2 = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let idx2 = t(&[1], &[0.0], backend);
    let vals2 = t(&[1, 2], &[1.0, 1.0], backend);
    let out2 = coeus_ops::index_put(&inp2, &idx2, &vals2, true, backend);
    assert_eq!(
        out2.as_slice(),
        &[2.0_f64, 3.0, 3.0, 4.0],
        "index_put accumulate"
    );
}

// SCATTER_ADD

fn check_scatter_add<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // scatter_add is the backward of gather.
    // input=zeros[2,3], dim=1, index[[0,2],[1,0]], src[[1,2],[3,4]]
    // out[0,0]+=src[0,0]=1, out[0,2]+=src[0,1]=2
    // out[1,1]+=src[1,0]=3, out[1,0]+=src[1,1]=4
    // -> [[1,0,2],[4,3,0]]
    let inp = t(&[2, 3], &[0.0; 6], backend);
    let idx = t(&[2, 2], &[0.0, 2.0, 1.0, 0.0], backend);
    let src = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let out = coeus_ops::scatter_add(&inp, 1, &idx, &src, backend);
    assert_eq!(out.shape(), &[2, 3], "scatter_add shape");
    assert_eq!(
        out.as_slice(),
        &[1.0_f64, 0.0, 2.0, 4.0, 3.0, 0.0],
        "scatter_add"
    );

    // Non-zero base: accumulates on top of existing values.
    // input=[[10,0,0],[0,10,0]], same idx/src as above
    // out[0,0]=10+1=11, out[0,2]=0+2=2, out[1,1]=10+3=13, out[1,0]=0+4=4
    // -> [[11,0,2],[4,13,0]]
    let inp2 = t(&[2, 3], &[10.0, 0.0, 0.0, 0.0, 10.0, 0.0], backend);
    let out2 = coeus_ops::scatter_add(&inp2, 1, &idx, &src, backend);
    assert_eq!(
        out2.as_slice(),
        &[11.0_f64, 0.0, 2.0, 4.0, 13.0, 0.0],
        "scatter_add nonzero base"
    );
}

// MASKED_SELECT

fn check_masked_select<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // 1-D: [10,20,30,40,50], mask=[1,0,1,0,1] -> [10,30,50]
    let inp = t(&[5], &[10.0, 20.0, 30.0, 40.0, 50.0], backend);
    let mask = t(&[5], &[1.0, 0.0, 1.0, 0.0, 1.0], backend);
    let out = coeus_ops::masked_select(&inp, &mask, backend);
    assert_eq!(out.shape(), &[3], "masked_select 1-D shape");
    assert_eq!(out.as_slice(), &[10.0_f64, 30.0, 50.0], "masked_select 1-D");

    // 2-D: [[1,2],[3,4]], mask=[[1,0],[0,1]] -> [1,4]
    let inp2 = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let mask2 = t(&[2, 2], &[1.0, 0.0, 0.0, 1.0], backend);
    let out2 = coeus_ops::masked_select(&inp2, &mask2, backend);
    assert_eq!(out2.shape(), &[2], "masked_select 2-D shape");
    assert_eq!(out2.as_slice(), &[1.0_f64, 4.0], "masked_select 2-D");

    // All-false mask -> empty result.
    let all_false = t(&[5], &[0.0; 5], backend);
    let out3 = coeus_ops::masked_select(&inp, &all_false, backend);
    assert!(out3.as_slice().is_empty(), "masked_select all-false");
}

// BMM

fn check_bmm<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // batch=2, M=2, K=2, N=2
    // a[0]=[[1,2],[3,4]], b[0]=I -> a[0]@I = [[1,2],[3,4]]
    // a[1]=[[5,0],[0,5]], b[1]=2I -> a[1]@(2I) = [[10,0],[0,10]]
    // Stored row-major: [1,2,3,4, 5,0,0,5] for a; [1,0,0,1, 2,0,0,2] for b
    let a = t(
        &[2, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 5.0],
        backend,
    );
    let b = t(
        &[2, 2, 2],
        &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        backend,
    );
    let out = coeus_ops::bmm(&a, &b, backend);
    assert_eq!(out.shape(), &[2, 2, 2], "bmm shape");
    assert_eq!(
        out.as_slice(),
        &[1.0_f64, 2.0, 3.0, 4.0, 10.0, 0.0, 0.0, 10.0],
        "bmm"
    );

    // batch=1 single matmul: use A=I[1,2,2], B=scaling[1,2,2].
    let a2 = t(&[1, 2, 2], &[1.0, 0.0, 0.0, 1.0], backend);
    let b2 = t(&[1, 2, 2], &[3.0, 0.0, 0.0, 5.0], backend);
    let out2 = coeus_ops::bmm(&a2, &b2, backend);
    assert_eq!(out2.shape(), &[1, 2, 2], "bmm batch=1 shape");
    assert_eq!(
        out2.as_slice(),
        &[3.0_f64, 0.0, 0.0, 5.0],
        "bmm batch=1 I@diag"
    );
}

// wrappers

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_gather(backend);
    check_index_select(backend);
    check_index_put(backend);
    check_scatter_add(backend);
    check_masked_select(backend);
    check_bmm(backend);
}

#[test]
fn sequential_index_ops_match_reference() {
    let backend = SequentialBackend;
    check_all(&backend);
}

#[test]
fn moirai_index_ops_match_reference() {
    let backend = MoiraiBackend;
    check_all(&backend);
}
