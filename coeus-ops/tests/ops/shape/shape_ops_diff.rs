//! Differential parity for shape-manipulation operations.
//!
//! Functions exercised:
//!   `flip`             - reverse elements along an axis
//!   `roll`             - circular shift along an axis
//!   `tril`             - lower-triangular mask (k-th diagonal offset)
//!   `triu`             - upper-triangular mask
//!   `sort`             - stable sort returning (values, index) pair
//!   `one_hot`          - integer indices -> float indicator matrix
//!   `repeat_interleave`- repeat each element n times along a dimension
//!   `outer`            - 1-D outer product -> 2-D matrix
//!   `cross`            - 3-element cross product along a dimension
//!
//! All reference values are integer-valued so assertions use `assert_eq!`.
//! SequentialBackend and MoiraiBackend must return bitwise-identical results.

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

// FLIP

fn check_flip<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // 1-D reverse: [1,2,3,4] -> [4,3,2,1]
    let v = t(&[4], &[1.0, 2.0, 3.0, 4.0], backend);
    let f = coeus_ops::flip(&v, 0, backend);
    assert_eq!(f.as_slice(), &[4.0_f64, 3.0, 2.0, 1.0], "flip 1-D");

    // 2-D flip along axis=0 (row order reversed):
    // [[1,2],[3,4]] -> [[3,4],[1,2]]
    let m = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let f0 = coeus_ops::flip(&m, 0, backend);
    assert_eq!(f0.as_slice(), &[3.0_f64, 4.0, 1.0, 2.0], "flip axis=0");

    // 2-D flip along axis=1 (column order reversed):
    // [[1,2],[3,4]] -> [[2,1],[4,3]]
    let f1 = coeus_ops::flip(&m, 1, backend);
    assert_eq!(f1.as_slice(), &[2.0_f64, 1.0, 4.0, 3.0], "flip axis=1");

    // Double-flip is identity.
    let ff = coeus_ops::flip(&f1, 1, backend);
    assert_eq!(ff.as_slice(), m.as_slice(), "flip double identity");
}

// ROLL

fn check_roll<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // roll([0,1,2,3], shift=1, dim=0): [3,0,1,2]
    let v = t(&[4], &[0.0, 1.0, 2.0, 3.0], backend);
    let r1 = coeus_ops::roll(&v, &[1], &[0], backend);
    assert_eq!(r1.as_slice(), &[3.0_f64, 0.0, 1.0, 2.0], "roll +1");

    // roll shift=-1: [1,2,3,0]
    let rm = coeus_ops::roll(&v, &[-1], &[0], backend);
    assert_eq!(rm.as_slice(), &[1.0_f64, 2.0, 3.0, 0.0], "roll -1");

    // roll shift=4 (full period) -> identity
    let r4 = coeus_ops::roll(&v, &[4], &[0], backend);
    assert_eq!(r4.as_slice(), v.as_slice(), "roll full period = identity");

    // 2-D roll along axis=1 by shift=1:
    // [[1,2,3],[4,5,6]] -> [[3,1,2],[6,4,5]]
    let m = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend);
    let rm2 = coeus_ops::roll(&m, &[1], &[1], backend);
    assert_eq!(
        rm2.as_slice(),
        &[3.0_f64, 1.0, 2.0, 6.0, 4.0, 5.0],
        "roll 2-D axis=1"
    );
}

// TRIL / TRIU

fn check_tril_triu<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // [[1,2,3],[4,5,6],[7,8,9]] shape [3,3]
    let m = t(
        &[3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        backend,
    );

    // tril(k=0): keep j <= i
    // [[1,0,0],[4,5,0],[7,8,9]]
    let l = coeus_ops::tril(&m, 0, backend);
    assert_eq!(
        l.as_slice(),
        &[1.0_f64, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0],
        "tril k=0"
    );

    // triu(k=0): keep j >= i
    // [[1,2,3],[0,5,6],[0,0,9]]
    let u = coeus_ops::triu(&m, 0, backend);
    assert_eq!(
        u.as_slice(),
        &[1.0_f64, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0],
        "triu k=0"
    );

    // tril(k=1): keep j <= i+1 (includes one super-diagonal)
    // [[1,2,0],[4,5,6],[7,8,9]]
    let l1 = coeus_ops::tril(&m, 1, backend);
    assert_eq!(
        l1.as_slice(),
        &[1.0_f64, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "tril k=1"
    );

    // triu(k=1): keep j >= i+1 (strict super-diagonal)
    // [[0,2,3],[0,0,6],[0,0,0]]
    let u1 = coeus_ops::triu(&m, 1, backend);
    assert_eq!(
        u1.as_slice(),
        &[0.0_f64, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0],
        "triu k=1"
    );

    // tril + triu with k=0 should sum to m + diag(m) (diagonal counted twice).
    // l[i,j] + u[i,j] = m[i,j]*2 if i==j else m[i,j].
    // Check that off-diagonal elements of l and u don't overlap.
    for (l_val, u_val) in l.as_slice().iter().zip(u.as_slice().iter()) {
        assert!(
            *l_val == 0.0 || *u_val == 0.0 || l_val == u_val,
            "tril+triu overlap: l={l_val} u={u_val}"
        );
    }
}

// SORT

fn check_sort<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // 1-D ascending: [3,1,4,1,5] -> values=[1,1,3,4,5], indices=[1,3,0,2,4]
    let v = t(&[5], &[3.0, 1.0, 4.0, 1.0, 5.0], backend);
    let (sv, si) = coeus_ops::sort(&v, 0, false, backend);
    assert_eq!(sv.shape(), &[5], "sort shape");
    assert_eq!(
        sv.as_slice(),
        &[1.0_f64, 1.0, 3.0, 4.0, 5.0],
        "sort ascending values"
    );
    assert_eq!(
        si.as_slice(),
        &[1.0_f64, 3.0, 0.0, 2.0, 4.0],
        "sort ascending indices"
    );

    // 1-D descending: values=[5,4,3,1,1], indices=[4,2,0,1,3]
    let (sv_d, si_d) = coeus_ops::sort(&v, 0, true, backend);
    assert_eq!(
        sv_d.as_slice(),
        &[5.0_f64, 4.0, 3.0, 1.0, 1.0],
        "sort descending values"
    );
    assert_eq!(
        si_d.as_slice(),
        &[4.0_f64, 2.0, 0.0, 1.0, 3.0],
        "sort descending indices"
    );

    // Already-sorted input: indices are 0,1,2,...
    let sorted = t(&[3], &[1.0, 2.0, 3.0], backend);
    let (ss, si_s) = coeus_ops::sort(&sorted, 0, false, backend);
    assert_eq!(
        ss.as_slice(),
        sorted.as_slice(),
        "sort already-sorted values"
    );
    assert_eq!(
        si_s.as_slice(),
        &[0.0_f64, 1.0, 2.0],
        "sort already-sorted indices"
    );
}

// ONE_HOT

fn check_one_hot<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // indices=[0,2,1], num_classes=3:
    // row 0: [1,0,0], row 1: [0,0,1], row 2: [0,1,0]
    let idx = t(&[3], &[0.0, 2.0, 1.0], backend);
    let oh = coeus_ops::one_hot(&idx, 3, backend);
    assert_eq!(oh.shape(), &[3, 3], "one_hot shape");
    assert_eq!(
        oh.as_slice(),
        &[1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "one_hot"
    );

    // Single index [0], 4 classes -> [1,0,0,0]
    let i0 = t(&[1], &[0.0], backend);
    let oh0 = coeus_ops::one_hot(&i0, 4, backend);
    assert_eq!(oh0.shape(), &[1, 4], "one_hot single shape");
    assert_eq!(oh0.as_slice(), &[1.0_f64, 0.0, 0.0, 0.0], "one_hot single");
}

// REPEAT_INTERLEAVE

fn check_repeat_interleave<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // [1,2,3] repeat=2 along dim 0: [1,1,2,2,3,3]
    let v = t(&[3], &[1.0, 2.0, 3.0], backend);
    let ri = coeus_ops::repeat_interleave(&v, 2, 0, backend);
    assert_eq!(ri.shape(), &[6], "repeat_interleave shape");
    assert_eq!(
        ri.as_slice(),
        &[1.0_f64, 1.0, 2.0, 2.0, 3.0, 3.0],
        "repeat_interleave 1-D"
    );

    // [[1,2],[3,4]] repeat=2 along dim=0: [[1,2],[1,2],[3,4],[3,4]] shape [4,2]
    let m = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let ri2 = coeus_ops::repeat_interleave(&m, 2, 0, backend);
    assert_eq!(ri2.shape(), &[4, 2], "repeat_interleave 2-D dim=0 shape");
    assert_eq!(
        ri2.as_slice(),
        &[1.0_f64, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0],
        "repeat_interleave 2-D dim=0"
    );
}

// OUTER

fn check_outer<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // outer([1,2,3], [4,5]): shape [3,2]
    // [[1*4,1*5],[2*4,2*5],[3*4,3*5]] = [[4,5],[8,10],[12,15]]
    let a = t(&[3], &[1.0, 2.0, 3.0], backend);
    let b = t(&[2], &[4.0, 5.0], backend);
    let o = coeus_ops::outer(&a, &b, backend);
    assert_eq!(o.shape(), &[3, 2], "outer shape");
    assert_eq!(
        o.as_slice(),
        &[4.0_f64, 5.0, 8.0, 10.0, 12.0, 15.0],
        "outer"
    );

    // outer([1,0], [0,1]) = [[0,1],[0,0]] (unit vectors)
    let e0 = t(&[2], &[1.0, 0.0], backend);
    let e1 = t(&[2], &[0.0, 1.0], backend);
    let oe = coeus_ops::outer(&e0, &e1, backend);
    assert_eq!(
        oe.as_slice(),
        &[0.0_f64, 1.0, 0.0, 0.0],
        "outer unit vectors"
    );
}

// CROSS

fn check_cross<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // [1,0,0] x [0,1,0] = [0,0,1] (e1 x e2 = e3)
    let a = t(&[3], &[1.0, 0.0, 0.0], backend);
    let b = t(&[3], &[0.0, 1.0, 0.0], backend);
    let c = coeus_ops::cross(&a, &b, 0);
    assert_eq!(c.as_slice(), &[0.0_f64, 0.0, 1.0], "cross e1xe2=e3");

    // [0,1,0] x [0,0,1] = [1,0,0] (e2 x e3 = e1)
    let e2 = t(&[3], &[0.0, 1.0, 0.0], backend);
    let e3 = t(&[3], &[0.0, 0.0, 1.0], backend);
    let c2 = coeus_ops::cross(&e2, &e3, 0);
    assert_eq!(c2.as_slice(), &[1.0_f64, 0.0, 0.0], "cross e2xe3=e1");

    // Anti-commutativity: a x b = -(b x a)
    let ba = coeus_ops::cross(&b, &a, 0);
    for (x, y) in c.as_slice().iter().zip(ba.as_slice().iter()) {
        assert_eq!(*x, -*y, "cross anti-commutativity");
    }

    // Batch: [1,3] shape with 1 batch
    let av = t(&[1, 3], &[1.0, 0.0, 0.0], backend);
    let bv = t(&[1, 3], &[0.0, 1.0, 0.0], backend);
    let cv = coeus_ops::cross(&av, &bv, 1);
    assert_eq!(cv.shape(), &[1, 3], "cross batch shape");
    assert_eq!(cv.as_slice(), &[0.0_f64, 0.0, 1.0], "cross batch");
}

// wrappers

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_flip(backend);
    check_roll(backend);
    check_tril_triu(backend);
    check_sort(backend);
    check_one_hot(backend);
    check_repeat_interleave(backend);
    check_outer(backend);
    check_cross(backend);
}

#[test]
fn sequential_shape_ops_match_reference() {
    let backend = SequentialBackend;
    check_all(&backend);
}

#[test]
fn moirai_shape_ops_match_reference() {
    let backend = MoiraiBackend;
    check_all(&backend);
}
