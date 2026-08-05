//! Differential parity for `prod`, `tile`, and `masked_fill`.
//!
//! Functions exercised:
//!   `prod`        - global product of all elements (returns T)
//!   `tile`        - replicate tensor along each dimension by given counts
//!   `masked_fill` - replace elements where mask != 0 with a fill value
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
{
    Tensor::from_slice_on(shape.to_vec(), vals, backend)
}

// PROD

fn check_prod<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
{
    // prod([1,2,3,4]) = 24
    let v = t(&[4], &[1.0, 2.0, 3.0, 4.0], backend);
    assert_eq!(coeus_ops::prod(&v, backend), 24.0_f64, "prod 1-D");

    // prod([5]) = 5 (single element)
    let s = t(&[1], &[5.0], backend);
    assert_eq!(coeus_ops::prod(&s, backend), 5.0_f64, "prod single");

    // prod containing zero: any product with 0 = 0
    let z = t(&[3], &[2.0, 0.0, 7.0], backend);
    assert_eq!(coeus_ops::prod(&z, backend), 0.0_f64, "prod with zero");

    // 2-D [[1,2],[3,4]]: global prod = 1*2*3*4 = 24
    let m = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    assert_eq!(coeus_ops::prod(&m, backend), 24.0_f64, "prod 2-D");
}

// TILE

fn check_tile<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // 1-D [1,2,3], reps=[3]: repeat 3x -> [1,2,3,1,2,3,1,2,3]
    let v = t(&[3], &[1.0, 2.0, 3.0], backend);
    let r1 = coeus_ops::tile(&v, &[3], backend);
    assert_eq!(r1.shape(), &[9], "tile 1-D reps=3 shape");
    assert_eq!(
        r1.as_slice(),
        &[1.0_f64, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "tile 1-D reps=3"
    );

    // 2-D [[1,2],[3,4]] shape [2,2], reps=[2,3]:
    // -> shape [4,6]: each row repeated 3x, then stacked 2x
    // row0_tiled: [1,2,1,2,1,2]  row1_tiled: [3,4,3,4,3,4]
    // result: [row0_tiled, row1_tiled, row0_tiled, row1_tiled]
    let m = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let r2 = coeus_ops::tile(&m, &[2, 3], backend);
    assert_eq!(r2.shape(), &[4, 6], "tile 2-D shape");
    assert_eq!(
        r2.as_slice(),
        &[
            1.0_f64, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0,
            1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        ],
        "tile 2-D reps=[2,3]"
    );

    // reps=[1] on a 1-D tensor is identity.
    let id = coeus_ops::tile(&v, &[1], backend);
    assert_eq!(id.as_slice(), v.as_slice(), "tile reps=[1] identity");
}

// MASKED_FILL

fn check_masked_fill<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // 1-D [1,2,3,4], mask=[0,1,0,1], fill=99:
    // positions where mask != 0 become 99 -> [1,99,3,99]
    let inp = t(&[4], &[1.0, 2.0, 3.0, 4.0], backend);
    let mask = t(&[4], &[0.0, 1.0, 0.0, 1.0], backend);
    let out = coeus_ops::masked_fill(&inp, &mask, 99.0, backend);
    assert_eq!(out.shape(), &[4], "masked_fill 1-D shape");
    assert_eq!(
        out.as_slice(),
        &[1.0_f64, 99.0, 3.0, 99.0],
        "masked_fill 1-D"
    );

    // All-false mask: output equals input.
    let all_false = t(&[4], &[0.0; 4], backend);
    let id = coeus_ops::masked_fill(&inp, &all_false, 99.0, backend);
    assert_eq!(
        id.as_slice(),
        inp.as_slice(),
        "masked_fill all-false = identity"
    );

    // All-true mask: output is constant fill.
    let all_true = t(&[4], &[1.0; 4], backend);
    let fill = coeus_ops::masked_fill(&inp, &all_true, -7.0, backend);
    assert_eq!(
        fill.as_slice(),
        &[-7.0_f64; 4],
        "masked_fill all-true = fill"
    );

    // 2-D [[1,2,3],[4,5,6]] shape [2,3]:
    // mask [[1,0,0],[0,1,0]] -> fill positions (0,0) and (1,1) with 0.
    // -> [[0,2,3],[4,0,6]]
    let m = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend);
    let mask2 = t(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], backend);
    let out2 = coeus_ops::masked_fill(&m, &mask2, 0.0, backend);
    assert_eq!(out2.shape(), &[2, 3], "masked_fill 2-D shape");
    assert_eq!(
        out2.as_slice(),
        &[0.0_f64, 2.0, 3.0, 4.0, 0.0, 6.0],
        "masked_fill 2-D"
    );
}

// wrappers

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_prod(backend);
    check_tile(backend);
    check_masked_fill(backend);
}

#[test]
fn sequential_prod_tile_maskfill_match_reference() {
    let backend = SequentialBackend;
    check_all(&backend);
}

#[test]
fn moirai_prod_tile_maskfill_match_reference() {
    let backend = MoiraiBackend;
    check_all(&backend);
}
