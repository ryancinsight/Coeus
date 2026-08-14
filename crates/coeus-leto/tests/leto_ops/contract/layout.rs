#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use super::support::layout;
use super::{
    broadcast_layout, broadcast_shape, concat_values, contiguous_values, from_shape_fn_values,
    normal_values, pad_values, permute_layout, reshape_layout, split_values, stack_values,
    to_leto_view, uniform_values, CpuStorage, Layout, Shape, Storage, Strides,
};
use coeus_core::CpuAddressableStorage;

#[test]
fn pad_dispatch_covers_strided_input_view() {
    let storage = vec![1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
    let transposed = Layout::from_shape_strides(
        Shape::from(vec![2usize, 3]),
        Strides::from_slice(&[1usize, 2]),
        0,
    );

    let padded = pad_values(&transposed, &storage, &[(1, 0), (0, 1)], -1.0).unwrap();

    assert_eq!(
        padded,
        vec![-1.0, -1.0, -1.0, -1.0, 1.0, 2.0, 3.0, -1.0, 4.0, 5.0, 6.0, -1.0]
    );
}

#[test]
fn concat_dispatch_covers_strided_input_views() {
    let first_storage = vec![1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
    let second_storage = vec![7.0f64, 10.0, 8.0, 11.0, 9.0, 12.0];
    let transposed = Layout::from_shape_strides(
        Shape::from(vec![2usize, 3]),
        Strides::from_slice(&[1usize, 2]),
        0,
    );

    let concatenated = concat_values(
        &[&transposed, &transposed],
        &[&first_storage, &second_storage],
        0,
    )
    .unwrap();

    assert_eq!(
        concatenated,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn split_dispatch_covers_strided_input_view() {
    let storage = vec![1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
    let transposed = Layout::from_shape_strides(
        Shape::from(vec![2usize, 3]),
        Strides::from_slice(&[1usize, 2]),
        0,
    );

    let chunks = split_values(&transposed, &storage, 1, &[2, 1]).unwrap();

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], vec![1.0, 2.0, 4.0, 5.0]);
    assert_eq!(chunks[1], vec![3.0, 6.0]);
}

#[test]
fn stack_dispatch_covers_strided_input_views() {
    let first_storage = vec![1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
    let second_storage = vec![7.0f64, 10.0, 8.0, 11.0, 9.0, 12.0];
    let transposed = Layout::from_shape_strides(
        Shape::from(vec![2usize, 3]),
        Strides::from_slice(&[1usize, 2]),
        0,
    );

    let stacked = stack_values(
        &[&transposed, &transposed],
        &[&first_storage, &second_storage],
        1,
    )
    .unwrap();
    let first_view = to_leto_view::<f64, 2>(&transposed, &first_storage).unwrap();
    let second_view = to_leto_view::<f64, 2>(&transposed, &second_storage).unwrap();
    let direct = leto::application::stack::<f64, 2, 3>(&[first_view, second_view], 1).unwrap();

    assert_eq!(stacked, direct.storage().as_slice());
    assert_eq!(
        stacked,
        vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn random_dispatch_matches_leto_seeded_constructors() {
    let uniform = uniform_values(&[2usize, 3], -2.0f64, 5.0, 42).unwrap();
    let direct_uniform = leto_ops::uniform_with_seed([2usize, 3], -2.0f64, 5.0, 42).unwrap();
    assert_eq!(uniform, direct_uniform.storage().as_slice());

    let normal = normal_values(&[2usize, 3], 1.0f64, 2.0, 11).unwrap();
    let direct_normal = leto_ops::normal_with_seed([2usize, 3], 1.0f64, 2.0, 11).unwrap();
    assert_eq!(normal, direct_normal.storage().as_slice());
}

#[test]
fn contiguous_dispatch_matches_leto_view_materialization() {
    let data = (0..12).collect::<Vec<i32>>();
    let source = CpuStorage::from_slice(&data);
    let sliced = layout(&[3, 4]).slice(&[(0, 3), (1, 4)]);
    let view = Layout::from_shape_strides(
        Shape::from(vec![3, 3]),
        Strides::from_slice(&[sliced.strides()[1], sliced.strides()[0]]),
        sliced.offset(),
    );

    let values = contiguous_values(&view, source.as_slice()).unwrap();
    let direct = to_leto_view::<i32, 2>(&view, source.as_slice())
        .unwrap()
        .to_contiguous();

    assert_eq!(values, direct.storage().as_slice());
    assert_eq!(values, vec![1, 5, 9, 2, 6, 10, 3, 7, 11]);
}

#[test]
fn reshape_layout_dispatch_matches_leto_validation() {
    let sliced = layout(&[8]).slice(&[(2, 6)]);
    let reshaped = reshape_layout(&sliced, &[2, 2]).unwrap();
    let direct = coeus_leto::to_leto_layout::<1>(&sliced)
        .unwrap()
        .reshape::<2>([2, 2])
        .unwrap();

    assert_eq!(reshaped.shape(), direct.shape);
    assert_eq!(reshaped.strides(), &[2, 1]);
    assert_eq!(reshaped.offset(), 2);

    let transposed =
        Layout::from_shape_strides(Shape::from(vec![3, 2]), Strides::from_slice(&[1, 3]), 0);
    assert!(reshape_layout(&transposed, &[6]).is_err());
}

#[test]
fn permute_layout_dispatch_matches_leto_validation() {
    let source = layout(&[2, 3, 4]);
    let permuted = permute_layout(&source, &[2, 0, 1]).unwrap();
    let direct = coeus_leto::to_leto_layout::<3>(&source)
        .unwrap()
        .transpose([2, 0, 1])
        .unwrap();

    assert_eq!(permuted.shape(), direct.shape);
    assert_eq!(permuted.strides(), &[1, 12, 4]);
    assert_eq!(permuted.offset(), 0);
    assert!(permute_layout(&source, &[0, 0, 1]).is_err());
}

#[test]
fn broadcast_layout_dispatch_matches_leto_validation() {
    let row = layout(&[1, 3]);
    let broadcasted = broadcast_layout(&row, &[2, 3]).unwrap();
    let direct = coeus_leto::to_leto_layout::<2>(&row)
        .unwrap()
        .broadcast::<2>([2, 3])
        .unwrap();

    assert_eq!(broadcasted.shape(), direct.shape);
    assert_eq!(broadcasted.strides(), &[0, 1]);
    assert_eq!(broadcasted.offset(), 0);
    assert_eq!(broadcast_shape(&[2, 1], &[1, 3]).unwrap(), vec![2, 3]);
    assert!(broadcast_shape(&[2, 2], &[3, 2]).is_err());
}

#[test]
fn shape_function_dispatch_matches_leto_coordinate_order() {
    let values = from_shape_fn_values(&[2usize, 3, 2], |index| {
        i32::try_from(index[0] * 100 + index[1] * 10 + index[2]).unwrap()
    })
    .unwrap();
    let direct = leto::Array::<i32, _, 3>::from_shape_fn([2, 3, 2], |index| {
        i32::try_from(index[0] * 100 + index[1] * 10 + index[2]).unwrap()
    });

    assert_eq!(values, direct.storage().as_slice());
    assert_eq!(
        values,
        vec![0, 1, 10, 11, 20, 21, 100, 101, 110, 111, 120, 121]
    );
}

#[test]
fn view_over_cpu_storage_reads_logical_values() {
    // Prove the adapter binds directly to coeus CpuStorage slices.
    let storage = CpuStorage::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let la = layout(&[2, 3]);
    let view = to_leto_view::<f64, 2>(&la, storage.as_slice()).unwrap();
    assert_eq!(view.shape(), [2, 3]);
    assert_eq!(*view.get([1, 2]).unwrap(), 6.0);
}
