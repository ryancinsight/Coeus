use coeus_core::{CpuAddressableStorage, Layout, SequentialBackend, Shape, Strides};
use coeus_tensor::{broadcast::broadcast_shapes, Tensor};

#[test]
fn tensor_broadcast_matches_leto_layout_metadata_and_values() {
    let backend = SequentialBackend;
    let row = Tensor::<i32, SequentialBackend>::from_slice_on([1, 3], &[10, 20, 30], &backend);

    let broadcasted = row.broadcast([2, 3]);
    let direct = coeus_leto::to_leto_layout::<2>(row.layout())
        .unwrap()
        .broadcast::<2>([2, 3])
        .unwrap();

    assert_eq!(broadcasted.shape(), direct.shape);
    assert_eq!(broadcasted.strides(), &[0, 1]);
    assert_eq!(broadcasted.storage().as_slice(), row.storage().as_slice());
    assert_eq!(
        broadcasted.storage().as_slice()[broadcasted.layout().physical_index(&[0, 0])],
        10
    );
    assert_eq!(
        broadcasted.storage().as_slice()[broadcasted.layout().physical_index(&[0, 2])],
        30
    );
    assert_eq!(
        broadcasted.storage().as_slice()[broadcasted.layout().physical_index(&[1, 0])],
        10
    );
    assert_eq!(
        broadcasted.storage().as_slice()[broadcasted.layout().physical_index(&[1, 2])],
        30
    );
}

#[test]
fn tensor_broadcast_preserves_source_offset_for_sliced_views() {
    let backend = SequentialBackend;
    let matrix = Tensor::<i32, SequentialBackend>::from_slice_on(
        [3, 3],
        &[1, 2, 3, 4, 5, 6, 7, 8, 9],
        &backend,
    );
    let row = matrix.slice(&[(1, 2), (0, 3)]);
    let broadcasted = row.broadcast([2, 3]);

    assert_eq!(broadcasted.shape(), &[2, 3]);
    assert_eq!(broadcasted.strides(), &[0, 1]);
    assert_eq!(broadcasted.layout().offset(), 3);
    assert_eq!(
        broadcasted.storage().as_slice()[broadcasted.layout().physical_index(&[0, 0])],
        4
    );
    assert_eq!(
        broadcasted.storage().as_slice()[broadcasted.layout().physical_index(&[1, 2])],
        6
    );
}

#[test]
fn tensor_scalar_broadcast_matches_leto_layout_metadata_and_values() {
    let backend = SequentialBackend;
    let scalar = Tensor::<i32, SequentialBackend>::from_slice_on([], &[7], &backend);
    let broadcasted = scalar.broadcast([2, 2]);

    assert_eq!(broadcasted.shape(), &[2, 2]);
    assert_eq!(broadcasted.strides(), &[0, 0]);
    assert_eq!(
        broadcasted.storage().as_slice()[broadcasted.layout().physical_index(&[0, 0])],
        7
    );
    assert_eq!(
        broadcasted.storage().as_slice()[broadcasted.layout().physical_index(&[1, 1])],
        7
    );
}

#[test]
fn broadcast_shape_helper_routes_through_leto_bridge() {
    let shape = broadcast_shapes(&[2, 1, 3], &[1, 4, 1]).unwrap();
    assert_eq!(shape.as_ref(), &[2, 4, 3]);
    assert!(broadcast_shapes(&[2, 2], &[3, 2]).is_none());
}

#[test]
#[should_panic(expected = "coeus-leto broadcast validation failed")]
fn tensor_broadcast_rejects_incompatible_shape() {
    let backend = SequentialBackend;
    let tensor = Tensor::<i32, SequentialBackend>::from_slice_on([2, 2], &[1, 2, 3, 4], &backend);
    let _ = tensor.broadcast([3, 2]);
}

#[test]
fn leto_bridge_matches_manual_strided_layout_case() {
    let source = Layout::from_shape_strides(
        Shape::from(vec![1usize, 3]),
        Strides::from_slice(&[3usize, 1]),
        4,
    );
    let bridge = coeus_leto::broadcast_layout(&source, &[2, 3]).unwrap();
    let direct = coeus_leto::to_leto_layout::<2>(&source)
        .unwrap()
        .broadcast::<2>([2, 3])
        .unwrap();

    assert_eq!(bridge.shape(), direct.shape);
    assert_eq!(bridge.strides(), &[0, 1]);
    assert_eq!(bridge.offset(), 4);
}
