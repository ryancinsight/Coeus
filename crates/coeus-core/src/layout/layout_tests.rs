//! Tests for the runtime [`Layout`](super::Layout) and
//! [`ConstLayout`](super::ConstLayout) descriptors.

use super::*;
use crate::layout::ConstShape;

#[test]
fn test_const_layout() {
    const SHAPE: ConstShape<3> = ConstShape::new([2, 3, 4]);
    const LAYOUT: ConstLayout<3> = ConstLayout::new(SHAPE);

    // Verify shape
    assert_eq!(LAYOUT.shape.dims, [2, 3, 4]);

    // Verify calculated strides
    assert_eq!(LAYOUT.strides, [12, 4, 1]);

    // Verify contiguity
    assert!(LAYOUT.is_contiguous());

    // Verify non-contiguous case
    const NON_CONTIG: ConstLayout<3> = ConstLayout {
        shape: ConstShape::new([2, 3, 4]),
        strides: [12, 5, 1], // not contiguous
        offset: 0,
    };
    assert!(!NON_CONTIG.is_contiguous());
}

#[test]
fn test_squeeze_unsqueeze() {
    let l = Layout::new([2, 3].into());
    assert_eq!(l.shape(), &[2, 3]);
    assert_eq!(l.strides(), &[3, 1]);

    // Unsqueeze at 0 -> [1, 2, 3]
    let l2 = l.unsqueeze(0);
    assert_eq!(l2.shape(), &[1, 2, 3]);
    assert_eq!(l2.strides(), &[3, 3, 1]);

    // Unsqueeze at 1 -> [2, 1, 3]
    let l3 = l.unsqueeze(1);
    assert_eq!(l3.shape(), &[2, 1, 3]);
    assert_eq!(l3.strides(), &[3, 1, 1]);

    // Unsqueeze at 2 -> [2, 3, 1]
    let l4 = l.unsqueeze(2);
    assert_eq!(l4.shape(), &[2, 3, 1]);
    assert_eq!(l4.strides(), &[3, 1, 1]);

    // Squeeze axis 1 of l3 -> [2, 3]
    let l5 = l3.squeeze(1);
    assert_eq!(l5.shape(), &[2, 3]);
    assert_eq!(l5.strides(), &[3, 1]);

    // Squeeze all on a layout with multiple 1s: [1, 2, 1, 3] -> [2, 3]
    let l_multi =
        Layout::from_shape_strides([1, 2, 1, 3].into(), smallvec::smallvec![6, 3, 3, 1], 0);
    let l_squeezed = l_multi.squeeze_all();
    assert_eq!(l_squeezed.shape(), &[2, 3]);
    assert_eq!(l_squeezed.strides(), &[3, 1]);
}

#[test]
fn split_axis_preserves_strides_and_checked_offsets() {
    let layout = Layout::from_shape_strides([2, 4].into(), smallvec::smallvec![9, 2], 3);
    let (first, second) = layout.split_axis(1, 2).expect("valid split");
    assert_eq!(first.shape(), &[2, 2]);
    assert_eq!(second.shape(), &[2, 2]);
    assert_eq!(first.strides(), &[9, 2]);
    assert_eq!(second.strides(), &[9, 2]);
    assert_eq!(first.offset(), 3);
    assert_eq!(second.offset(), 7);
    assert!(layout.split_axis(2, 0).is_none());
    assert!(layout.split_axis(1, 5).is_none());

    let overflow = Layout::from_shape_strides([2].into(), smallvec::smallvec![usize::MAX], 1);
    assert!(overflow.split_axis(0, 1).is_none());
}
