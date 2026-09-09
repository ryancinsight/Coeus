use coeus_core::Float;
use coeus_ops::{Axis, StaggeredPairOps};

pub(crate) fn staggered_preserves_clones<T: Float, B: StaggeredPairOps<T>>(backend: &B, one: T) {
    use super::optimizer::{assert_values, upload};
    use coeus_core::Layout;
    let two = one + one;
    let three = two + one;
    let four = two + two;
    let pair = backend
        .prepare_staggered_pair(2, [one; 3])
        .expect("valid unit-grid stencil");
    for (axis, shape) in [
        (Axis::X, [3, 1, 1]),
        (Axis::Y, [1, 3, 1]),
        (Axis::Z, [1, 1, 3]),
    ] {
        let layout = Layout::new(shape.into());
        let input = upload(backend, &[one, two, four]);
        let original = upload(backend, &[four, three, two]);
        let mut gradient = original.clone();
        let mut divergence = original.clone();
        backend
            .staggered_gradient(&pair, axis, &input, &layout, &mut gradient, &layout)
            .expect("valid staggered gradient");
        backend
            .staggered_divergence(&pair, axis, &input, &layout, &mut divergence, &layout)
            .expect("valid staggered divergence");
        // Reflected order-two G=[[-1,1,0],[0,-1,1],[0,0,0]], D=-transpose(G).
        assert_values(backend, &original, &[four, three, two]);
        assert_values(backend, &gradient, &[one, two, T::zero()]);
        assert_values(backend, &divergence, &[one, one, T::zero() - two]);
    }
}
