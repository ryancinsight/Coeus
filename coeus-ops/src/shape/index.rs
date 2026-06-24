/// Convert a row-major flat index to a multi-dimensional index.
pub(super) fn flat_to_nd(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut idx = vec![0usize; ndim];
    for d in (0..ndim).rev() {
        idx[d] = flat % shape[d];
        flat /= shape[d];
    }
    idx
}
