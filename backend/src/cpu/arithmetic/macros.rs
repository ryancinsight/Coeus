#[macro_export]
macro_rules! binary_strided_primitive {
    ($name:ident, $op_trait:ident, $op_fn:ident) => {
        pub fn $name<T: crate::DataType>(
            lhs_data: &[T],
            lhs_shape: &[usize],
            lhs_strides: &[usize],
            lhs_offset: usize,
            rhs_data: &[T],
            rhs_shape: &[usize],
            rhs_strides: &[usize],
            rhs_offset: usize,
            result_data: &mut [T],
        ) -> crate::Result<()>
        where
            T: core::ops::$op_trait<Output = T> + Copy + Default,
        {
            let out_len = result_data.len();
            let out_shape = lhs_shape;

            for i in 0..out_len {
                let lhs_idx = lhs_offset + storage::iter::compute_strided_index_fast(i, out_shape, lhs_strides);
                let rhs_idx = rhs_offset + storage::iter::compute_strided_index_fast(i, out_shape, rhs_strides);
                result_data[i] = core::ops::$op_trait::$op_fn(lhs_data[lhs_idx], rhs_data[rhs_idx]);
            }

            Ok(())
        }
    };
}
