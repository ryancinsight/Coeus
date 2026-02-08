#[macro_export]
macro_rules! binary_strided_primitive {
    ($name:ident, $op_trait:ident, $op_fn:ident) => {
        pub fn $name<T: crate::DataType>(
            lhs_data: &[T],
            lhs_shape: &[usize],
            lhs_strides: &[usize],
            lhs_offset: usize,
            rhs_data: &[T],
            _rhs_shape: &[usize],
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

#[macro_export]
macro_rules! unary_strided_primitive {
    ($name:ident, $op_fn:expr $(, $bounds:path)*) => {
        pub fn $name<T: crate::DataType>(
            input_data: &[T],
            input_shape: &[usize],
            input_strides: &[usize],
            input_offset: usize,
            result_data: &mut [T],
        ) -> crate::Result<()>
        where
            T: Copy + Default $(+ $bounds)*,
        {
            let out_len = result_data.len();

            for i in 0..out_len {
                let idx = input_offset + storage::iter::compute_strided_index_fast(i, input_shape, input_strides);
                result_data[i] = $op_fn(input_data[idx]);
            }

            Ok(())
        }
    };
}

#[macro_export]
macro_rules! unary_csr_primitive {
    ($name:ident, $op_fn:expr $(, $bounds:path)*) => {
        pub fn $name<T: crate::DataType>(
            input_data: &[T],
            input_indices: &[usize],
            input_indptr: &[usize],
            input_shape: &[usize],
        ) -> crate::Result<(Vec<T>, Vec<usize>, Vec<usize>)>
        where
            T: Copy + Default $(+ $bounds)*,
        {
            let mut result_data = vec![T::default(); input_data.len()];
            for i in 0..input_data.len() {
                result_data[i] = $op_fn(input_data[i]);
            }
            Ok((result_data, input_indices.to_vec(), input_indptr.to_vec()))
        }
    };
}
