//! Cross-repo contract tests: pin coeus's assumptions about the leto kernels it
//! delegates to. A failure here is a leto regression from coeus's perspective.

use coeus_core::{
    BinaryOp, CpuAddressableStorage, CpuStorage, CpuUnaryOp, Layout, ReductionOp, Shape, Strides,
};
use coeus_leto::{
    argmax_into, argmin_into, batched_matmul_accumulate_into, batched_matmul_into,
    broadcast_layout, broadcast_shape, concat_values, contiguous_values, cumsum_into,
    elementwise_add_into, elementwise_binary_into, elementwise_unary_into, from_shape_fn_values,
    matmul_accumulate_into, matmul_into, normal_values, pad_values, permute_layout, reduce_into,
    reshape_layout, split_values, stack_values, suffix_sum_into, to_leto_view, uniform_values,
};
use leto::Storage;

fn layout(shape: &[usize]) -> Layout {
    Layout::new(Shape::from(shape.to_vec()))
}

#[test]
fn add_matches_reference_rank2() {
    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![10.0f64, 20.0, 30.0, 40.0];
    let mut out = vec![0.0f64; 4];
    let la = layout(&[2, 2]);

    elementwise_add_into(&la, &a, &la, &b, &la, &mut out).unwrap();
    assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn add_broadcasts_rowvec_into_matrix() {
    // [2,1] + [1,2] -> [2,2], exercising the broadcast-aware leto kernel from
    // coeus's dynamic-rank entry point.
    let a = vec![1.0f64, 2.0]; // shape [2,1]
    let b = vec![10.0f64, 20.0]; // shape [1,2]
    let mut out = vec![0.0f64; 4];

    elementwise_add_into(
        &layout(&[2, 1]),
        &a,
        &layout(&[1, 2]),
        &b,
        &layout(&[2, 2]),
        &mut out,
    )
    .unwrap();
    // rows: [1+10, 1+20], [2+10, 2+20]
    assert_eq!(out, vec![11.0, 21.0, 12.0, 22.0]);
}

#[test]
fn binary_dispatch_covers_arithmetic_ops() {
    let la = layout(&[2, 2]);
    let a = vec![8.0f64, 9.0, 10.0, 12.0];
    let b = vec![2.0f64, 3.0, 5.0, 6.0];
    let mut out = vec![0.0f64; 4];

    elementwise_binary_into(BinaryOp::Sub, &la, &a, &la, &b, &la, &mut out).unwrap();
    assert_eq!(out, vec![6.0, 6.0, 5.0, 6.0]);

    elementwise_binary_into(BinaryOp::Mul, &la, &a, &la, &b, &la, &mut out).unwrap();
    assert_eq!(out, vec![16.0, 27.0, 50.0, 72.0]);

    elementwise_binary_into(BinaryOp::Div, &la, &a, &la, &b, &la, &mut out).unwrap();
    assert_eq!(out, vec![4.0, 3.0, 2.0, 2.0]);
}

#[test]
fn unary_dispatch_covers_scalar_mapping() {
    let input = vec![-4.0f64, -1.0, 0.0, 9.0];
    let mut out = vec![0.0f64; 4];
    let la = layout(&[2, 2]);

    elementwise_unary_into(CpuUnaryOp::Relu, &la, &input, &la, &mut out).unwrap();
    assert_eq!(out, vec![0.0, 0.0, 0.0, 9.0]);

    elementwise_unary_into(CpuUnaryOp::Abs, &la, &input, &la, &mut out).unwrap();
    assert_eq!(out, vec![4.0, 1.0, 0.0, 9.0]);

    elementwise_unary_into(CpuUnaryOp::Neg, &la, &input, &la, &mut out).unwrap();
    assert_eq!(out, vec![4.0, 1.0, -0.0, -9.0]);
}

#[test]
fn unary_dispatch_exp_log_sqrt_matches_scalar_reference() {
    let input = vec![0.0f64, 1.0, 4.0, 16.0];
    let mut out = vec![0.0f64; 4];
    let la = layout(&[2, 2]);

    elementwise_unary_into(CpuUnaryOp::Exp, &la, &input, &la, &mut out).unwrap();
    assert_eq!(out, vec![1.0, 1.0_f64.exp(), 4.0_f64.exp(), 16.0_f64.exp()]);

    elementwise_unary_into(CpuUnaryOp::Log, &la, &input, &la, &mut out).unwrap();
    assert_eq!(
        out,
        vec![f64::NEG_INFINITY, 0.0, 4.0_f64.ln(), 16.0_f64.ln()]
    );

    elementwise_unary_into(CpuUnaryOp::Sqrt, &la, &input, &la, &mut out).unwrap();
    assert_eq!(out, vec![0.0, 1.0, 2.0, 4.0]);
}

#[test]
fn unary_dispatch_special_functions_match_reference_values() {
    let input = vec![0.0f64, 0.5, 1.0, 5.0];
    let mut out = vec![0.0f64; 4];
    let la = layout(&[4]);

    elementwise_unary_into(CpuUnaryOp::Erf, &la, &input, &la, &mut out).unwrap();
    assert!(
        (out[1] - 0.520_499_877_813_046_5).abs() <= 2.0e-15,
        "erf(0.5)"
    );

    elementwise_unary_into(CpuUnaryOp::Erfc, &la, &input, &la, &mut out).unwrap();
    assert!(
        (out[3] - 1.537_459_794_428_034_7e-12).abs() <= 2.0e-25,
        "erfc(5)"
    );

    elementwise_unary_into(CpuUnaryOp::Lgamma, &la, &input, &la, &mut out).unwrap();
    assert!(out[0].is_infinite(), "lgamma(0)");
    assert!(
        (out[1] - 0.572_364_942_924_700_1).abs() <= 2.0e-15,
        "lgamma(0.5)"
    );
    assert_eq!(out[2], 0.0);
    assert!((out[3] - 24.0_f64.ln()).abs() <= 2.0e-15, "lgamma(5)");
}

#[test]
fn reduction_dispatch_covers_keepdim_axis_ops() {
    let input = vec![1.0f64, 4.0, -2.0, 5.0, 3.0, 6.0];
    let input_layout = layout(&[2, 3]);
    let output_layout = layout(&[2, 1]);
    let mut out = vec![0.0f64; 2];

    reduce_into(
        ReductionOp::Sum,
        &input_layout,
        &input,
        1,
        &output_layout,
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![3.0, 14.0]);

    reduce_into(
        ReductionOp::Mean,
        &input_layout,
        &input,
        1,
        &output_layout,
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![1.0, 14.0 / 3.0]);

    reduce_into(
        ReductionOp::Max,
        &input_layout,
        &input,
        1,
        &output_layout,
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![4.0, 6.0]);

    reduce_into(
        ReductionOp::Min,
        &input_layout,
        &input,
        1,
        &output_layout,
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![-2.0, 3.0]);
}

#[test]
fn arg_reduction_dispatch_covers_keepdim_axis_ops() {
    let input = vec![1.0f64, 4.0, -2.0, 5.0, 3.0, 6.0];
    let input_layout = layout(&[2, 3]);
    let output_layout = layout(&[2, 1]);
    let mut out = vec![0i64; 2];

    argmax_into(&input_layout, &input, 1, &output_layout, &mut out).unwrap();
    assert_eq!(out, vec![1, 2]);

    argmin_into(&input_layout, &input, 1, &output_layout, &mut out).unwrap();
    assert_eq!(out, vec![2, 1]);
}

#[test]
fn scan_dispatch_covers_forward_and_reverse_axis_ops() {
    let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_layout = layout(&[2, 3]);
    let mut out = vec![0.0f64; 6];

    cumsum_into(&input_layout, &input, 1, &input_layout, &mut out).unwrap();
    assert_eq!(out, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);

    suffix_sum_into(&input_layout, &input, 1, &input_layout, &mut out).unwrap();
    assert_eq!(out, vec![6.0, 5.0, 3.0, 15.0, 11.0, 6.0]);
}

#[test]
fn matmul_matches_reference() {
    // [[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut out = vec![0.0f64; 4];

    matmul_into(
        &layout(&[2, 3]),
        &a,
        &layout(&[3, 2]),
        &b,
        &layout(&[2, 2]),
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_handles_transposed_input_view() {
    // a stored as [3,2] but used transposed as [2,3] via explicit strides.
    let a_storage = vec![1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0]; // logical [3,2]
                                                           // transposed layout: shape [2,3], strides swapped.
    let a_t = Layout::from_shape_strides(
        Shape::from(vec![2usize, 3]),
        Strides::from_slice(&[1usize, 2]),
        0,
    );
    let b = vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut out = vec![0.0f64; 4];

    matmul_into(
        &a_t,
        &a_storage,
        &layout(&[3, 2]),
        &b,
        &layout(&[2, 2]),
        &mut out,
    )
    .unwrap();
    // transposed a is [[1,2,3],[4,5,6]] -> same product as the contiguous case.
    assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn batched_matmul_dispatch_covers_rhs_batch_broadcast() {
    let lhs = vec![
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let rhs = vec![2.0f64, 3.0, 5.0, 7.0, 11.0, 13.0];
    let mut out = vec![0.0f64; 8];
    let lhs_layout = layout(&[2, 2, 3]);
    let rhs_layout = layout(&[1, 3, 2]);
    let out_layout = layout(&[2, 2, 2]);

    batched_matmul_into(&lhs_layout, &lhs, &rhs_layout, &rhs, &out_layout, &mut out).unwrap();

    assert_eq!(
        out,
        vec![45.0, 56.0, 99.0, 125.0, 153.0, 194.0, 207.0, 263.0]
    );
}

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

#[test]
fn rank_beyond_dispatch_bound_is_rejected() {
    let a = vec![0.0f64; 128];
    let la = layout(&[2, 2, 2, 2, 2, 2, 2]); // rank 7 > MAX_DISPATCH_RANK
    let mut out = vec![0.0f64; 128];
    assert!(elementwise_add_into(&la, &a, &la, &a, &la, &mut out).is_err());
}

#[test]
fn matmul_accumulate_adds_into_existing_output() {
    // out += A*B (must accumulate onto a non-zero output, not overwrite).
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]] -> A*B = [[19,22],[43,50]].
    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![5.0f64, 6.0, 7.0, 8.0];
    let mut out = vec![1.0f64; 4]; // pre-seeded
    let l = layout(&[2, 2]);

    matmul_accumulate_into(&l, &a, &l, &b, &l, &mut out).unwrap();
    // [[19+1, 22+1], [43+1, 50+1]]
    assert_eq!(out, vec![20.0, 23.0, 44.0, 51.0]);
}

#[test]
fn batched_matmul_accumulate_adds_per_batch() {
    // out += A*B over a batch of 2. Batch 0: I*B0 = B0; batch 1: 2I*B1 = 2*B1.
    let a = vec![
        1.0f64, 0.0, 0.0, 1.0, // batch 0: identity
        2.0, 0.0, 0.0, 2.0, // batch 1: 2*identity
    ];
    let b = vec![
        5.0f64, 6.0, 7.0, 8.0, // batch 0
        1.0, 1.0, 1.0, 1.0, // batch 1
    ];
    let mut out = vec![1.0f64; 8]; // pre-seeded
    let l = layout(&[2, 2, 2]);

    batched_matmul_accumulate_into(&l, &a, &l, &b, &l, &mut out).unwrap();
    // batch 0: [5,6,7,8] + 1; batch 1: [2,2,2,2] + 1
    assert_eq!(out, vec![6.0, 7.0, 8.0, 9.0, 3.0, 3.0, 3.0, 3.0]);
}
