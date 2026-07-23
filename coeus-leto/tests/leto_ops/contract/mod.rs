//! Cross-repo contracts for the dynamic-rank Leto adapter.

use coeus_core::{BinaryOp, CpuStorage, CpuUnaryOp, Layout, ReductionOp, Shape, Strides};
use coeus_leto::{
    argmax_into, argmin_into, batched_matmul_accumulate_into, batched_matmul_into,
    broadcast_layout, broadcast_shape, concat_values, contiguous_values, cumsum_into,
    elementwise_add_into, elementwise_binary_into, elementwise_unary_into, from_shape_fn_values,
    matmul_accumulate_into, matmul_into, normal_values, pad_values, permute_layout, reduce_into,
    reshape_layout, split_values, stack_values, suffix_sum_into, to_leto_view, uniform_values,
};
use leto::Storage;

#[path = "accumulation.rs"]
mod accumulation;
#[path = "arithmetic.rs"]
mod arithmetic;
#[path = "layout.rs"]
mod layout;
#[path = "matmul.rs"]
mod matmul;
#[path = "reductions.rs"]
mod reductions;
mod support;
