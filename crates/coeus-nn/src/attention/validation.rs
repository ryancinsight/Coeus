use crate::module::{ModuleError, ModuleError::ShapeMismatch};
use coeus_autograd::Var;
use coeus_core::Scalar;
use std::error::Error;

#[derive(Clone, Copy)]
pub(super) struct AttentionDimensions {
    pub(super) batch: usize,
    pub(super) seq_q: usize,
    pub(super) seq_k: usize,
    pub(super) d_k: usize,
    pub(super) d_v: usize,
}

pub(super) fn sdp_dimensions<T, B>(
    module: &'static str,
    query: &Var<T, B>,
    key: &Var<T, B>,
    value: &Var<T, B>,
    key_padding_mask: Option<&Var<T, B>>,
    head_count: usize,
) -> Result<AttentionDimensions, ModuleError<B::Error>>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    B::Error: Error + 'static,
{
    let query_shape = query.tensor.shape();
    let key_shape = key.tensor.shape();
    let value_shape = value.tensor.shape();
    for shape in [query_shape, key_shape, value_shape] {
        if shape.len() != 3 {
            return Err(ModuleError::InvalidRank {
                module,
                expected: "3",
                actual: shape.len(),
            });
        }
    }

    let dimensions = AttentionDimensions {
        batch: query_shape[0],
        seq_q: query_shape[1],
        seq_k: key_shape[1],
        d_k: query_shape[2],
        d_v: value_shape[2],
    };
    let execution_batch =
        dimensions
            .batch
            .checked_mul(head_count)
            .ok_or_else(|| ShapeMismatch {
                module,
                parameter: "batch and head product",
                expected: vec![usize::MAX],
                actual: vec![dimensions.batch, head_count],
            })?;
    for (parameter, actual) in [
        ("query batch", dimensions.batch),
        ("query sequence", dimensions.seq_q),
        ("key sequence", dimensions.seq_k),
        ("query feature", dimensions.d_k),
        ("value feature", dimensions.d_v),
    ] {
        if actual == 0 {
            return Err(ShapeMismatch {
                module,
                parameter,
                expected: vec![1],
                actual: vec![0],
            });
        }
    }
    if key_shape[0] != dimensions.batch {
        return Err(shape_mismatch(
            module,
            "key batch",
            &[dimensions.batch],
            &[key_shape[0]],
        ));
    }
    if value_shape[0] != dimensions.batch {
        return Err(shape_mismatch(
            module,
            "value batch",
            &[dimensions.batch],
            &[value_shape[0]],
        ));
    }
    if key_shape[2] != dimensions.d_k {
        return Err(shape_mismatch(
            module,
            "key feature",
            &[dimensions.d_k],
            &[key_shape[2]],
        ));
    }
    if value_shape[1] != dimensions.seq_k {
        return Err(shape_mismatch(
            module,
            "value sequence",
            &[dimensions.seq_k],
            &[value_shape[1]],
        ));
    }

    if let Some(mask) = key_padding_mask {
        validate_mask(
            module,
            mask.tensor.shape(),
            dimensions.seq_k,
            dimensions.batch,
            execution_batch,
        )?;
    }
    Ok(dimensions)
}

pub(super) fn projection<T, B>(
    module: &'static str,
    parameter: &'static str,
    weight: &Var<T, B>,
    bias: Option<&Var<T, B>>,
    d_model: usize,
) -> Result<(), ModuleError<B::Error>>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    B::Error: Error + 'static,
{
    let weight_shape = weight.tensor.shape();
    if weight_shape != [d_model, d_model] {
        return Err(shape_mismatch(
            module,
            parameter,
            &[d_model, d_model],
            weight_shape,
        ));
    }
    if let Some(bias) = bias {
        let bias_shape = bias.tensor.shape();
        if bias_shape != [d_model] {
            return Err(shape_mismatch(module, parameter, &[d_model], bias_shape));
        }
    }
    Ok(())
}

fn validate_mask<E>(
    module: &'static str,
    shape: &[usize],
    seq_k: usize,
    input_batch: usize,
    execution_batch: usize,
) -> Result<(), ModuleError<E>>
where
    E: Error + 'static,
{
    let valid = match shape {
        [mask_seq] => *mask_seq == seq_k,
        [mask_batch, mask_seq] => {
            let batch_matches = matches!(*mask_batch, 1)
                || *mask_batch == input_batch
                || *mask_batch == execution_batch;
            *mask_batch > 0 && *mask_seq == seq_k && batch_matches
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(shape_mismatch(
            module,
            "key padding mask",
            &[input_batch, seq_k],
            shape,
        ))
    }
}

pub(super) fn shape_mismatch<E>(
    module: &'static str,
    parameter: &'static str,
    expected: &[usize],
    actual: &[usize],
) -> ModuleError<E>
where
    E: Error + 'static,
{
    ShapeMismatch {
        module,
        parameter,
        expected: expected.to_vec(),
        actual: actual.to_vec(),
    }
}
