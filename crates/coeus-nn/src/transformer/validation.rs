use crate::module::ModuleError;
use coeus_autograd::Var;
use coeus_core::Scalar;
use std::error::Error;

pub(super) fn rank_three<T, B>(
    module: &'static str,
    input: &Var<T, B>,
) -> Result<[usize; 3], ModuleError<B::Error>>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    B::Error: Error + 'static,
{
    let shape = input.tensor.shape();
    let [batch, sequence, features] = shape else {
        return Err(ModuleError::InvalidRank {
            module,
            expected: "3",
            actual: shape.len(),
        });
    };
    for (parameter, actual) in [
        ("batch", *batch),
        ("sequence", *sequence),
        ("feature", *features),
    ] {
        if actual == 0 {
            return Err(ModuleError::ShapeMismatch {
                module,
                parameter,
                expected: vec![1],
                actual: vec![0],
            });
        }
    }
    Ok([*batch, *sequence, *features])
}

pub(super) fn decoder_inputs<T, B>(
    module: &'static str,
    target: &Var<T, B>,
    memory: &Var<T, B>,
) -> Result<([usize; 3], [usize; 3]), ModuleError<B::Error>>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    B::Error: Error + 'static,
{
    let target_shape = rank_three(module, target)?;
    let memory_shape = rank_three(module, memory)?;
    if memory_shape[0] != target_shape[0] {
        return Err(ModuleError::ShapeMismatch {
            module,
            parameter: "memory batch",
            expected: vec![target_shape[0]],
            actual: vec![memory_shape[0]],
        });
    }
    if memory_shape[2] != target_shape[2] {
        return Err(ModuleError::ShapeMismatch {
            module,
            parameter: "memory feature",
            expected: vec![target_shape[2]],
            actual: vec![memory_shape[2]],
        });
    }
    Ok((target_shape, memory_shape))
}

pub(super) fn affine_vector<T, B>(
    module: &'static str,
    parameter: &'static str,
    value: &Var<T, B>,
    width: usize,
) -> Result<(), ModuleError<B::Error>>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    B::Error: Error + 'static,
{
    let actual = value.tensor.shape();
    if actual == [width] {
        Ok(())
    } else {
        Err(ModuleError::ShapeMismatch {
            module,
            parameter,
            expected: vec![width],
            actual: actual.to_vec(),
        })
    }
}

pub(super) fn feed_forward<T, B>(
    module: &'static str,
    d_model: usize,
    w1: &Var<T, B>,
    b1: Option<&Var<T, B>>,
    w2: &Var<T, B>,
    b2: Option<&Var<T, B>>,
) -> Result<(), ModuleError<B::Error>>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    B::Error: Error + 'static,
{
    let first = w1.tensor.shape();
    let [d_ff, input_width] = first else {
        return Err(ModuleError::ShapeMismatch {
            module,
            parameter: "feed-forward input weight",
            expected: vec![1, d_model],
            actual: first.to_vec(),
        });
    };
    if *d_ff == 0 || *input_width != d_model {
        return Err(ModuleError::ShapeMismatch {
            module,
            parameter: "feed-forward input weight",
            expected: vec![1, d_model],
            actual: first.to_vec(),
        });
    }
    if let Some(bias) = b1 {
        affine_vector(module, "feed-forward input bias", bias, *d_ff)?;
    }

    let second = w2.tensor.shape();
    if second != [d_model, *d_ff] {
        return Err(ModuleError::ShapeMismatch {
            module,
            parameter: "feed-forward output weight",
            expected: vec![d_model, *d_ff],
            actual: second.to_vec(),
        });
    }
    if let Some(bias) = b2 {
        affine_vector(module, "feed-forward output bias", bias, d_model)?;
    }
    Ok(())
}
