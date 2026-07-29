use crate::module::ModuleError;
use std::error::Error;

pub(super) fn cell_input<E>(
    shape: &[usize],
    input_size: usize,
    module: &'static str,
) -> Result<usize, ModuleError<E>>
where
    E: Error + 'static,
{
    let [batch, features] = shape else {
        return Err(ModuleError::InvalidRank {
            module,
            expected: "2",
            actual: shape.len(),
        });
    };
    if *features != input_size {
        return Err(ModuleError::ShapeMismatch {
            module,
            parameter: "input",
            expected: vec![*batch, input_size],
            actual: shape.to_vec(),
        });
    }
    Ok(*batch)
}

pub(super) fn state<E>(
    shape: &[usize],
    batch: usize,
    hidden_size: usize,
    module: &'static str,
    parameter: &'static str,
) -> Result<(), ModuleError<E>>
where
    E: Error + 'static,
{
    let expected = [batch, hidden_size];
    if shape != expected {
        return Err(ModuleError::ShapeMismatch {
            module,
            parameter,
            expected: expected.to_vec(),
            actual: shape.to_vec(),
        });
    }
    Ok(())
}

pub(super) fn sequence_input<E>(
    shape: &[usize],
    input_size: usize,
    module: &'static str,
) -> Result<(usize, usize), ModuleError<E>>
where
    E: Error + 'static,
{
    let [batch, sequence, features] = shape else {
        return Err(ModuleError::InvalidRank {
            module,
            expected: "3",
            actual: shape.len(),
        });
    };
    if *features != input_size {
        return Err(ModuleError::ShapeMismatch {
            module,
            parameter: "input",
            expected: vec![*batch, *sequence, input_size],
            actual: shape.to_vec(),
        });
    }
    if *sequence == 0 {
        return Err(ModuleError::InsufficientElements {
            module,
            minimum: 1,
            actual: 0,
        });
    }
    Ok((*batch, *sequence))
}

pub(super) fn sequence_layout<E>(
    shape: &[usize],
    module: &'static str,
) -> Result<(usize, usize), ModuleError<E>>
where
    E: Error + 'static,
{
    let [batch, sequence, _features] = shape else {
        return Err(ModuleError::InvalidRank {
            module,
            expected: "3",
            actual: shape.len(),
        });
    };
    if *sequence == 0 {
        return Err(ModuleError::InsufficientElements {
            module,
            minimum: 1,
            actual: 0,
        });
    }
    Ok((*batch, *sequence))
}

pub(super) fn child_sequence_output<E>(
    shape: &[usize],
    batch: usize,
    sequence: usize,
    module: &'static str,
    parameter: &'static str,
) -> Result<(), ModuleError<E>>
where
    E: Error + 'static,
{
    let [actual_batch, actual_sequence, hidden] = shape else {
        return Err(ModuleError::InvalidRank {
            module,
            expected: "3",
            actual: shape.len(),
        });
    };
    if *actual_batch != batch || *actual_sequence != sequence {
        return Err(ModuleError::ShapeMismatch {
            module,
            parameter,
            expected: vec![batch, sequence, *hidden],
            actual: shape.to_vec(),
        });
    }
    Ok(())
}

pub(super) fn matching_child_outputs<E>(
    forward: &[usize],
    backward: &[usize],
) -> Result<(), ModuleError<E>>
where
    E: Error + 'static,
{
    if forward != backward {
        return Err(ModuleError::ShapeMismatch {
            module: "Bidirectional",
            parameter: "backward output",
            expected: forward.to_vec(),
            actual: backward.to_vec(),
        });
    }
    Ok(())
}
