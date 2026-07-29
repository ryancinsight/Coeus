use crate::module::ModuleError;
use std::error::Error;

pub(super) fn backend<E>(module: &'static str, source: E) -> ModuleError<E>
where
    E: Error + 'static,
{
    ModuleError::Backend { module, source }
}

pub(super) fn invalid_rank<E>(
    module: &'static str,
    expected: &'static str,
    actual: usize,
) -> ModuleError<E>
where
    E: Error + 'static,
{
    ModuleError::InvalidRank {
        module,
        expected,
        actual,
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
    ModuleError::ShapeMismatch {
        module,
        parameter,
        expected: expected.to_vec(),
        actual: actual.to_vec(),
    }
}

pub(super) fn channel_mismatch<E>(
    module: &'static str,
    expected: usize,
    actual: usize,
) -> ModuleError<E>
where
    E: Error + 'static,
{
    ModuleError::ChannelMismatch {
        module,
        expected,
        actual,
    }
}

pub(super) fn state_borrow<E>(module: &'static str, state: &'static str) -> ModuleError<E>
where
    E: Error + 'static,
{
    ModuleError::StateBorrow { module, state }
}

pub(super) fn insufficient_elements<E>(
    module: &'static str,
    minimum: usize,
    actual: usize,
) -> ModuleError<E>
where
    E: Error + 'static,
{
    ModuleError::InsufficientElements {
        module,
        minimum,
        actual,
    }
}
