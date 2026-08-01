use leto::{ArrayViewMut, LetoError, Result};
use leto_ops::{
    stateful_update as provider_update, AdaGrad, AdaGradParameters, Adam, AdamParameters, AdamW,
    AdamWParameters, RealScalar, RmsProp, RmsPropParameters, Sgd, SgdParameters,
    StatefulUpdateRule,
};

use crate::{to_leto_view, to_leto_view_mut};

use super::{ReadOperand, WriteOperand};

/// Mutable persistent state supplied to one provider-owned update.
pub enum StatefulUpdateState<'a, T> {
    /// One mutable state tensor.
    One(WriteOperand<'a, T>),
    /// Two mutable state tensors.
    Two(WriteOperand<'a, T>, WriteOperand<'a, T>),
}

/// Borrowed dynamic-rank operands for one stateful parameter update.
pub struct StatefulUpdateOperands<'a, T> {
    /// Parameter storage updated in place.
    pub parameter: WriteOperand<'a, T>,
    /// Read-only gradient storage.
    pub gradient: ReadOperand<'a, T>,
    /// Persistent optimizer state updated in place.
    pub state: StatefulUpdateState<'a, T>,
}

mod sealed {
    pub trait Sealed {}
}

/// Closed bridge from a dynamic Coeus rank to one Leto update rule.
pub trait StatefulUpdateDispatchRule<T: RealScalar>: sealed::Sealed {
    /// Validated provider parameter type.
    type Parameters: Copy;

    #[doc(hidden)]
    fn apply<const N: usize>(
        operands: StatefulUpdateOperands<'_, T>,
        parameters: Self::Parameters,
    ) -> Result<()>;
}

fn invalid_state(rule: &'static str, expected: usize) -> LetoError {
    LetoError::InvalidInput(format!(
        "{rule} requires exactly {expected} persistent state tensor(s)"
    ))
}

fn apply_one<T, Rule, const N: usize>(
    operands: StatefulUpdateOperands<'_, T>,
    parameters: <Rule as StatefulUpdateRule<T, N>>::Parameters,
) -> Result<()>
where
    T: RealScalar,
    for<'state> Rule: StatefulUpdateRule<T, N, State<'state> = ArrayViewMut<'state, T, N>>,
{
    let StatefulUpdateOperands {
        parameter,
        gradient,
        state,
    } = operands;
    let StatefulUpdateState::One(state) = state else {
        return Err(invalid_state(std::any::type_name::<Rule>(), 1));
    };
    provider_update::<T, Rule, N>(
        to_leto_view_mut::<T, N>(parameter.layout, parameter.data)?,
        to_leto_view::<T, N>(gradient.layout, gradient.data)?,
        to_leto_view_mut::<T, N>(state.layout, state.data)?,
        parameters,
    )
}

fn apply_two<T, Rule, const N: usize>(
    operands: StatefulUpdateOperands<'_, T>,
    parameters: <Rule as StatefulUpdateRule<T, N>>::Parameters,
) -> Result<()>
where
    T: RealScalar,
    for<'state> Rule: StatefulUpdateRule<
        T,
        N,
        State<'state> = (ArrayViewMut<'state, T, N>, ArrayViewMut<'state, T, N>),
    >,
{
    let StatefulUpdateOperands {
        parameter,
        gradient,
        state,
    } = operands;
    let StatefulUpdateState::Two(first, second) = state else {
        return Err(invalid_state(std::any::type_name::<Rule>(), 2));
    };
    provider_update::<T, Rule, N>(
        to_leto_view_mut::<T, N>(parameter.layout, parameter.data)?,
        to_leto_view::<T, N>(gradient.layout, gradient.data)?,
        (
            to_leto_view_mut::<T, N>(first.layout, first.data)?,
            to_leto_view_mut::<T, N>(second.layout, second.data)?,
        ),
        parameters,
    )
}

impl sealed::Sealed for Sgd {}
impl sealed::Sealed for Adam {}
impl sealed::Sealed for AdamW {}
impl sealed::Sealed for RmsProp {}
impl sealed::Sealed for AdaGrad {}

impl<T: RealScalar> StatefulUpdateDispatchRule<T> for Sgd {
    type Parameters = SgdParameters<T>;

    fn apply<const N: usize>(
        operands: StatefulUpdateOperands<'_, T>,
        parameters: Self::Parameters,
    ) -> Result<()> {
        apply_one::<T, Self, N>(operands, parameters)
    }
}

impl<T: RealScalar> StatefulUpdateDispatchRule<T> for Adam {
    type Parameters = AdamParameters<T>;

    fn apply<const N: usize>(
        operands: StatefulUpdateOperands<'_, T>,
        parameters: Self::Parameters,
    ) -> Result<()> {
        apply_two::<T, Self, N>(operands, parameters)
    }
}

impl<T: RealScalar> StatefulUpdateDispatchRule<T> for AdamW {
    type Parameters = AdamWParameters<T>;

    fn apply<const N: usize>(
        operands: StatefulUpdateOperands<'_, T>,
        parameters: Self::Parameters,
    ) -> Result<()> {
        apply_two::<T, Self, N>(operands, parameters)
    }
}

impl<T: RealScalar> StatefulUpdateDispatchRule<T> for RmsProp {
    type Parameters = RmsPropParameters<T>;

    fn apply<const N: usize>(
        operands: StatefulUpdateOperands<'_, T>,
        parameters: Self::Parameters,
    ) -> Result<()> {
        apply_one::<T, Self, N>(operands, parameters)
    }
}

impl<T: RealScalar> StatefulUpdateDispatchRule<T> for AdaGrad {
    type Parameters = AdaGradParameters<T>;

    fn apply<const N: usize>(
        operands: StatefulUpdateOperands<'_, T>,
        parameters: Self::Parameters,
    ) -> Result<()> {
        apply_one::<T, Self, N>(operands, parameters)
    }
}

/// Dispatch one provider-owned update over borrowed storage.
///
/// Runtime rank selection occurs once here. The selected Leto rule and rank
/// monomorphize the complete scalar-preserving update loop.
///
/// # Errors
///
/// Returns a typed provider error for unsupported rank, invalid layout or
/// storage, state arity, or rule parameters.
pub fn stateful_update<T, Rule>(
    operands: StatefulUpdateOperands<'_, T>,
    parameters: Rule::Parameters,
) -> Result<()>
where
    T: RealScalar,
    Rule: StatefulUpdateDispatchRule<T>,
{
    let rank = operands.parameter.layout.ndim();
    match rank {
        0 => Rule::apply::<0>(operands, parameters),
        1 => Rule::apply::<1>(operands, parameters),
        2 => Rule::apply::<2>(operands, parameters),
        3 => Rule::apply::<3>(operands, parameters),
        4 => Rule::apply::<4>(operands, parameters),
        5 => Rule::apply::<5>(operands, parameters),
        6 => Rule::apply::<6>(operands, parameters),
        7 => Rule::apply::<7>(operands, parameters),
        8 => Rule::apply::<8>(operands, parameters),
        _ => Err(LetoError::InvalidInput(format!(
            "stateful update does not support rank {rank}; maximum rank is {MAX_STATEFUL_UPDATE_RANK}"
        ))),
    }
}

/// Maximum runtime rank resolved to a monomorphized Leto update.
pub const MAX_STATEFUL_UPDATE_RANK: usize = 8;
