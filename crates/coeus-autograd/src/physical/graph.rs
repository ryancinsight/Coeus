use core::marker::PhantomData;

use aequitas::{dimension::DivideDimension, Quantity};
use coeus_core::{ComputeBackend, Float};
use coeus_tensor::Tensor;

use super::Error;
use crate::{is_grad_enabled, Var};

type Invariant<'scope> = PhantomData<fn(&'scope ()) -> &'scope ()>;

/// A homogeneous tensor stored in the canonical SI unit for dimension `D`.
pub type QuantityTensor<T, B, D> = Quantity<Tensor<T, B>, D>;

/// Authority to create constants in a single physical differentiation graph.
pub struct Scope<'scope>(Invariant<'scope>);

/// A homogeneous physical variable tied to the current differentiation.
pub struct Variable<'scope, T: Float, B: ComputeBackend + Default, D> {
    pub(super) inner: Var<T, B>,
    scope: Invariant<'scope>,
    dimension: PhantomData<D>,
}

impl<'scope, T: Float, B: ComputeBackend + Default, D> Variable<'scope, T, B, D> {
    pub(super) fn new(inner: Var<T, B>) -> Self {
        Self {
            inner,
            scope: PhantomData,
            dimension: PhantomData,
        }
    }
}

impl<'scope> Scope<'scope> {
    /// Introduce a constant whose tensor already holds canonical SI values.
    ///
    /// # Errors
    ///
    /// Returns an error for a rank-zero tensor or backend failure.
    pub fn constant<T: Float, B: ComputeBackend + Default, D>(
        &self,
        value: QuantityTensor<T, B, D>,
    ) -> Result<Variable<'scope, T, B, D>, Error<B::Error>> {
        require_axis(value.as_base())?;
        Var::new(value.into_base(), false)
            .map(Variable::new)
            .map_err(Error::Backend)
    }
}

/// A gradient computed for one objective and input, stored in canonical SI.
pub struct Gradient<T: Float, B: ComputeBackend, Output, Input> {
    pub(super) values: Tensor<T, B>,
    dimensions: PhantomData<(Output, Input)>,
}

impl<T: Float, B: ComputeBackend, Output, Input> Gradient<T, B, Output, Input>
where
    Output: DivideDimension<Input>,
{
    /// Return the physical gradient with SI exponents equal to output minus input.
    ///
    /// Aequitas's quotient normalizes semantic tags. Absolute-temperature
    /// differentials carry interval units, independent of the point origin.
    pub fn into_base(self) -> QuantityTensor<T, B, <Output as DivideDimension<Input>>::Output> {
        Quantity::from_base(self.values)
    }
}

/// Value and input-shaped gradient of a one-element physical objective.
pub type ValueAndGradient<T, B, Output, Input> =
    (QuantityTensor<T, B, Output>, Gradient<T, B, Output, Input>);

pub(super) fn require_axis<T: Float, B: ComputeBackend>(
    tensor: &Tensor<T, B>,
) -> Result<(), Error<B::Error>> {
    if tensor.ndim() == 0 {
        Err(Error::RankZero)
    } else {
        Ok(())
    }
}

pub(super) fn require_recording<E>() -> Result<(), Error<E>> {
    if is_grad_enabled() {
        Ok(())
    } else {
        Err(Error::RecordingDisabled)
    }
}

/// Differentiate a one-element physical objective with respect to one field.
///
/// Each call creates a fresh graph and performs one reverse pass. The callback
/// receives a tracked input and may create constants through the scope.
/// An output independent of the input has a zero derivative in the same units
/// and shape as any other derivative of that output with respect to that input.
///
/// # Errors
///
/// Returns an error for a rank-zero input, disabled recording, a non-scalar
/// objective, an unavailable gradient buffer, or a backend failure.
pub fn value_and_gradient<T, B, Output, Input>(
    input: QuantityTensor<T, B, Input>,
    objective: impl for<'scope> FnOnce(
        Scope<'scope>,
        Variable<'scope, T, B, Input>,
    ) -> Result<Variable<'scope, T, B, Output>, Error<B::Error>>,
) -> Result<ValueAndGradient<T, B, Output, Input>, Error<B::Error>>
where
    T: Float,
    B: ComputeBackend + Default,
    Output: DivideDimension<Input>,
{
    require_recording()?;
    require_axis(input.as_base())?;
    let input = Var::new(input.into_base(), true).map_err(Error::Backend)?;
    let output = objective(Scope(PhantomData), Variable::new(input.clone()))?.inner;
    require_recording()?;

    let elements = output.tensor.numel();
    if elements != 1 {
        return Err(Error::NonScalarObjective { elements });
    }
    output.backward().map_err(Error::Backend)?;
    let values = input.grad().ok_or(Error::MissingGradient)?;
    Ok((
        Quantity::from_base(output.tensor),
        Gradient {
            values,
            dimensions: PhantomData,
        },
    ))
}
