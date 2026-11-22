use core::ops::{Add, Sub};

use aequitas::{
    dimension::{DivideDimension, MultiplyDimension},
    systems::si::dimensions::Dimensionless,
    Quantity,
};
use coeus_core::Float;
use coeus_ops::BackendOps;

use super::{graph::require_recording, Error, Variable};

/// Add compatible physical quantities according to Aequitas's additive laws.
///
/// # Errors
///
/// Returns an error for disabled recording or backend failure.
pub fn add<'scope, T, B, L, R, O>(
    lhs: &Variable<'scope, T, B, L>,
    rhs: &Variable<'scope, T, B, R>,
) -> Result<Variable<'scope, T, B, O>, Error<B::Error>>
where
    T: Float,
    B: BackendOps<T> + Default,
    Quantity<T, L>: Add<Quantity<T, R>, Output = Quantity<T, O>>,
{
    require_recording()?;
    crate::add(&lhs.inner, &rhs.inner)
        .map(Variable::new)
        .map_err(Error::Backend)
}

/// Subtract compatible physical quantities according to Aequitas's laws.
///
/// # Errors
///
/// Returns an error for disabled recording or backend failure.
pub fn sub<'scope, T, B, L, R, O>(
    lhs: &Variable<'scope, T, B, L>,
    rhs: &Variable<'scope, T, B, R>,
) -> Result<Variable<'scope, T, B, O>, Error<B::Error>>
where
    T: Float,
    B: BackendOps<T> + Default,
    Quantity<T, L>: Sub<Quantity<T, R>, Output = Quantity<T, O>>,
{
    require_recording()?;
    crate::sub(&lhs.inner, &rhs.inner)
        .map(Variable::new)
        .map_err(Error::Backend)
}

/// Multiply canonical physical quantities, adding their SI exponents.
///
/// # Errors
///
/// Returns an error for disabled recording or backend failure.
pub fn mul<'scope, T, B, L, R, O>(
    lhs: &Variable<'scope, T, B, L>,
    rhs: &Variable<'scope, T, B, R>,
) -> Result<Variable<'scope, T, B, O>, Error<B::Error>>
where
    T: Float,
    B: BackendOps<T> + Default,
    L: MultiplyDimension<R, Output = O>,
{
    require_recording()?;
    crate::mul(&lhs.inner, &rhs.inner)
        .map(Variable::new)
        .map_err(Error::Backend)
}

/// Divide canonical physical quantities, subtracting their SI exponents.
///
/// # Errors
///
/// Returns an error for disabled recording or backend failure. Operand
/// magnitudes follow the numerical backend's division semantics.
pub fn div<'scope, T, B, L, R, O>(
    lhs: &Variable<'scope, T, B, L>,
    rhs: &Variable<'scope, T, B, R>,
) -> Result<Variable<'scope, T, B, O>, Error<B::Error>>
where
    T: Float,
    B: BackendOps<T> + Default,
    L: DivideDimension<R, Output = O>,
{
    require_recording()?;
    crate::div(&lhs.inner, &rhs.inner)
        .map(Variable::new)
        .map_err(Error::Backend)
}

/// Natural logarithm of a positive dimensionless ratio.
///
/// # Errors
///
/// Returns an error for disabled recording or backend failure. Operand
/// magnitudes follow the numerical backend's logarithm semantics.
pub fn log<'scope, T, B>(
    value: &Variable<'scope, T, B, Dimensionless>,
) -> Result<Variable<'scope, T, B, Dimensionless>, Error<B::Error>>
where
    T: Float,
    B: BackendOps<T> + Default,
{
    require_recording()?;
    crate::log(&value.inner)
        .map(Variable::new)
        .map_err(Error::Backend)
}

/// Sum a field whose physical quantity supports addition with itself.
///
/// # Errors
///
/// Returns an error for disabled recording or backend failure.
pub fn sum<'scope, T, B, D>(
    value: &Variable<'scope, T, B, D>,
) -> Result<Variable<'scope, T, B, D>, Error<B::Error>>
where
    T: Float + leto_ops::Scalar,
    B: BackendOps<T> + Default,
    Quantity<T, D>: Add<Output = Quantity<T, D>>,
{
    require_recording()?;
    crate::sum(&value.inner)
        .map(Variable::new)
        .map_err(Error::Backend)
}
