use core::marker::PhantomData;

use aequitas::{
    unit::{AffineUnit, Unit},
    Quantity,
};
use coeus_core::Float;
use coeus_ops::ElementwiseOps;
use coeus_tensor::Tensor;
use eunomia::UnitScalar;

use super::{graph::require_axis, Error, Gradient, QuantityTensor};

/// A homogeneous coordinate system for a physical quantity.
pub struct Coordinates<D> {
    scale: f64,
    offset: Option<f64>,
    dimension: PhantomData<D>,
}

impl<D> Coordinates<D> {
    /// Use a linear unit associated with this dimension.
    pub fn linear<U: Unit<D>>() -> Self {
        Self {
            scale: U::SCALE,
            offset: None,
            dimension: PhantomData,
        }
    }

    /// Use an affine unit associated with a point quantity.
    pub fn affine<U: AffineUnit<D>>() -> Self {
        Self {
            scale: U::SCALE,
            offset: Some(U::OFFSET),
            dimension: PhantomData,
        }
    }

    /// Normalize coordinate readings to canonical SI before recording the graph.
    ///
    /// # Errors
    ///
    /// Returns an error for rank-zero readings, unrepresentable conversion
    /// coefficients or backend failure. Magnitudes follow the numerical
    /// backend's arithmetic semantics.
    pub fn normalize<T, B>(
        &self,
        readings: Tensor<T, B>,
    ) -> Result<QuantityTensor<T, B, D>, Error<B::Error>>
    where
        T: Float + UnitScalar,
        B: ElementwiseOps<T> + Default,
    {
        require_axis(&readings)?;
        let scale = positive_scale::<T, B::Error>(T::ONE.scale_by_factor(self.scale))?;
        let backend = B::default();
        let scale = Tensor::full_on([1], scale, &backend).map_err(Error::Backend)?;
        let scaled = coeus_ops::mul(&readings, &scale, &backend).map_err(Error::Backend)?;
        let base = if let Some(offset) = self.offset {
            let offset = T::ONE.scale_by_factor(offset);
            if !Float::is_finite(offset) {
                return Err(Error::CoordinateRange);
            }
            let offset = Tensor::full_on([1], offset, &backend).map_err(Error::Backend)?;
            coeus_ops::add(&scaled, &offset, &backend).map_err(Error::Backend)?
        } else {
            scaled
        };
        Ok(Quantity::from_base(base))
    }
}

impl<T, B, Output, Input> Gradient<T, B, Output, Input>
where
    T: Float + UnitScalar,
    B: ElementwiseOps<T> + Default,
{
    /// Express the gradient as the output coordinate change per input coordinate change.
    ///
    /// Multiplying by input scale over output scale is the chain rule for the
    /// two coordinate maps. The coordinate origins have zero derivative.
    ///
    /// # Errors
    ///
    /// Returns an error for an unrepresentable scale or backend failure.
    pub fn in_coordinates(
        &self,
        output: &Coordinates<Output>,
        input: &Coordinates<Input>,
    ) -> Result<Tensor<T, B>, Error<B::Error>> {
        let input_scale = positive_scale(T::ONE.scale_by_factor(input.scale))?;
        let output_scale = positive_scale(T::ONE.scale_by_factor(output.scale))?;
        let ratio = positive_scale(input_scale / output_scale)?;
        let backend = B::default();
        let ratio = Tensor::full_on([1], ratio, &backend).map_err(Error::Backend)?;
        coeus_ops::mul(&self.values, &ratio, &backend).map_err(Error::Backend)
    }
}

fn positive_scale<T: Float, E>(scale: T) -> Result<T, Error<E>> {
    if Float::is_finite(scale) && scale > T::ZERO {
        Ok(scale)
    } else {
        Err(Error::CoordinateRange)
    }
}
