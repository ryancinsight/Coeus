use super::CrossProductProvider;
use crate::HephaestusProvider;
use coeus_core::Scalar;
use hephaestus_core::{ComputeDevice, CrossProductOps};

type Buffer<P, T> = <<P as HephaestusProvider>::Device as ComputeDevice>::Buffer<T>;

/// Allocate and initialize provider storage for the batched cross product of
/// `a` and `b`, each holding `triples` consecutive `(x, y, z)` triples.
///
/// The caller (a `coeus_ops::CrossOps` implementor) is responsible for
/// establishing that condition — this bridge covers exactly the layout
/// `hephaestus_core::CrossProductOps::cross_into` accepts (flat, contiguous)
/// and nothing more; see ADR 0077 for why a non-conforming layout stays on
/// the shared host-fold path instead of a permute-and-retry here.
///
/// # Errors
///
/// Returns a typed provider failure for allocation or dispatch failure.
pub fn cross_product<P, T>(
    a: &Buffer<P, T>,
    b: &Buffer<P, T>,
    triples: usize,
) -> hephaestus_core::Result<Buffer<P, T>>
where
    P: CrossProductProvider<T>,
    T: Scalar,
{
    let device = P::try_device()?;
    let output = device.alloc_uninitialized::<T>(triples * 3)?;
    let operations = P::Operations::default();
    operations.cross_into(device, a, b, &output)?;
    Ok(output)
}
