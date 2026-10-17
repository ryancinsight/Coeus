use super::RandomInitProvider;
use crate::HephaestusProvider;
use coeus_core::{Layout, Scalar};
use hephaestus_core::{ComputeDevice, HephaestusError, RandomInitOps};

type Buffer<P, T> = <<P as HephaestusProvider>::Device as ComputeDevice>::Buffer<T>;

fn shape<const N: usize>(layout: &Layout) -> hephaestus_core::Result<[usize; N]> {
    layout
        .shape()
        .try_into()
        .map_err(|_| HephaestusError::InvalidConfiguration {
            message: format!(
                "random initialization expected rank {N}, got {}",
                layout.ndim()
            ),
        })
}

fn uniform_rank<P, T, const N: usize>(
    layout: &Layout,
    low: T,
    high: T,
    seed: u64,
) -> hephaestus_core::Result<Buffer<P, T>>
where
    P: RandomInitProvider<T>,
    T: Scalar,
{
    P::Operations::default().uniform_with_seed(
        P::try_device()?,
        shape::<N>(layout)?,
        low,
        high,
        seed,
    )
}

/// Allocate provider storage filled with deterministic uniform values.
///
/// # Errors
///
/// Returns a typed provider failure for unsupported rank, allocation,
/// generation, or transfer failure.
pub fn uniform<P, T>(
    layout: &Layout,
    low: T,
    high: T,
    seed: u64,
) -> hephaestus_core::Result<Buffer<P, T>>
where
    P: RandomInitProvider<T>,
    T: Scalar,
{
    match layout.ndim() {
        1 => uniform_rank::<P, T, 1>(layout, low, high, seed),
        2 => uniform_rank::<P, T, 2>(layout, low, high, seed),
        3 => uniform_rank::<P, T, 3>(layout, low, high, seed),
        4 => uniform_rank::<P, T, 4>(layout, low, high, seed),
        5 => uniform_rank::<P, T, 5>(layout, low, high, seed),
        6 => uniform_rank::<P, T, 6>(layout, low, high, seed),
        rank => Err(HephaestusError::InvalidConfiguration {
            message: format!("random initialization supports rank 1..=6, got {rank}"),
        }),
    }
}

fn normal_rank<P, T, const N: usize>(
    layout: &Layout,
    mean: T,
    std_dev: T,
    seed: u64,
) -> hephaestus_core::Result<Buffer<P, T>>
where
    P: RandomInitProvider<T>,
    T: Scalar,
{
    P::Operations::default().normal_with_seed(
        P::try_device()?,
        shape::<N>(layout)?,
        mean,
        std_dev,
        seed,
    )
}

/// Allocate provider storage filled with deterministic normal values.
///
/// # Errors
///
/// Returns a typed provider failure for unsupported rank, allocation,
/// generation, or transfer failure.
pub fn normal<P, T>(
    layout: &Layout,
    mean: T,
    std_dev: T,
    seed: u64,
) -> hephaestus_core::Result<Buffer<P, T>>
where
    P: RandomInitProvider<T>,
    T: Scalar,
{
    match layout.ndim() {
        1 => normal_rank::<P, T, 1>(layout, mean, std_dev, seed),
        2 => normal_rank::<P, T, 2>(layout, mean, std_dev, seed),
        3 => normal_rank::<P, T, 3>(layout, mean, std_dev, seed),
        4 => normal_rank::<P, T, 4>(layout, mean, std_dev, seed),
        5 => normal_rank::<P, T, 5>(layout, mean, std_dev, seed),
        6 => normal_rank::<P, T, 6>(layout, mean, std_dev, seed),
        rank => Err(HephaestusError::InvalidConfiguration {
            message: format!("random initialization supports rank 1..=6, got {rank}"),
        }),
    }
}
