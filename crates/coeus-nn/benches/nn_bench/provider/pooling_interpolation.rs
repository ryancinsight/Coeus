//! pooling interpolation benchmarks.

use super::*;

#[path = "pooling_interpolation/interpolation.rs"]
mod interpolation;
#[path = "pooling_interpolation/pool.rs"]
mod pool;

pub(crate) use interpolation::*;
pub(crate) use pool::*;

#[path = "pooling_interpolation/extended.rs"]
mod extended;
pub(crate) use extended::*;
