mod conv1d;
mod conv2d;
mod conv3d;

pub(crate) use conv1d::{conv1d, conv1d_backward};
pub(crate) use conv2d::{conv2d, conv2d_backward};
pub(crate) use conv3d::{conv3d, conv3d_backward};

use melinoe::MelinoeCell;

/// Reinterpret an exclusive `&mut [T]` borrow as a slice of brand-tagged
/// [`MelinoeCell`]s so it can be sharded across a Melinoe `brand_scope`.
///
/// Shared SSOT for the contiguous-fast-path row partitioning in `conv1d` and
/// `conv2d`.
///
/// # Safety
/// The caller must hold an exclusive borrow of `slice` for the entire `'brand`
/// scope; `partition_for_each_with` then splits it into disjoint shards.
#[inline]
pub(super) unsafe fn brand_mut_slice<'brand, T>(
    slice: &'brand mut [T],
) -> &'brand mut [MelinoeCell<'brand, T>] {
    let ptr = slice as *mut [T] as *mut [MelinoeCell<'brand, T>];
    // SAFETY: `MelinoeCell<'brand, T>` is `#[repr(transparent)]` over
    // `UnsafeCell<T>`, which is itself transparent over `T`, so `[T]` and
    // `[MelinoeCell<'brand, T>]` share layout and slice metadata.
    unsafe { &mut *ptr }
}
