//! Tensor-constructor differential tests.

#[path = "constructors/arange_leto_diff.rs"]
mod arange_leto_diff;
#[path = "constructors/from_fn_leto_diff.rs"]
mod from_fn_leto_diff;
#[path = "constructors/identity_leto_diff.rs"]
mod identity_leto_diff;
#[path = "constructors/linspace_leto_diff.rs"]
mod linspace_leto_diff;

use coeus_core::{BackendError, SequentialBackend};
use coeus_tensor::Tensor;

#[test]
fn constructors_report_cpu_allocation_overflow() {
    let backend = SequentialBackend::new();
    let results = [
        Tensor::<u16, SequentialBackend>::alloc_on([usize::MAX], &backend),
        Tensor::<u16, SequentialBackend>::zeros_on([usize::MAX], &backend),
        Tensor::<u16, SequentialBackend>::ones_on([usize::MAX], &backend),
        Tensor::<u16, SequentialBackend>::full_on([usize::MAX], 7, &backend),
    ];

    for result in results {
        let Err(error) = result else {
            panic!("an overflowing CPU allocation must fail");
        };
        assert_eq!(
            error,
            BackendError::Overflow {
                operation: "cpu allocation",
                reason: "element-count byte-size overflow",
            }
        );
    }
}
