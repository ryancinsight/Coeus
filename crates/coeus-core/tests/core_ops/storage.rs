//! Copy-on-write storage contract tests.

#[path = "storage/backend_write_tests.rs"]
mod backend_write_tests;
#[path = "storage/backend_writes.rs"]
mod backend_writes;

#[path = "storage/cow_storage_tests.rs"]
mod cow_storage_tests;
