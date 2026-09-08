use super::backend_writes::{preserves_cloned_storage, Write};
use coeus_core::{MoiraiBackend, SequentialBackend};

#[test]
fn backend_fill_preserves_cloned_storage() {
    preserves_cloned_storage(&SequentialBackend::new(), Write::Fill);
    preserves_cloned_storage(&MoiraiBackend::new(), Write::Fill);
}

#[test]
fn backend_zero_fill_preserves_cloned_storage() {
    preserves_cloned_storage(&SequentialBackend::new(), Write::FillZero);
    preserves_cloned_storage(&MoiraiBackend::new(), Write::FillZero);
}

#[test]
fn backend_upload_preserves_cloned_storage() {
    preserves_cloned_storage(&SequentialBackend::new(), Write::CopyToDevice);
    preserves_cloned_storage(&MoiraiBackend::new(), Write::CopyToDevice);
}
