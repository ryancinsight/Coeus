//! Shared concurrency and loopback-mesh helpers for distributed contracts.

use coeus_dist::TcpMesh;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::thread;

pub(super) fn assert_any_thread_panicked(handles: Vec<thread::JoinHandle<bool>>, message: &str) {
    let panicked = handles
        .into_iter()
        .map(|h| h.join().unwrap_or(true))
        .collect::<Vec<_>>();
    assert!(panicked.iter().any(|&p| p), "{}", message);
}

pub(super) fn spawn_maybe_panicking<F>(f: F) -> thread::JoinHandle<bool>
where
    F: FnOnce() + Send + 'static,
{
    thread::spawn(move || panic::catch_unwind(AssertUnwindSafe(f)).is_err())
}

pub(super) fn loopback_meshes(world_size: usize) -> Vec<TcpMesh> {
    let world_size =
        NonZeroUsize::new(world_size).expect("TCP test cluster requires a non-zero world size");
    TcpMesh::create_loopback_cluster(world_size)
}

pub(super) fn single_rank_tcp_mesh() -> TcpMesh {
    loopback_meshes(1)
        .into_iter()
        .next()
        .expect("one-rank loopback cluster must contain its rank")
}
