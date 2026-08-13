//! Shared concurrency and loopback-mesh helpers for distributed contracts.

use coeus_dist::TcpMesh;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard};
use std::thread;

/// Global lock that serializes all multi-rank TCP tests.
///
/// Parallel TCP test execution on Windows causes RST races: when one test's
/// `TcpMesh` is dropped, Windows sends RST segments that disrupt any other
/// test that is simultaneously exchanging data on a different loopback mesh.
/// Holding this lock for the duration of each multi-rank test eliminates
/// the races without requiring `--test-threads=1` (which would serialize
/// all tests, not just the TCP ones).
static TCP_CLUSTER_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the TCP cluster lock.  Call at the top of any test that creates a
/// multi-rank loopback `TcpMesh`.
///
/// `TcpMesh` calls `Moirai::shutdown_timeout` on drop, which waits for the
/// async reactor to stop before returning.  A brief additional sleep here
/// gives the OS time to fully release the loopback ports and clear any pending
/// RST segments before the next mesh binds.
pub(super) fn tcp_lock() -> MutexGuard<'static, ()> {
    // Recover from a poisoned mutex so that one test's panic does not prevent
    // subsequent tests from acquiring the lock.
    let guard = TCP_CLUSTER_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    // Give the global `moirai-global-reactor` time to process completion events
    // from the previous test's now-closed sockets and remove their FD entries.
    // Without this pause, Windows reuses the same FD numbers for the next
    // test's sockets, and a late-arriving completion for the old FD wakes the
    // new socket's future prematurely, corrupting the TCP exchange.
    std::thread::sleep(std::time::Duration::from_secs(1));
    guard
}

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
