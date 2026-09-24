//! Peer messaging and lifecycle for [`TcpMesh`]: bounded `send`/`recv`
//! with link poisoning, the graceful `shutdown` teardown, and the synchronous
//! last-resort `Drop`.

use super::super::error::TcpMeshError;
use super::{PeerLink, TcpMesh};
use moirai_async::{AsyncReadExt, AsyncWriteExt};

impl TcpMesh {
    /// Send raw bytes to a target rank.
    ///
    /// # Errors
    ///
    /// Returns [`TcpMeshError::Send`] when the write fails and
    /// [`TcpMeshError::SendTimedOut`] when it does not complete within the
    /// [`MeshDeadlines`](super::super::deadlines::MeshDeadlines)::io bound. Either poisons the link, so every later
    /// operation on it returns [`TcpMeshError::LinkPoisoned`].
    ///
    /// # Panics
    ///
    /// Panics if `target` is the local rank or not below the cluster size.
    #[inline]
    pub fn send(&self, target: usize, bytes: &[u8]) -> Result<(), TcpMeshError> {
        let link = self.link_for_peer(target, "send");
        let mut slot = link.lock();
        let Some(stream) = slot.as_mut() else {
            return Err(self.poisoned(target, link));
        };
        let outcome = self.runtime.block_on(moirai_async::timeout(
            self.io_deadline,
            stream.write_all(bytes),
        ));
        let failure = match outcome {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(error)) => TcpMeshError::Send {
                rank: self.rank,
                peer: target,
                address: link.address,
                source: error,
            },
            Err(_) => TcpMeshError::SendTimedOut {
                rank: self.rank,
                peer: target,
                address: link.address,
                deadline: self.io_deadline,
            },
        };
        // Poison the link and close its socket.
        *slot = None;
        Err(failure)
    }

    /// Receive raw bytes from a source rank, filling `bytes` exactly.
    ///
    /// # Errors
    ///
    /// Returns [`TcpMeshError::Recv`] when the read fails or the peer closes
    /// early, and [`TcpMeshError::RecvTimedOut`] when it does not complete
    /// within the [`MeshDeadlines`](super::super::deadlines::MeshDeadlines)::io bound. Either poisons the link, so
    /// every later operation on it returns [`TcpMeshError::LinkPoisoned`].
    ///
    /// # Panics
    ///
    /// Panics if `source` is the local rank or not below the cluster size.
    #[inline]
    pub fn recv(&self, source: usize, bytes: &mut [u8]) -> Result<(), TcpMeshError> {
        let link = self.link_for_peer(source, "recv");
        let mut slot = link.lock();
        let Some(stream) = slot.as_mut() else {
            return Err(self.poisoned(source, link));
        };
        let outcome = self.runtime.block_on(moirai_async::timeout(
            self.io_deadline,
            stream.read_exact(bytes),
        ));
        let failure = match outcome {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(error)) => TcpMeshError::Recv {
                rank: self.rank,
                peer: source,
                address: link.address,
                source: error,
            },
            Err(_) => TcpMeshError::RecvTimedOut {
                rank: self.rank,
                peer: source,
                address: link.address,
                deadline: self.io_deadline,
            },
        };
        // Poison the link and close its socket.
        *slot = None;
        Err(failure)
    }

    /// The error for an operation on `link` after it was poisoned.
    fn poisoned(&self, peer: usize, link: &PeerLink) -> TcpMeshError {
        TcpMeshError::LinkPoisoned {
            rank: self.rank,
            peer,
            address: link.address,
        }
    }

    /// Poison every link, closing its socket.
    ///
    /// A collective that fails on one link leaves its other streams at
    /// unknown frame positions, and peers waiting on this rank would
    /// otherwise wait out their I/O deadline. Closing every socket ends those
    /// waits at once with end of stream or a reset, and every later
    /// operation on this mesh returns [`TcpMeshError::LinkPoisoned`].
    pub(crate) fn poison_all_links(&self) {
        for link in self.links.iter().flatten() {
            *link.lock() = None;
        }
    }

    /// Gracefully close every peer stream and stop the mesh's dedicated
    /// runtime.
    ///
    /// Half-closes (`FIN`) each established connection so an in-flight peer
    /// receive completes rather than aborting: dropping a `TcpStream` with
    /// unread kernel-buffered data sends `RST` instead of `FIN` on Windows,
    /// which resets the peer's in-flight receive and fails sequential TCP
    /// collective tests with connection resets and timeouts. Then stops the
    /// runtime so its worker thread does not idle across rounds.
    ///
    /// Every owner -- production code and tests alike -- calls this before
    /// the mesh goes out of scope. `Drop` is a synchronous last resort (see
    /// its impl below) that never blocks or awaits, so the graceful teardown
    /// lives here instead.
    pub fn shutdown(&mut self) {
        // Lock each stream in this synchronous scope (matching `send`/`recv`)
        // rather than inside the `async` block below, so the `MutexGuard`
        // never crosses an await point.
        for link in self.links.iter().flatten() {
            let mut slot = link.lock();
            // A poisoned link's socket is already closed.
            let Some(stream) = slot.as_mut() else {
                continue;
            };
            self.runtime.block_on(async {
                // A peer that already half-closed its own side returns an
                // error here; that is the expected steady state, not a fault.
                let _ = stream.shutdown().await;
            });
        }
        self.runtime.shutdown();
        self.shutdown_complete = true;
    }
}

impl Drop for TcpMesh {
    fn drop(&mut self) {
        // Synchronous last resort: performs no I/O and never blocks or
        // awaits (a destructor must not block or await). `shutdown()` is the
        // graceful teardown path and every owner calls it first, so reaching
        // this branch means a caller skipped it -- traced, not silently
        // degraded, so the abrupt close is visible rather than a silent
        // downgrade in behavior.
        if !self.shutdown_complete {
            tracing::warn!(
                rank = self.rank,
                "TcpMesh dropped without calling shutdown() first; its streams \
                 and runtime close abruptly instead of gracefully"
            );
        }
    }
}
