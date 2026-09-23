use super::error::TcpMeshError;
use super::setup::{self, SetupDeadline};
use moirai::Moirai;
use moirai_async::{AsyncReadExt, AsyncWriteExt, TcpListener, TcpStream};
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::panic;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

/// Peer slots of one rank, indexed by peer rank; the local rank's slot is empty.
type PeerStreams = Vec<Option<Mutex<TcpStream>>>;

/// Fully-connected mesh of TCP streams between all ranks.
///
/// Rank `r` dials every rank above it and accepts every rank below it; each
/// dialled stream opens with the dialler's rank as a little-endian `u64`.
pub struct TcpMesh {
    rank: usize,
    size: usize,
    // Field order is lifecycle order: sockets close before their reactor runtime.
    streams: PeerStreams,
    runtime: Moirai,
    /// Set once [`Self::shutdown`] completes. `Drop` traces instead of
    /// silently degrading when this is still `false`: a mesh dropped without
    /// an explicit `shutdown()` call closes its streams and runtime through
    /// their own default `Drop` impls, which is abrupt rather than graceful.
    shutdown_complete: bool,
}

impl TcpMesh {
    fn runtime(rank: usize) -> Result<Moirai, TcpMeshError> {
        // Mesh I/O is serialized per peer, so one scheduler and reactor worker
        // provide all execution capacity this synchronous facade can consume.
        Moirai::builder()
            .worker_threads(1)
            .async_threads(1)
            .build()
            .map_err(|error| TcpMeshError::Runtime {
                rank,
                source: io::Error::other(error),
            })
    }

    fn bind(
        runtime: &Moirai,
        rank: usize,
        address: SocketAddr,
    ) -> Result<TcpListener, TcpMeshError> {
        let local = address.to_string();
        runtime
            .block_on(async { TcpListener::bind(&local).await })
            .map_err(|source| TcpMeshError::Bind {
                rank,
                address,
                source,
            })
    }

    #[inline]
    fn debug_timeout() -> Option<Duration> {
        cfg!(debug_assertions).then_some(Duration::from_secs(45))
    }

    /// Await established-stream I/O, bounded by the debug-build timeout.
    async fn peer_io<F, T>(operation: F) -> io::Result<T>
    where
        F: Future<Output = io::Result<T>>,
    {
        match Self::debug_timeout() {
            Some(timeout) => moirai_async::timeout(timeout, operation)
                .await
                .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "peer I/O timed out"))?,
            None => operation.await,
        }
    }

    #[inline]
    fn assert_configuration(rank: usize, size: usize, addresses: &[SocketAddr]) {
        assert!(size > 0, "world size must be > 0");
        assert!(rank < size, "rank must be less than world size");
        assert_eq!(
            addresses.len(),
            size,
            "addresses list length must match world size"
        );
    }

    /// Bind `addresses[rank]` and connect to every other rank.
    ///
    /// # Errors
    ///
    /// Returns the [`TcpMeshError`] of the first failing step: runtime start,
    /// bind, connection to a higher rank, acceptance of a lower rank,
    /// `TCP_NODELAY`, or the rank handshake. Connecting and accepting stop
    /// when `deadline` elapses.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero, `rank >= size`, or `addresses.len() != size`.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_dist::{SetupDeadline, TcpMesh};
    ///
    /// let address = "127.0.0.1:0".parse().unwrap();
    /// let mut mesh = TcpMesh::new(0, 1, &[address], SetupDeadline::default())?;
    /// assert_eq!(mesh.size(), 1);
    /// mesh.shutdown();
    /// # Ok::<(), coeus_dist::TcpMeshError>(())
    /// ```
    pub fn new(
        rank: usize,
        size: usize,
        addresses: &[SocketAddr],
        deadline: SetupDeadline,
    ) -> Result<Self, TcpMeshError> {
        Self::assert_configuration(rank, size, addresses);
        let runtime = Self::runtime(rank)?;
        let listener = Self::bind(&runtime, rank, addresses[rank])?;
        Self::from_listener(rank, size, addresses, &listener, runtime, deadline)
    }

    /// Create an in-process cluster backed by real loopback TCP sockets.
    ///
    /// Each listener remains bound from allocation through peer connection, so
    /// concurrent callers cannot claim a selected port between discovery and
    /// mesh construction.
    ///
    /// # Errors
    ///
    /// Returns the lowest failing rank's [`TcpMeshError`]; ranks that did
    /// connect are shut down before it is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_dist::{SetupDeadline, TcpMesh};
    /// use std::num::NonZeroUsize;
    ///
    /// let size = NonZeroUsize::new(2).unwrap();
    /// let mut meshes = TcpMesh::create_loopback_cluster(size, SetupDeadline::default())?;
    /// assert_eq!(meshes[1].rank(), 1);
    /// meshes.iter_mut().for_each(TcpMesh::shutdown);
    /// # Ok::<(), coeus_dist::TcpMeshError>(())
    /// ```
    pub fn create_loopback_cluster(
        size: NonZeroUsize,
        deadline: SetupDeadline,
    ) -> Result<Vec<Self>, TcpMeshError> {
        let size = size.get();
        let loopback = SocketAddr::from(([127, 0, 0, 1], 0));

        let mut endpoints = Vec::with_capacity(size);
        let mut addresses = Vec::with_capacity(size);
        for rank in 0..size {
            let runtime = Self::runtime(rank)?;
            let listener = Self::bind(&runtime, rank, loopback)?;
            let address = listener
                .local_addr()
                .map_err(|source| TcpMeshError::ListenerAddress { rank, source })?;
            addresses.push(address);
            endpoints.push((listener, runtime));
        }

        let outcomes = thread::scope(|scope| {
            let workers = endpoints
                .into_iter()
                .enumerate()
                .map(|(rank, (listener, runtime))| {
                    let addresses = &addresses;
                    scope.spawn(move || {
                        Self::from_listener(rank, size, addresses, &listener, runtime, deadline)
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .unwrap_or_else(|payload| panic::resume_unwind(payload))
                })
                .collect::<Vec<_>>()
        });

        let mut meshes = Vec::with_capacity(size);
        let mut first_error = None;
        for outcome in outcomes {
            match outcome {
                Ok(mesh) => meshes.push(mesh),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        match first_error {
            None => Ok(meshes),
            Some(error) => {
                meshes.iter_mut().for_each(Self::shutdown);
                Err(error)
            }
        }
    }

    fn from_listener(
        rank: usize,
        size: usize,
        addresses: &[SocketAddr],
        listener: &TcpListener,
        runtime: Moirai,
        deadline: SetupDeadline,
    ) -> Result<Self, TcpMeshError> {
        Self::assert_configuration(rank, size, addresses);
        let expiry = deadline.expiry_from_now();
        let local = listener
            .local_addr()
            .map_err(|source| TcpMeshError::ListenerAddress { rank, source })?;
        let mut streams = (0..size).map(|_| None).collect::<PeerStreams>();
        runtime.block_on(async {
            Self::dial_higher_ranks(rank, addresses, &mut streams, expiry).await?;
            Self::accept_lower_ranks(rank, listener, local, &mut streams, expiry).await
        })?;

        Ok(Self {
            rank,
            size,
            streams,
            runtime,
            shutdown_complete: false,
        })
    }

    async fn dial_higher_ranks(
        rank: usize,
        addresses: &[SocketAddr],
        streams: &mut PeerStreams,
        expiry: Instant,
    ) -> Result<(), TcpMeshError> {
        let rank_bytes = (rank as u64).to_le_bytes();
        for (peer, &address) in addresses.iter().enumerate().skip(rank + 1) {
            let mut stream =
                setup::connect(address, expiry)
                    .await
                    .map_err(|source| TcpMeshError::Connect {
                        rank,
                        peer,
                        address,
                        source,
                    })?;
            stream
                .set_nodelay(true)
                .map_err(|source| TcpMeshError::NoDelay {
                    rank,
                    address,
                    source,
                })?;
            setup::within(expiry, stream.write_all(&rank_bytes))
                .await
                .map_err(|source| TcpMeshError::Handshake {
                    rank,
                    address,
                    source,
                })?;
            streams[peer] = Some(Mutex::new(stream));
        }
        Ok(())
    }

    async fn accept_lower_ranks(
        rank: usize,
        listener: &TcpListener,
        local: SocketAddr,
        streams: &mut PeerStreams,
        expiry: Instant,
    ) -> Result<(), TcpMeshError> {
        for _ in 0..rank {
            let (mut stream, address) =
                setup::within(expiry, listener.accept())
                    .await
                    .map_err(|source| TcpMeshError::Accept {
                        rank,
                        address: local,
                        source,
                    })?;
            stream
                .set_nodelay(true)
                .map_err(|source| TcpMeshError::NoDelay {
                    rank,
                    address,
                    source,
                })?;
            let mut rank_bytes = [0u8; 8];
            setup::within(expiry, stream.read_exact(&mut rank_bytes))
                .await
                .map_err(|source| TcpMeshError::Handshake {
                    rank,
                    address,
                    source,
                })?;
            let claimed = u64::from_le_bytes(rank_bytes);
            // Only a lower rank dials this rank, and each dials it once.
            let slot = usize::try_from(claimed)
                .ok()
                .filter(|&peer| peer < rank)
                .and_then(|peer| streams.get_mut(peer))
                .filter(|slot| slot.is_none())
                .ok_or(TcpMeshError::PeerRank {
                    rank,
                    address,
                    claimed,
                })?;
            *slot = Some(Mutex::new(stream));
        }
        Ok(())
    }

    #[inline]
    fn stream_for_peer(&self, peer: usize, op: &'static str) -> &Mutex<TcpStream> {
        assert!(peer < self.size, "{op} peer out of bounds");
        assert!(peer != self.rank, "{op} peer must differ from local rank");
        self.streams[peer]
            .as_ref()
            .unwrap_or_else(|| panic!("{op} stream not established for peer {peer}"))
    }

    /// Access local rank.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Access cluster size.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Send raw bytes to a target rank.
    ///
    /// # Errors
    ///
    /// Returns [`TcpMeshError::Send`] when the write fails, or, in debug
    /// builds, does not complete within 45 s.
    ///
    /// # Panics
    ///
    /// Panics if `target` is the local rank or not below the cluster size.
    #[inline]
    pub fn send(&self, target: usize, bytes: &[u8]) -> Result<(), TcpMeshError> {
        let stream_mutex = self.stream_for_peer(target, "send");
        let mut stream = stream_mutex.lock().expect(
            "invariant: no prior holder of this peer's stream lock panicked while holding it",
        );
        self.runtime
            .block_on(Self::peer_io(stream.write_all(bytes)))
            .map_err(|source| TcpMeshError::Send {
                rank: self.rank,
                peer: target,
                source,
            })
    }

    /// Receive raw bytes from a source rank, filling `bytes` exactly.
    ///
    /// # Errors
    ///
    /// Returns [`TcpMeshError::Recv`] when the read fails or the peer closes
    /// early, or, in debug builds, does not complete within 45 s.
    ///
    /// # Panics
    ///
    /// Panics if `source` is the local rank or not below the cluster size.
    #[inline]
    pub fn recv(&self, source: usize, bytes: &mut [u8]) -> Result<(), TcpMeshError> {
        let stream_mutex = self.stream_for_peer(source, "recv");
        let mut stream = stream_mutex.lock().expect(
            "invariant: no prior holder of this peer's stream lock panicked while holding it",
        );
        self.runtime
            .block_on(Self::peer_io(stream.read_exact(bytes)))
            .map(drop)
            .map_err(|error| TcpMeshError::Recv {
                rank: self.rank,
                peer: source,
                source: error,
            })
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
        for slot in &self.streams {
            let Some(stream_mutex) = slot else {
                continue;
            };
            let mut stream = stream_mutex.lock().expect(
                "invariant: no prior holder of this peer's stream lock panicked while holding it",
            );
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

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "test assertions surface failures immediately by design"
    )]

    use super::*;
    use std::sync::Arc;

    /// `shutdown()` documents that it stops the runtime, joining its worker
    /// thread, before returning. Prove it: a task spawned on the mesh's own
    /// runtime holds a marker `Arc` clone that only drops when the worker
    /// thread that ran the task unwinds its stack. If `shutdown()` returned
    /// before that join happened, the clone would still be live and the
    /// strong count would read 2, not 1.
    #[test]
    fn shutdown_joins_the_runtime_worker_thread() {
        let mut mesh = TcpMesh::create_loopback_cluster(
            NonZeroUsize::new(1).unwrap(),
            SetupDeadline::default(),
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap();

        let marker = Arc::new(());
        let marker_for_worker = Arc::clone(&marker);
        mesh.runtime.spawn_fn(move || drop(marker_for_worker));

        mesh.shutdown();

        assert_eq!(
            Arc::strong_count(&marker),
            1,
            "shutdown() must join the runtime's worker thread before returning"
        );
    }

    /// A dialler announcing a rank that is not below the accepting rank is
    /// rejected with its claim and address, not trusted as a peer slot index.
    #[test]
    fn accept_rejects_a_peer_rank_that_is_not_lower() {
        const CLAIMED: u64 = 7;
        let runtime = TcpMesh::runtime(1).unwrap();
        let listener = TcpMesh::bind(&runtime, 1, SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        let listen_address = listener.local_addr().unwrap();

        let dialler = thread::spawn(move || {
            use std::io::Write;
            let mut stream = std::net::TcpStream::connect(listen_address).unwrap();
            stream.write_all(&CLAIMED.to_le_bytes()).unwrap();
            stream.local_addr().unwrap()
        });
        // Rank 1 of 2 dials nobody and accepts exactly one lower rank.
        let addresses = [SocketAddr::from(([127, 0, 0, 1], 9)), listen_address];
        let outcome = TcpMesh::from_listener(
            1,
            2,
            &addresses,
            &listener,
            runtime,
            SetupDeadline::new(Duration::from_secs(10)),
        );
        let dialler_address = dialler.join().unwrap();

        match outcome {
            Err(TcpMeshError::PeerRank {
                rank,
                address,
                claimed,
            }) => {
                assert_eq!(rank, 1);
                assert_eq!(address, dialler_address);
                assert_eq!(claimed, CLAIMED);
            }
            Err(other) => panic!("expected PeerRank, got {other:?}"),
            Ok(_) => panic!("a peer claiming rank {CLAIMED} must be rejected"),
        }
    }
}
