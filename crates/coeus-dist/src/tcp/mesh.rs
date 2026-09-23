use super::deadlines::{self, MeshDeadlines};
use super::error::TcpMeshError;
use moirai::Moirai;
use moirai_async::{AsyncReadExt, AsyncWriteExt, TcpListener, TcpStream};
use std::io;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::panic;
use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

/// An established stream to one peer and the address it was reached at.
struct PeerLink {
    /// `None` once the link is poisoned: the first failed or timed-out send
    /// or receive may stop mid-frame, so later bytes on the stream no longer
    /// align with frames. Dropping the stream at that point closes the socket
    /// at once, so the peer's pending read ends with end of stream or a reset
    /// instead of waiting out its own I/O deadline.
    stream: Mutex<Option<TcpStream>>,
    address: SocketAddr,
}

impl PeerLink {
    fn new(stream: TcpStream, address: SocketAddr) -> Self {
        Self {
            stream: Mutex::new(Some(stream)),
            address,
        }
    }

    fn lock(&self) -> MutexGuard<'_, Option<TcpStream>> {
        self.stream.lock().expect(
            "invariant: no prior holder of this peer's stream lock panicked while holding it",
        )
    }
}

/// A per-stream setup step whose failure names the stream's peer.
#[derive(Clone, Copy)]
enum StreamStep {
    NoDelay,
    Handshake,
}

impl StreamStep {
    /// The error for this step failing on the stream to `address`, whose rank
    /// is `peer` when known.
    fn failure(
        self,
        rank: usize,
        peer: Option<usize>,
        address: SocketAddr,
        source: io::Error,
    ) -> TcpMeshError {
        match self {
            Self::NoDelay => TcpMeshError::NoDelay {
                rank,
                peer,
                address,
                source,
            },
            Self::Handshake => TcpMeshError::Handshake {
                rank,
                peer,
                address,
                source,
            },
        }
    }
}

/// Dials one peer within the given bound.
type Dial = fn(&SocketAddr, Duration) -> io::Result<std::net::TcpStream>;

/// The production dialler; tests substitute one that returns a prepared
/// stream.
const DIAL: Dial = std::net::TcpStream::connect_timeout;

/// Peer links of one rank, indexed by peer rank; the local rank's slot is empty.
type PeerLinks = Vec<Option<PeerLink>>;

/// Fully-connected mesh of TCP streams between all ranks.
///
/// Rank `r` dials every rank above it and accepts every rank below it; each
/// dialled stream opens with the dialler's rank as a little-endian `u64`.
pub struct TcpMesh {
    rank: usize,
    size: usize,
    io_deadline: Duration,
    // Field order is lifecycle order: sockets close before their reactor runtime.
    links: PeerLinks,
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
    /// when the [`MeshDeadlines::setup`] bound elapses.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero, `rank >= size`, or `addresses.len() != size`.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_dist::{MeshDeadlines, TcpMesh};
    ///
    /// let address = "127.0.0.1:0".parse().unwrap();
    /// let mut mesh = TcpMesh::new(0, 1, &[address], MeshDeadlines::DEFAULT)?;
    /// assert_eq!(mesh.size(), 1);
    /// mesh.shutdown();
    /// # Ok::<(), coeus_dist::TcpMeshError>(())
    /// ```
    pub fn new(
        rank: usize,
        size: usize,
        addresses: &[SocketAddr],
        deadlines: MeshDeadlines,
    ) -> Result<Self, TcpMeshError> {
        Self::assert_configuration(rank, size, addresses);
        let runtime = Self::runtime(rank)?;
        let listener = Self::bind(&runtime, rank, addresses[rank])?;
        Self::from_listener(rank, size, addresses, &listener, runtime, deadlines, DIAL)
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
    /// use coeus_dist::{MeshDeadlines, TcpMesh};
    /// use std::num::NonZeroUsize;
    ///
    /// let size = NonZeroUsize::new(2).unwrap();
    /// let mut meshes = TcpMesh::create_loopback_cluster(size, MeshDeadlines::DEFAULT)?;
    /// assert_eq!(meshes[1].rank(), 1);
    /// meshes.iter_mut().for_each(TcpMesh::shutdown);
    /// # Ok::<(), coeus_dist::TcpMeshError>(())
    /// ```
    pub fn create_loopback_cluster(
        size: NonZeroUsize,
        deadlines: MeshDeadlines,
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
                        Self::from_listener(
                            rank, size, addresses, &listener, runtime, deadlines, DIAL,
                        )
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
        deadlines: MeshDeadlines,
        dial: Dial,
    ) -> Result<Self, TcpMeshError> {
        Self::assert_configuration(rank, size, addresses);
        let expiry = deadlines.setup_expiry_from_now();
        let local = listener
            .local_addr()
            .map_err(|source| TcpMeshError::ListenerAddress { rank, source })?;
        let mut links = (0..size).map(|_| None).collect::<PeerLinks>();
        runtime.block_on(async {
            Self::dial_higher_ranks(rank, addresses, &mut links, expiry, dial).await?;
            Self::accept_lower_ranks(rank, listener, local, &mut links, expiry).await
        })?;

        Ok(Self {
            rank,
            size,
            io_deadline: deadlines.io(),
            links,
            runtime,
            shutdown_complete: false,
        })
    }

    async fn dial_higher_ranks(
        rank: usize,
        addresses: &[SocketAddr],
        links: &mut PeerLinks,
        expiry: Instant,
        dial: Dial,
    ) -> Result<(), TcpMeshError> {
        let rank_bytes = (rank as u64).to_le_bytes();
        for (peer, &address) in addresses.iter().enumerate().skip(rank + 1) {
            let mut stream = deadlines::connect(address, expiry, dial)
                .await
                .and_then(TcpStream::from_std)
                .map_err(|source| TcpMeshError::Connect {
                    rank,
                    peer,
                    address,
                    source,
                })?;
            let failure = |step: StreamStep| {
                move |source: io::Error| step.failure(rank, Some(peer), address, source)
            };
            stream
                .set_nodelay(true)
                .map_err(failure(StreamStep::NoDelay))?;
            deadlines::within(expiry, stream.write_all(&rank_bytes))
                .await
                .map_err(failure(StreamStep::Handshake))?;
            links[peer] = Some(PeerLink::new(stream, address));
        }
        Ok(())
    }

    async fn accept_lower_ranks(
        rank: usize,
        listener: &TcpListener,
        local: SocketAddr,
        links: &mut PeerLinks,
        expiry: Instant,
    ) -> Result<(), TcpMeshError> {
        for _ in 0..rank {
            let (mut stream, address) = deadlines::within(expiry, listener.accept())
                .await
                .map_err(|source| TcpMeshError::Accept {
                    rank,
                    address: local,
                    source,
                })?;
            // The accepted peer's rank arrives only in the handshake.
            let failure = |step: StreamStep| {
                move |source: io::Error| step.failure(rank, None, address, source)
            };
            stream
                .set_nodelay(true)
                .map_err(failure(StreamStep::NoDelay))?;
            let mut rank_bytes = [0u8; 8];
            deadlines::within(expiry, stream.read_exact(&mut rank_bytes))
                .await
                .map_err(failure(StreamStep::Handshake))?;
            let claimed = u64::from_le_bytes(rank_bytes);
            // Only a lower rank dials this rank, and each dials it once.
            let slot = usize::try_from(claimed)
                .ok()
                .filter(|&peer| peer < rank)
                .and_then(|peer| links.get_mut(peer))
                .filter(|slot| slot.is_none())
                .ok_or(TcpMeshError::PeerRank {
                    rank,
                    address,
                    claimed,
                })?;
            *slot = Some(PeerLink::new(stream, address));
        }
        Ok(())
    }

    #[inline]
    fn link_for_peer(&self, peer: usize, op: &'static str) -> &PeerLink {
        assert!(peer < self.size, "{op} peer out of bounds");
        assert!(peer != self.rank, "{op} peer must differ from local rank");
        self.links[peer]
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
    /// Returns [`TcpMeshError::Send`] when the write fails and
    /// [`TcpMeshError::SendTimedOut`] when it does not complete within the
    /// [`MeshDeadlines::io`] bound. Either poisons the link, so every later
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
    /// within the [`MeshDeadlines::io`] bound. Either poisons the link, so
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

#[cfg(test)]
mod tests;
