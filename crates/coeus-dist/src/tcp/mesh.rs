#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use moirai::Moirai;
use moirai_async::{AsyncReadExt, AsyncWriteExt, TcpListener, TcpStream};
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

/// Fully-connected mesh of TCP streams between all ranks.
pub struct TcpMesh {
    rank: usize,
    size: usize,
    // Field order is lifecycle order: sockets close before their reactor runtime.
    streams: Vec<Option<Mutex<TcpStream>>>,
    runtime: Moirai,
}

impl TcpMesh {
    fn runtime() -> Moirai {
        // Mesh I/O is serialized per peer, so one scheduler and reactor worker
        // provide all execution capacity this synchronous facade can consume.
        Moirai::builder()
            .worker_threads(1)
            .async_threads(1)
            .build()
            .expect("failed to initialize dedicated TCP mesh runtime")
    }

    #[inline]
    fn debug_timeout() -> Option<Duration> {
        cfg!(debug_assertions).then_some(Duration::from_secs(45))
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

    /// Create a new TCP mesh connecting all ranks.
    pub fn new(rank: usize, size: usize, addresses: &[SocketAddr]) -> Self {
        Self::assert_configuration(rank, size, addresses);
        let runtime = Self::runtime();
        let local_addr = addresses[rank].to_string();
        let listener = runtime.block_on(async {
            TcpListener::bind(&local_addr)
                .await
                .unwrap_or_else(|error| {
                    panic!("rank {rank} failed to bind to {local_addr}: {error}")
                })
        });

        Self::from_listener(rank, size, addresses, listener, runtime)
    }

    /// Create an in-process cluster backed by real loopback TCP sockets.
    ///
    /// Each listener remains bound from allocation through peer connection, so
    /// concurrent callers cannot claim a selected port between discovery and
    /// mesh construction.
    pub fn create_loopback_cluster(size: NonZeroUsize) -> Vec<Self> {
        let size = size.get();

        let mut endpoints = Vec::with_capacity(size);
        let mut addresses = Vec::with_capacity(size);
        for _ in 0..size {
            let runtime = Self::runtime();
            let listener = runtime.block_on(async {
                TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("failed to bind loopback TCP listener")
            });
            addresses.push(
                listener
                    .local_addr()
                    .expect("loopback TCP listener must expose its address"),
            );
            endpoints.push((listener, runtime));
        }

        thread::scope(|scope| {
            let mut workers = Vec::with_capacity(size);
            for (rank, (listener, runtime)) in endpoints.into_iter().enumerate() {
                let addresses = &addresses;
                workers.push(
                    scope.spawn(move || {
                        Self::from_listener(rank, size, addresses, listener, runtime)
                    }),
                );
            }
            workers
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .unwrap_or_else(|_| panic!("loopback TCP mesh rank panicked"))
                })
                .collect()
        })
    }

    fn from_listener(
        rank: usize,
        size: usize,
        addresses: &[SocketAddr],
        listener: TcpListener,
        runtime: Moirai,
    ) -> Self {
        Self::assert_configuration(rank, size, addresses);
        let mut streams = (0..size).map(|_| None).collect::<Vec<_>>();

        runtime.block_on(async {
            // Connect to higher ranks
            for other in (rank + 1)..size {
                let other_addr = addresses[other].to_string();
                let mut delay = Duration::from_millis(5);
                let connect_future = async {
                    loop {
                        match TcpStream::connect(&other_addr).await {
                            Ok(s) => {
                                s.set_nodelay(true).unwrap();
                                let rank_bytes = (rank as u64).to_le_bytes();
                                let mut s_mut = s;
                                if s_mut.write_all(&rank_bytes).await.is_ok() {
                                    break s_mut;
                                }
                            }
                            Err(_) => {
                                moirai_async::sleep(delay).await;
                                if delay < Duration::from_millis(500) {
                                    delay *= 2;
                                }
                            }
                        }
                    }
                };
                let stream = if let Some(timeout) = Self::debug_timeout() {
                    moirai_async::timeout(timeout, connect_future)
                        .await
                        .unwrap_or_else(|_| {
                            panic!(
                                "rank {rank} timed out connecting to peer {other} at {other_addr}"
                            )
                        })
                } else {
                    connect_future.await
                };
                assert!(
                    streams[other].is_none(),
                    "outgoing stream slot already populated for peer {other}"
                );
                streams[other] = Some(Mutex::new(stream));
            }

            // Accept connections from lower ranks
            for _ in 0..rank {
                let (s, _) = if let Some(timeout) = Self::debug_timeout() {
                    moirai_async::timeout(timeout, listener.accept())
                        .await
                        .unwrap_or_else(|_| {
                            panic!("rank {rank} timed out accepting lower-rank peer connection")
                        })
                        .expect("failed to accept connection")
                } else {
                    listener
                        .accept()
                        .await
                        .expect("failed to accept connection")
                };
                s.set_nodelay(true).unwrap();
                let mut rank_bytes = [0u8; 8];
                let mut s_mut = s;
                if let Some(timeout) = Self::debug_timeout() {
                    moirai_async::timeout(timeout, s_mut.read_exact(&mut rank_bytes))
                        .await
                        .unwrap_or_else(|_| {
                            panic!("rank {rank} timed out reading incoming peer rank during accept")
                        })
                        .expect("failed to read rank from incoming connection");
                } else {
                    s_mut
                        .read_exact(&mut rank_bytes)
                        .await
                        .expect("failed to read rank from incoming connection");
                }
                let incoming_rank = u64::from_le_bytes(rank_bytes) as usize;
                assert!(
                    incoming_rank < rank,
                    "incoming rank must be less than current rank"
                );
                assert!(
                    streams[incoming_rank].is_none(),
                    "incoming stream slot already populated for peer {incoming_rank}"
                );
                streams[incoming_rank] = Some(Mutex::new(s_mut));
            }
        });

        Self {
            rank,
            size,
            streams,
            runtime,
        }
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
    #[inline]
    pub fn send(&self, target: usize, bytes: &[u8]) {
        let stream_mutex = self.stream_for_peer(target, "send");
        let mut stream = stream_mutex.lock().unwrap();
        self.runtime.block_on(async {
            if let Some(timeout) = Self::debug_timeout() {
                moirai_async::timeout(timeout, stream.write_all(bytes))
                    .await
                    .unwrap_or_else(|_| panic!("send to peer {target} timed out"))
                    .expect("failed to send bytes over TCP");
            } else {
                stream
                    .write_all(bytes)
                    .await
                    .expect("failed to send bytes over TCP");
            }
        });
    }

    /// Receive raw bytes from a source rank.
    #[inline]
    pub fn recv(&self, source: usize, bytes: &mut [u8]) {
        let stream_mutex = self.stream_for_peer(source, "recv");
        let mut stream = stream_mutex.lock().unwrap();
        self.runtime.block_on(async {
            if let Some(timeout) = Self::debug_timeout() {
                moirai_async::timeout(timeout, stream.read_exact(bytes))
                    .await
                    .unwrap_or_else(|_| panic!("recv from peer {source} timed out"))
                    .expect("failed to receive bytes over TCP");
            } else {
                stream
                    .read_exact(bytes)
                    .await
                    .expect("failed to receive bytes over TCP");
            }
        });
    }
}
