use moirai_async::{AsyncReadExt, AsyncWriteExt, TcpListener, TcpStream};
use std::net::SocketAddr;
use std::sync::Mutex;
use std::time::Duration;

/// Fully-connected mesh of TCP streams between all ranks.
pub struct TcpMesh {
    rank: usize,
    size: usize,
    streams: Vec<Option<Mutex<TcpStream>>>,
}

impl TcpMesh {
    /// Create a new TCP mesh connecting all ranks.
    pub fn new(rank: usize, size: usize, addresses: &[SocketAddr]) -> Self {
        assert!(rank < size, "rank must be less than world size");
        assert_eq!(
            addresses.len(),
            size,
            "addresses list length must match world size"
        );
        let mut streams = (0..size).map(|_| None).collect::<Vec<_>>();

        moirai::global().block_on(async {
            let local_addr = addresses[rank].to_string();
            let listener = TcpListener::bind(&local_addr)
                .await
                .unwrap_or_else(|e| panic!("Rank {rank} failed to bind to {local_addr}: {e}"));

            // Connect to higher ranks
            for other in (rank + 1)..size {
                let other_addr = addresses[other].to_string();
                let mut delay = Duration::from_millis(5);
                let stream = loop {
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
                };
                assert!(
                    streams[other].is_none(),
                    "outgoing stream slot already populated for peer {other}"
                );
                streams[other] = Some(Mutex::new(stream));
            }

            // Accept connections from lower ranks
            for _ in 0..rank {
                let (s, _) = listener
                    .accept()
                    .await
                    .expect("failed to accept connection");
                s.set_nodelay(true).unwrap();
                let mut rank_bytes = [0u8; 8];
                let mut s_mut = s;
                s_mut
                    .read_exact(&mut rank_bytes)
                    .await
                    .expect("failed to read rank from incoming connection");
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
        moirai::global().block_on(async {
            stream
                .write_all(bytes)
                .await
                .expect("failed to send bytes over TCP");
        });
    }

    /// Receive raw bytes from a source rank.
    #[inline]
    pub fn recv(&self, source: usize, bytes: &mut [u8]) {
        let stream_mutex = self.stream_for_peer(source, "recv");
        let mut stream = stream_mutex.lock().unwrap();
        moirai::global().block_on(async {
            stream
                .read_exact(bytes)
                .await
                .expect("failed to receive bytes over TCP");
        });
    }
}
