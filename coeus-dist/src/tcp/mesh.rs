use std::net::{TcpListener, TcpStream, SocketAddr};
use std::io::{Read, Write};
use std::time::Duration;

/// Fully-connected mesh of TCP streams between all ranks.
pub struct TcpMesh {
    rank: usize,
    size: usize,
    streams: Vec<Option<TcpStream>>,
}

impl TcpMesh {
    /// Create a new TCP mesh connecting all ranks.
    pub fn new(rank: usize, size: usize, addresses: &[SocketAddr]) -> Self {
        assert_eq!(addresses.len(), size, "addresses list length must match world size");
        let mut streams = (0..size).map(|_| None).collect::<Vec<_>>();

        let listener = TcpListener::bind(addresses[rank])
            .unwrap_or_else(|e| panic!("Rank {rank} failed to bind to {}: {e}", addresses[rank]));

        // Connect to higher ranks
        for other in (rank + 1)..size {
            let mut delay = Duration::from_millis(5);
            let stream = loop {
                match TcpStream::connect(addresses[other]) {
                    Ok(mut s) => {
                        s.set_nodelay(true).unwrap();
                        let rank_bytes = (rank as u64).to_le_bytes();
                        if s.write_all(&rank_bytes).is_ok() {
                            break s;
                        }
                    }
                    Err(_) => {
                        std::thread::sleep(delay);
                        if delay < Duration::from_millis(500) {
                            delay *= 2;
                        }
                    }
                }
            };
            streams[other] = Some(stream);
        }

        // Accept connections from lower ranks
        for _ in 0..rank {
            let (mut s, _) = listener.accept().expect("failed to accept connection");
            s.set_nodelay(true).unwrap();
            let mut rank_bytes = [0u8; 8];
            s.read_exact(&mut rank_bytes).expect("failed to read rank from incoming connection");
            let incoming_rank = u64::from_le_bytes(rank_bytes) as usize;
            assert!(incoming_rank < rank, "incoming rank must be less than current rank");
            streams[incoming_rank] = Some(s);
        }

        Self { rank, size, streams }
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
        let stream = self.streams[target].as_ref().expect("stream not found");
        let mut stream_ref = stream;
        (&mut stream_ref).write_all(bytes).expect("failed to send bytes over TCP");
    }

    /// Receive raw bytes from a source rank.
    #[inline]
    pub fn recv(&self, source: usize, bytes: &mut [u8]) {
        let stream = self.streams[source].as_ref().expect("stream not found");
        let mut stream_ref = stream;
        (&mut stream_ref).read_exact(bytes).expect("failed to receive bytes over TCP");
    }
}
