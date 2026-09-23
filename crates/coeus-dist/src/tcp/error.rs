use std::io;
use std::net::SocketAddr;
use std::time::Duration;

/// Failure of a [`TcpMesh`](super::TcpMesh) operation.
///
/// Every variant names the local `rank`. Socket variants carry the operating
/// system's [`io::Error`] as their [`source`](std::error::Error::source);
/// their `Display` adds only the mesh context, so a chain reporter prints the
/// cause once. A setup step that exceeds the setup deadline reports a source
/// of kind [`io::ErrorKind::TimedOut`]; a send or receive that exceeds the
/// I/O deadline is [`SendTimedOut`](Self::SendTimedOut) or
/// [`RecvTimedOut`](Self::RecvTimedOut), which carry the elapsed `deadline`
/// and no source. Any send or receive failure poisons that peer's link:
/// every later operation on it returns [`LinkPoisoned`](Self::LinkPoisoned).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TcpMeshError {
    /// The dedicated reactor runtime could not start.
    #[error("rank {rank} could not start its TCP mesh runtime")]
    Runtime {
        /// Local rank.
        rank: usize,
        /// Runtime construction failure.
        #[source]
        source: io::Error,
    },
    /// The listener could not bind the local address.
    #[error("rank {rank} could not bind {address}")]
    Bind {
        /// Local rank.
        rank: usize,
        /// Local address that was requested.
        address: SocketAddr,
        /// Bind failure.
        #[source]
        source: io::Error,
    },
    /// The bound listener did not report its address.
    #[error("rank {rank} could not read its listener address")]
    ListenerAddress {
        /// Local rank.
        rank: usize,
        /// Address query failure.
        #[source]
        source: io::Error,
    },
    /// No connection to a higher-ranked peer succeeded before the setup
    /// deadline.
    #[error("rank {rank} could not connect to peer {peer} at {address}")]
    Connect {
        /// Local rank.
        rank: usize,
        /// Rank of the unreachable peer.
        peer: usize,
        /// Peer address that was dialled.
        address: SocketAddr,
        /// Error of the last connection attempt, or a timeout when the
        /// deadline elapsed during an attempt.
        #[source]
        source: io::Error,
    },
    /// Accepting a lower-ranked peer failed or exceeded the setup deadline.
    #[error("rank {rank} could not accept a lower-ranked peer on {address}")]
    Accept {
        /// Local rank.
        rank: usize,
        /// Local listener address.
        address: SocketAddr,
        /// Accept failure.
        #[source]
        source: io::Error,
    },
    /// `TCP_NODELAY` could not be enabled on an established stream.
    #[error("rank {rank} could not disable Nagle's algorithm on the stream to {address}")]
    NoDelay {
        /// Local rank.
        rank: usize,
        /// Peer rank: known for a dialled peer, `None` for an accepted
        /// stream, whose rank arrives only in the handshake.
        peer: Option<usize>,
        /// Peer address of the stream.
        address: SocketAddr,
        /// Socket option failure.
        #[source]
        source: io::Error,
    },
    /// Sending or receiving the rank handshake failed.
    #[error("rank {rank} could not exchange ranks with {address}")]
    Handshake {
        /// Local rank.
        rank: usize,
        /// Peer rank: known for a dialled peer, `None` for an accepted
        /// stream, whose rank this handshake would have announced.
        peer: Option<usize>,
        /// Peer address of the stream.
        address: SocketAddr,
        /// Handshake I/O failure.
        #[source]
        source: io::Error,
    },
    /// An accepted peer announced a rank that is not a lower rank without an
    /// established stream.
    #[error("rank {rank} received invalid peer rank {claimed} from {address}")]
    PeerRank {
        /// Local rank.
        rank: usize,
        /// Peer address of the stream.
        address: SocketAddr,
        /// Rank the peer announced.
        claimed: u64,
    },
    /// Sending bytes to an established peer failed.
    #[error("rank {rank} could not send to peer {peer} at {address}")]
    Send {
        /// Local rank.
        rank: usize,
        /// Destination rank.
        peer: usize,
        /// Peer address of the stream.
        address: SocketAddr,
        /// Write failure.
        #[source]
        source: io::Error,
    },
    /// Receiving bytes from an established peer failed.
    #[error("rank {rank} could not receive from peer {peer} at {address}")]
    Recv {
        /// Local rank.
        rank: usize,
        /// Source rank.
        peer: usize,
        /// Peer address of the stream.
        address: SocketAddr,
        /// Read failure, [`io::ErrorKind::UnexpectedEof`] when the peer
        /// closed before sending every requested byte.
        #[source]
        source: io::Error,
    },
    /// A send did not complete within the I/O deadline.
    #[error("rank {rank} timed out after {deadline:?} sending to peer {peer} at {address}")]
    SendTimedOut {
        /// Local rank.
        rank: usize,
        /// Destination rank.
        peer: usize,
        /// Peer address of the stream.
        address: SocketAddr,
        /// The elapsed [`MeshDeadlines::io`](super::MeshDeadlines::io) bound.
        deadline: Duration,
    },
    /// A receive did not complete within the I/O deadline.
    #[error("rank {rank} timed out after {deadline:?} receiving from peer {peer} at {address}")]
    RecvTimedOut {
        /// Local rank.
        rank: usize,
        /// Source rank.
        peer: usize,
        /// Peer address of the stream.
        address: SocketAddr,
        /// The elapsed [`MeshDeadlines::io`](super::MeshDeadlines::io) bound.
        deadline: Duration,
    },
    /// An earlier send or receive on this link failed or timed out, so the
    /// stream may hold a partial frame; the link is unusable.
    #[error("rank {rank} link to peer {peer} at {address} is poisoned by an earlier failure")]
    LinkPoisoned {
        /// Local rank.
        rank: usize,
        /// Peer rank of the link.
        peer: usize,
        /// Peer address of the stream.
        address: SocketAddr,
    },
}

/// `Display` of an error followed by each source, separated by `": "`.
pub(crate) struct ErrorChain<'error>(pub(crate) &'error TcpMeshError);

impl std::fmt::Display for ErrorChain<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)?;
        let mut source = std::error::Error::source(self.0);
        while let Some(cause) = source {
            write!(formatter, ": {cause}")?;
            source = cause.source();
        }
        Ok(())
    }
}
