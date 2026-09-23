use std::io;
use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant};

/// First pause between refused connection attempts.
///
/// Peers start independently, so a higher rank's listener is often not bound
/// yet; a short first pause keeps the common near-simultaneous start fast.
const INITIAL_BACKOFF: Duration = Duration::from_millis(5);

/// Upper bound on the doubling pause between connection attempts, so a peer
/// that binds late is reached within this interval of binding.
const MAX_BACKOFF: Duration = Duration::from_millis(500);

/// Wall-clock bounds on a [`TcpMesh`](super::TcpMesh): one for establishing
/// it and one for each peer send or receive afterwards.
///
/// The setup deadline covers a rank's whole setup: dialling every higher
/// rank (with retries) and accepting every lower rank, including the rank
/// handshake on each stream. Setup that has not completed when it elapses
/// fails with a [`TcpMeshError`](super::TcpMeshError) whose source has kind
/// [`io::ErrorKind::TimedOut`], or the last transient connection error.
///
/// A connection attempt that fails with a transient error (refused while the
/// peer is still starting, timed out, interrupted, would block) is retried
/// after a pause of 5 ms, doubling up to 500 ms; any other error fails setup
/// at once. Attempts start at 0, 5, 15, 35, 75, 155, 315 and 635 ms, then
/// every 500 ms, and only before the deadline, so a setup deadline `d` above
/// 635 ms admits at most `7 + ⌈(d − 635 ms) / 500 ms⌉` attempts per peer
/// (96 for the 45 s default).
///
/// The I/O deadline bounds each [`TcpMesh::send`](super::TcpMesh::send) and
/// [`TcpMesh::recv`](super::TcpMesh::recv) call; a peer that stays silent past
/// it yields `SendTimedOut` or `RecvTimedOut`.
///
/// # Examples
///
/// ```
/// use coeus_dist::MeshDeadlines;
/// use std::time::Duration;
///
/// let deadlines = MeshDeadlines::DEFAULT.with_setup(Duration::from_secs(10));
/// assert_eq!(deadlines.setup(), Duration::from_secs(10));
/// assert_eq!(deadlines.io(), Duration::from_secs(300));
/// assert_eq!(MeshDeadlines::default(), MeshDeadlines::DEFAULT);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshDeadlines {
    setup: Duration,
    io: Duration,
}

impl MeshDeadlines {
    /// 45 s to establish the mesh: long enough for peer processes started by
    /// a launcher to bind. 300 s per send or receive: one call moves a whole
    /// tensor, and 300 s carries 30 GB at 100 MB/s.
    pub const DEFAULT: Self = Self {
        setup: Duration::from_secs(45),
        io: Duration::from_mins(5),
    };

    /// These deadlines with the setup bound replaced. A zero bound fails
    /// setup that has any peer to reach.
    #[must_use]
    pub const fn with_setup(self, setup: Duration) -> Self {
        Self { setup, ..self }
    }

    /// These deadlines with the per-call send and receive bound replaced.
    #[must_use]
    pub const fn with_io(self, io: Duration) -> Self {
        Self { io, ..self }
    }

    /// Bound on establishing the mesh.
    #[must_use]
    pub const fn setup(self) -> Duration {
        self.setup
    }

    /// Bound on one send or receive call.
    #[must_use]
    pub const fn io(self) -> Duration {
        self.io
    }

    /// The instant at which setup starting now expires.
    pub(super) fn setup_expiry_from_now(self) -> Instant {
        Instant::now() + self.setup
    }
}

impl Default for MeshDeadlines {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Pauses between connection attempts: [`INITIAL_BACKOFF`] doubling up to
/// [`MAX_BACKOFF`].
struct Backoff {
    next: Duration,
}

impl Backoff {
    const fn new() -> Self {
        Self {
            next: INITIAL_BACKOFF,
        }
    }

    fn advance(&mut self) -> Duration {
        let pause = self.next;
        self.next = (pause * 2).min(MAX_BACKOFF);
        pause
    }
}

/// Whether a failed connection attempt may succeed if repeated: the peer's
/// listener is not bound yet, or the attempt was cut short.
fn is_transient(kind: io::ErrorKind) -> bool {
    matches!(
        kind,
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::TimedOut
            | io::ErrorKind::Interrupted
            | io::ErrorKind::WouldBlock
    )
}

/// Time left until `expiry`, or a `TimedOut` error once it has passed.
fn remaining(expiry: Instant) -> io::Result<Duration> {
    let left = expiry.saturating_duration_since(Instant::now());
    if left.is_zero() {
        Err(deadline_elapsed())
    } else {
        Ok(left)
    }
}

/// The error reported when the setup deadline elapses mid-operation.
fn deadline_elapsed() -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, "mesh setup deadline elapsed")
}

/// Await `operation` for at most the time left until `expiry`.
pub(super) async fn within<F, T>(expiry: Instant, operation: F) -> io::Result<T>
where
    F: std::future::Future<Output = io::Result<T>>,
{
    moirai_async::timeout(remaining(expiry)?, operation)
        .await
        .map_err(|_| deadline_elapsed())?
}

/// Dial `address` with `dial` until it succeeds, fails permanently, or
/// `expiry` passes; `dial` receives the time left as its attempt bound.
///
/// Returns a permanent error at once, the last transient error when no
/// further attempt fits before the deadline, and a `TimedOut` error when the
/// deadline has passed before the first attempt. Production passes
/// [`TcpStream::connect_timeout`]: the async `connect` of the pinned Moirai
/// performs a blocking connect inside its future, which no timeout future can
/// interrupt (a refused loopback connect takes about 2 s on Windows). The
/// blocking attempt runs on the mesh's dedicated setup runtime, which has no
/// other work to starve.
pub(super) async fn connect<D>(
    address: SocketAddr,
    expiry: Instant,
    mut dial: D,
) -> io::Result<TcpStream>
where
    D: FnMut(&SocketAddr, Duration) -> io::Result<TcpStream>,
{
    let mut pauses = Backoff::new();
    let mut left = remaining(expiry)?;
    loop {
        let error = match dial(&address, left) {
            Ok(stream) => return Ok(stream),
            Err(error) => error,
        };
        if !is_transient(error.kind()) {
            return Err(error);
        }
        let pause = pauses.advance();
        if remaining(expiry).map_or(true, |now_left| pause >= now_left) {
            return Err(error);
        }
        moirai_async::sleep(pause).await;
        // A pause that overran the deadline ends with the attempt's error.
        left = remaining(expiry).map_err(|_| error)?;
    }
}

#[cfg(test)]
mod tests;
