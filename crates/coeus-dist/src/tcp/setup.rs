use moirai_async::TcpStream;
use std::io;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

/// First pause between refused connection attempts.
///
/// Peers start independently, so a higher rank's listener is often not bound
/// yet; a short first pause keeps the common near-simultaneous start fast.
const INITIAL_BACKOFF: Duration = Duration::from_millis(5);

/// Upper bound on the doubling pause between connection attempts, so a peer
/// that binds late is reached within this interval of binding.
const MAX_BACKOFF: Duration = Duration::from_millis(500);

/// Wall-clock bound on establishing a [`TcpMesh`](super::TcpMesh).
///
/// One deadline covers the whole setup of a rank: dialling every higher rank
/// (with retries) and accepting every lower rank, including the rank
/// handshake on each stream. Setup that has not completed when it elapses
/// fails with a [`TcpMeshError`](super::TcpMeshError) whose source has kind
/// [`io::ErrorKind::TimedOut`], or the last refused attempt's error.
///
/// Connection attempts to one peer pause 5 ms after the first refusal and
/// double the pause up to 500 ms, so a deadline of `d` admits at most
/// `8 + ⌈(d − 635 ms) / 500 ms⌉` attempts per peer (97 for the 45 s
/// default).
///
/// # Examples
///
/// ```
/// use coeus_dist::SetupDeadline;
/// use std::time::Duration;
///
/// let deadline = SetupDeadline::new(Duration::from_secs(10));
/// assert_eq!(deadline.duration(), Duration::from_secs(10));
/// assert_eq!(SetupDeadline::default().duration(), Duration::from_secs(45));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupDeadline(Duration);

impl SetupDeadline {
    /// Bound setup by `duration`. A zero duration fails setup that has any
    /// peer to reach.
    #[must_use]
    pub const fn new(duration: Duration) -> Self {
        Self(duration)
    }

    /// The configured bound.
    #[must_use]
    pub const fn duration(self) -> Duration {
        self.0
    }

    /// The instant at which setup starting now expires.
    pub(super) fn expiry_from_now(self) -> Instant {
        Instant::now() + self.0
    }
}

impl Default for SetupDeadline {
    /// 45 s: long enough for peer processes started by a launcher to bind.
    fn default() -> Self {
        Self(Duration::from_secs(45))
    }
}

/// Time left until `expiry`, or a `TimedOut` error once it has passed.
pub(super) fn remaining(expiry: Instant) -> io::Result<Duration> {
    let left = expiry.saturating_duration_since(Instant::now());
    if left.is_zero() {
        Err(deadline_elapsed())
    } else {
        Ok(left)
    }
}

/// The error reported when the setup deadline elapses mid-operation.
pub(super) fn deadline_elapsed() -> io::Error {
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

/// Dial `address` until a connection succeeds or `expiry` passes.
///
/// Returns the last attempt's error when the deadline passes during a pause,
/// and a `TimedOut` error when it passes during an attempt.
///
/// Each attempt is a `connect_timeout` bounded by the time left: the async
/// `TcpStream::connect` of the pinned Moirai performs a blocking connect
/// inside its future, which no timeout future can interrupt (a refused
/// loopback connect takes about 2 s on Windows). The blocking attempt runs on
/// the mesh's dedicated setup runtime, which has no other work to starve.
pub(super) async fn connect(address: SocketAddr, expiry: Instant) -> io::Result<TcpStream> {
    let mut backoff = INITIAL_BACKOFF;
    loop {
        let attempt = std::net::TcpStream::connect_timeout(&address, remaining(expiry)?);
        let last_error = match attempt.and_then(TcpStream::from_std) {
            Ok(stream) => return Ok(stream),
            Err(error) => error,
        };
        let Ok(left) = remaining(expiry) else {
            return Err(last_error);
        };
        moirai_async::sleep(backoff.min(left)).await;
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
}
