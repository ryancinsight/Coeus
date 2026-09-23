#![expect(
    clippy::unwrap_used,
    reason = "test assertions surface failures immediately by design"
)]

use super::*;
use moirai::Moirai;

fn runtime() -> Moirai {
    Moirai::builder()
        .worker_threads(1)
        .async_threads(1)
        .build()
        .unwrap()
}

/// Attempts `connect` starts before `deadline` when every attempt fails
/// instantly: one at each cumulative [`Backoff`] offset below the deadline.
fn attempt_starts(deadline: Duration) -> Vec<Duration> {
    let mut pauses = Backoff::new();
    let mut start = Duration::ZERO;
    let mut starts = Vec::new();
    while start < deadline {
        starts.push(start);
        start += pauses.advance();
    }
    starts
}

#[test]
fn backoff_schedule_matches_the_documented_attempt_offsets() {
    let millis = attempt_starts(Duration::from_millis(1_136))
        .iter()
        .map(Duration::as_millis)
        .collect::<Vec<_>>();
    assert_eq!(millis, [0, 5, 15, 35, 75, 155, 315, 635, 1_135]);
}

#[test]
fn attempt_count_matches_the_documented_bound() {
    // `7 + ceil((d - 635 ms) / 500 ms)` for d > 635 ms, as `MeshDeadlines`
    // documents; 635 ms is the offset at which the pause reaches its cap.
    let capped_from = attempt_starts(Duration::from_mins(1))
        .windows(2)
        .find(|pair| pair[1].checked_sub(pair[0]) == Some(MAX_BACKOFF))
        .unwrap()[0];
    assert_eq!(capped_from, Duration::from_millis(635));
    let cap = MAX_BACKOFF.as_millis();
    for millis in [636_u128, 1_000, 1_135, 1_136, 2_000, 45_000] {
        let documented = 7 + (millis - capped_from.as_millis()).div_ceil(cap);
        let deadline = Duration::from_millis(u64::try_from(millis).unwrap());
        assert_eq!(
            attempt_starts(deadline).len() as u128,
            documented,
            "deadline {millis} ms"
        );
    }
    assert_eq!(attempt_starts(MeshDeadlines::DEFAULT.setup()).len(), 96);
}

#[test]
fn permanent_connect_error_returns_without_retrying() {
    let runtime = runtime();
    let mut calls = 0;
    let address = SocketAddr::from(([127, 0, 0, 1], 9));
    let expiry = Instant::now() + Duration::from_secs(30);
    let outcome = runtime.block_on(connect(address, expiry, |_, _| {
        calls += 1;
        Err(io::Error::from(io::ErrorKind::PermissionDenied))
    }));
    assert_eq!(outcome.unwrap_err().kind(), io::ErrorKind::PermissionDenied);
    assert_eq!(calls, 1, "a permanent error must not be retried");
    runtime.shutdown();
}

#[test]
fn transient_connect_errors_are_retried_until_the_peer_listens() {
    let runtime = runtime();
    let listener = std::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
    let address = listener.local_addr().unwrap();
    let mut calls = 0;
    let expiry = Instant::now() + Duration::from_secs(30);
    let stream = runtime
        .block_on(connect(address, expiry, |target, left| {
            calls += 1;
            if calls < 3 {
                Err(io::Error::from(io::ErrorKind::ConnectionRefused))
            } else {
                TcpStream::connect_timeout(target, left)
            }
        }))
        .unwrap();
    assert_eq!(stream.peer_addr().unwrap(), address);
    assert_eq!(calls, 3);
    runtime.shutdown();
}

#[test]
fn transient_connect_errors_stop_at_the_deadline_with_the_last_error() {
    let runtime = runtime();
    let deadline = Duration::from_millis(50);
    let mut calls = 0;
    let address = SocketAddr::from(([127, 0, 0, 1], 9));
    let expiry = Instant::now() + deadline;
    let outcome = runtime.block_on(connect(address, expiry, |_, _| {
        calls += 1;
        Err(io::Error::from(io::ErrorKind::ConnectionRefused))
    }));
    assert_eq!(
        outcome.unwrap_err().kind(),
        io::ErrorKind::ConnectionRefused
    );
    // Pauses overrun their nominal length, so fewer attempts may fit.
    assert!(
        (1..=attempt_starts(deadline).len()).contains(&calls),
        "{calls} attempts within {deadline:?}"
    );
    runtime.shutdown();
}
