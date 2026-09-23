//! TCP mesh setup failures surface as typed errors, not panics.

use coeus_dist::{MeshDeadlines, TcpMesh, TcpMeshError};
use std::io::ErrorKind;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::time::Duration;

/// Short enough to keep each failing setup well inside the test budget, long
/// enough for a loopback connection attempt to be refused on every platform.
const SHORT_DEADLINE: MeshDeadlines = MeshDeadlines::DEFAULT.with_setup(Duration::from_millis(300));

fn ephemeral_loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

/// A loopback address whose port had a listener that has since closed.
///
/// Another process may bind the freed port before the test dials it; the
/// window is one ephemeral allocation, and std offers no bound-but-unlistened
/// TCP socket that would hold the port closed.
fn closed_loopback_port() -> SocketAddr {
    let listener = TcpListener::bind(ephemeral_loopback()).unwrap();
    listener.local_addr().unwrap()
}

#[test]
fn bind_of_an_address_in_use_is_a_typed_bind_error() {
    let occupant = TcpListener::bind(ephemeral_loopback()).unwrap();
    let occupied = occupant.local_addr().unwrap();

    match TcpMesh::new(0, 1, &[occupied], SHORT_DEADLINE) {
        Err(TcpMeshError::Bind {
            rank,
            address,
            source,
        }) => {
            assert_eq!(rank, 0);
            assert_eq!(address, occupied);
            assert_eq!(source.kind(), ErrorKind::AddrInUse);
        }
        Err(other) => panic!("expected Bind, got {other:?}"),
        Ok(_) => panic!("binding {occupied} while it is held must fail"),
    }
    drop(occupant);
}

#[test]
fn connect_without_a_listener_is_a_typed_connect_error_at_the_deadline() {
    let unreachable = closed_loopback_port();

    match TcpMesh::new(0, 2, &[ephemeral_loopback(), unreachable], SHORT_DEADLINE) {
        Err(TcpMeshError::Connect {
            rank,
            peer,
            address,
            source,
        }) => {
            assert_eq!((rank, peer), (0, 1));
            assert_eq!(address, unreachable);
            // Linux refuses a closed loopback port at once, so the last
            // attempt's refusal is reported; Windows retries the SYN for
            // longer than the deadline, which then interrupts the attempt.
            assert!(
                matches!(
                    source.kind(),
                    ErrorKind::ConnectionRefused | ErrorKind::TimedOut
                ),
                "unexpected source kind {:?}",
                source.kind()
            );
        }
        Err(other) => panic!("expected Connect, got {other:?}"),
        Ok(_) => panic!("connecting to closed port {unreachable} must fail"),
    }
}

#[test]
fn accept_with_no_dialler_is_a_typed_accept_timeout() {
    // Rank 1 of 2 dials nobody and waits for rank 0, which never starts;
    // rank 0's address is never dialled.
    let addresses = [
        SocketAddr::from((Ipv4Addr::LOCALHOST, 9)),
        ephemeral_loopback(),
    ];

    match TcpMesh::new(1, 2, &addresses, SHORT_DEADLINE) {
        Err(TcpMeshError::Accept {
            rank,
            address,
            source,
        }) => {
            assert_eq!(rank, 1);
            assert_eq!(address.ip(), Ipv4Addr::LOCALHOST);
            assert_ne!(address.port(), 0, "the bound listener port is reported");
            assert_eq!(source.kind(), ErrorKind::TimedOut);
        }
        Err(other) => panic!("expected Accept, got {other:?}"),
        Ok(_) => panic!("accepting with no dialler must time out"),
    }
}
