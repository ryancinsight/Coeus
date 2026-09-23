#![expect(
    clippy::unwrap_used,
    reason = "test assertions surface failures immediately by design"
)]

use super::*;
use std::sync::Arc;

const LOOPBACK: SocketAddr =
    SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), 0);

/// A two-rank loopback mesh plus each rank's listener address, which is the
/// address rank 0 dials rank 1 at.
fn two_ranks(deadlines: MeshDeadlines) -> (TcpMesh, TcpMesh, [SocketAddr; 2]) {
    let endpoints = [0, 1].map(|rank| {
        let runtime = TcpMesh::runtime(rank).unwrap();
        let listener = TcpMesh::bind(&runtime, rank, LOOPBACK).unwrap();
        (listener, runtime)
    });
    let addresses = [0, 1].map(|rank| endpoints[rank].0.local_addr().unwrap());
    let [(listener_0, runtime_0), (listener_1, runtime_1)] = endpoints;
    let (rank_0, rank_1) = thread::scope(|scope| {
        let rank_1 = scope.spawn(|| {
            TcpMesh::from_listener(1, 2, &addresses, &listener_1, runtime_1, deadlines).unwrap()
        });
        let rank_0 =
            TcpMesh::from_listener(0, 2, &addresses, &listener_0, runtime_0, deadlines).unwrap();
        (rank_0, rank_1.join().unwrap())
    });
    (rank_0, rank_1, addresses)
}

/// `shutdown()` documents that it stops the runtime, joining its worker
/// thread, before returning. Prove it: a task spawned on the mesh's own
/// runtime holds a marker `Arc` clone that only drops when the worker
/// thread that ran the task unwinds its stack. If `shutdown()` returned
/// before that join happened, the clone would still be live and the
/// strong count would read 2, not 1.
#[test]
fn shutdown_joins_the_runtime_worker_thread() {
    let mut mesh =
        TcpMesh::create_loopback_cluster(NonZeroUsize::new(1).unwrap(), MeshDeadlines::DEFAULT)
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

/// Run rank 1 of 2 against one raw dialler that `dial` drives, returning
/// the setup outcome and the dialler's local address.
fn accept_one_raw_dialler(
    dial: impl FnOnce(std::net::TcpStream) + Send + 'static,
) -> (Result<TcpMesh, TcpMeshError>, SocketAddr) {
    let runtime = TcpMesh::runtime(1).unwrap();
    let listener = TcpMesh::bind(&runtime, 1, LOOPBACK).unwrap();
    let listen_address = listener.local_addr().unwrap();
    let dialler = thread::spawn(move || {
        let stream = std::net::TcpStream::connect(listen_address).unwrap();
        let local = stream.local_addr().unwrap();
        dial(stream);
        local
    });
    // Rank 1 of 2 dials nobody and accepts exactly one lower rank; rank 0's
    // address is never dialled.
    let addresses = [SocketAddr::from(([127, 0, 0, 1], 9)), listen_address];
    let deadlines = MeshDeadlines::DEFAULT.with_setup(Duration::from_secs(10));
    let outcome = TcpMesh::from_listener(1, 2, &addresses, &listener, runtime, deadlines);
    (outcome, dialler.join().unwrap())
}

/// A dialler announcing a rank that is not below the accepting rank is
/// rejected with its claim and address, not trusted as a peer slot index.
#[test]
fn accept_rejects_a_peer_rank_that_is_not_lower() {
    const CLAIMED: u64 = 7;
    let (outcome, dialler) = accept_one_raw_dialler(|mut stream| {
        use std::io::Write;
        stream.write_all(&CLAIMED.to_le_bytes()).unwrap();
    });
    match outcome {
        Err(TcpMeshError::PeerRank {
            rank,
            address,
            claimed,
        }) => {
            assert_eq!(rank, 1);
            assert_eq!(address, dialler);
            assert_eq!(claimed, CLAIMED);
        }
        Err(other) => panic!("expected PeerRank, got {other:?}"),
        Ok(_) => panic!("a peer claiming rank {CLAIMED} must be rejected"),
    }
}

/// A dialler that closes after three of its eight rank bytes fails the
/// handshake with the end of stream it caused.
#[test]
fn accept_reports_a_handshake_cut_short_by_the_peer() {
    let (outcome, dialler) = accept_one_raw_dialler(|mut stream| {
        use std::io::Write;
        stream.write_all(&[0, 0, 0]).unwrap();
        stream.shutdown(std::net::Shutdown::Write).unwrap();
    });
    match outcome {
        Err(TcpMeshError::Handshake {
            rank,
            peer,
            address,
            source,
        }) => {
            assert_eq!(rank, 1);
            assert_eq!(peer, None, "the peer rank was never announced");
            assert_eq!(address, dialler);
            assert_eq!(source.kind(), io::ErrorKind::UnexpectedEof);
        }
        Err(other) => panic!("expected Handshake, got {other:?}"),
        Ok(_) => panic!("a truncated rank handshake must fail"),
    }
}

#[test]
fn recv_from_a_closed_peer_is_an_unexpected_end_of_stream() {
    let (mut rank_0, mut rank_1, addresses) = two_ranks(MeshDeadlines::DEFAULT);
    rank_1.shutdown();
    drop(rank_1);

    let mut bytes = [0u8; 8];
    match rank_0.recv(1, &mut bytes) {
        Err(TcpMeshError::Recv {
            rank,
            peer,
            address,
            source,
        }) => {
            assert_eq!((rank, peer), (0, 1));
            assert_eq!(address, addresses[1]);
            assert_eq!(source.kind(), io::ErrorKind::UnexpectedEof);
        }
        other => panic!("expected Recv, got {other:?}"),
    }
    rank_0.shutdown();
}

#[test]
fn send_to_a_closed_peer_fails_with_the_reset() {
    /// The peer's reset arrives after the first write into the closed
    /// socket; loopback delivers it within a few writes, so this bounds the
    /// loop without timing.
    const MAX_WRITES: usize = 1_024;
    let (mut rank_0, mut rank_1, addresses) = two_ranks(MeshDeadlines::DEFAULT);
    rank_1.shutdown();
    drop(rank_1);

    let chunk = vec![0u8; 64 * 1024];
    let error = (0..MAX_WRITES)
        .find_map(|_| rank_0.send(1, &chunk).err())
        .unwrap_or_else(|| panic!("{MAX_WRITES} writes to a closed peer all succeeded"));
    match error {
        TcpMeshError::Send {
            rank,
            peer,
            address,
            source,
        } => {
            assert_eq!((rank, peer), (0, 1));
            assert_eq!(address, addresses[1]);
            assert!(
                matches!(
                    source.kind(),
                    io::ErrorKind::ConnectionReset
                        | io::ErrorKind::ConnectionAborted
                        | io::ErrorKind::BrokenPipe
                ),
                "unexpected source kind {:?}",
                source.kind()
            );
        }
        other => panic!("expected Send, got {other:?}"),
    }
    rank_0.shutdown();
}

#[test]
fn recv_from_a_silent_peer_times_out_at_the_io_deadline() {
    let deadline = Duration::from_millis(200);
    let (mut rank_0, mut rank_1, addresses) = two_ranks(MeshDeadlines::DEFAULT.with_io(deadline));

    let mut bytes = [0u8; 8];
    match rank_0.recv(1, &mut bytes) {
        Err(TcpMeshError::RecvTimedOut {
            rank,
            peer,
            address,
            deadline: elapsed,
        }) => {
            assert_eq!((rank, peer), (0, 1));
            assert_eq!(address, addresses[1]);
            assert_eq!(elapsed, deadline);
        }
        other => panic!("expected RecvTimedOut, got {other:?}"),
    }
    rank_1.shutdown();
    rank_0.shutdown();
}
