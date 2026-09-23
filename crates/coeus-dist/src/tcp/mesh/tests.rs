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
    two_ranks_with([deadlines, deadlines])
}

/// [`two_ranks`] with separate deadlines per rank.
fn two_ranks_with(deadlines: [MeshDeadlines; 2]) -> (TcpMesh, TcpMesh, [SocketAddr; 2]) {
    let endpoints = [0, 1].map(|rank| {
        let runtime = TcpMesh::runtime(rank).unwrap();
        let listener = TcpMesh::bind(&runtime, rank, LOOPBACK).unwrap();
        (listener, runtime)
    });
    let addresses = [0, 1].map(|rank| endpoints[rank].0.local_addr().unwrap());
    let [(listener_0, runtime_0), (listener_1, runtime_1)] = endpoints;
    let (rank_0, rank_1) = thread::scope(|scope| {
        let rank_1 = scope.spawn(|| {
            TcpMesh::from_listener(1, 2, &addresses, &listener_1, runtime_1, deadlines[1], DIAL)
                .unwrap()
        });
        let rank_0 =
            TcpMesh::from_listener(0, 2, &addresses, &listener_0, runtime_0, deadlines[0], DIAL)
                .unwrap();
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
    let outcome = TcpMesh::from_listener(1, 2, &addresses, &listener, runtime, deadlines, DIAL);
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
        Err(TcpMeshError::StreamSetup {
            rank,
            peer,
            address,
            step: StreamStep::Handshake,
            source,
        }) => {
            assert_eq!(rank, 1);
            assert_eq!(peer, None, "the peer rank was never announced");
            assert_eq!(address, dialler);
            assert_eq!(source.kind(), io::ErrorKind::UnexpectedEof);
        }
        Err(other) => panic!("expected a Handshake StreamSetup, got {other:?}"),
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

/// Short I/O bound for timeout tests: long enough that loopback delivers
/// any bytes already written, short enough to keep each test brief.
const SHORT_IO: Duration = Duration::from_millis(200);

/// Upper bound on how long a `SHORT_IO` timeout may take to report: 25 times
/// the bound absorbs scheduler delay on a loaded runner (hosted CI runs this
/// suite about 15 times slower than a developer host) while staying far
/// below the 300 s default, so a timeout that ignored the configured bound
/// cannot pass.
const SHORT_IO_REPORT_BOUND: Duration = SHORT_IO.saturating_mul(25);

#[test]
fn recv_from_a_silent_peer_times_out_at_the_configured_io_deadline() {
    let (mut rank_0, mut rank_1, addresses) = two_ranks(MeshDeadlines::DEFAULT.with_io(SHORT_IO));

    let mut bytes = [0u8; 8];
    let started = Instant::now();
    let outcome = rank_0.recv(1, &mut bytes);
    let elapsed = started.elapsed();
    match outcome {
        Err(TcpMeshError::RecvTimedOut {
            rank,
            peer,
            address,
            deadline,
        }) => {
            assert_eq!((rank, peer), (0, 1));
            assert_eq!(address, addresses[1]);
            assert_eq!(deadline, SHORT_IO);
        }
        other => panic!("expected RecvTimedOut, got {other:?}"),
    }
    assert!(
        elapsed < SHORT_IO_REPORT_BOUND,
        "a {SHORT_IO:?} bound took {elapsed:?} to fire"
    );
    rank_1.shutdown();
    rank_0.shutdown();
}

#[test]
fn send_to_a_peer_that_never_reads_times_out_and_poisons_the_link() {
    /// Bytes per send. Windows loopback grows its send backlog to accept a
    /// single 256 MiB write, so the test keeps writing until the buffers of
    /// a peer that never reads are full.
    const CHUNK_BYTES: usize = 64 * 1024 * 1024;
    /// Bounds the writes without timing: 4 GiB unread exceeds any loopback
    /// buffering, so a send that never stalls fails the test.
    const MAX_CHUNKS: usize = 64;
    let (mut rank_0, mut rank_1, addresses) = two_ranks(MeshDeadlines::DEFAULT.with_io(SHORT_IO));

    let chunk = vec![0u8; CHUNK_BYTES];
    let first_failure = (0..MAX_CHUNKS)
        .find_map(|_| rank_0.send(1, &chunk).err())
        .unwrap_or_else(|| panic!("{MAX_CHUNKS} unread chunks were all accepted"));
    match first_failure {
        TcpMeshError::SendTimedOut {
            rank,
            peer,
            address,
            deadline,
        } => {
            assert_eq!((rank, peer), (0, 1));
            assert_eq!(address, addresses[1]);
            assert_eq!(deadline, SHORT_IO);
        }
        other => panic!("expected SendTimedOut, got {other:?}"),
    }
    match rank_0.send(1, &[1]) {
        Err(TcpMeshError::LinkPoisoned {
            rank,
            peer,
            address,
        }) => {
            assert_eq!((rank, peer), (0, 1));
            assert_eq!(address, addresses[1]);
        }
        other => panic!("expected LinkPoisoned, got {other:?}"),
    }
    rank_1.shutdown();
    rank_0.shutdown();
}

/// A receive that times out after half a frame poisons the link: the rest of
/// that frame would otherwise be read as the start of the next one.
#[test]
fn recv_timed_out_mid_frame_poisons_the_link() {
    let (mut rank_0, mut rank_1, addresses) = two_ranks(MeshDeadlines::DEFAULT.with_io(SHORT_IO));

    rank_1.send(0, &[1, 2, 3, 4]).unwrap();
    let mut frame = [0u8; 8];
    assert!(matches!(
        rank_0.recv(1, &mut frame),
        Err(TcpMeshError::RecvTimedOut { peer: 1, .. })
    ));
    // The frame's remainder plus a whole next frame are now in flight; an
    // unpoisoned link would return bytes 5..=12 as a frame.
    rank_1.send(0, &[5, 6, 7, 8, 9, 10, 11, 12]).unwrap();
    match rank_0.recv(1, &mut frame) {
        Err(TcpMeshError::LinkPoisoned {
            rank,
            peer,
            address,
        }) => {
            assert_eq!((rank, peer), (0, 1));
            assert_eq!(address, addresses[1]);
        }
        other => panic!("expected LinkPoisoned, got {other:?} with frame {frame:?}"),
    }
    rank_1.shutdown();
    rank_0.shutdown();
}

/// A dialled stream whose write half is already closed fails the rank
/// handshake; the error names the rank that was being dialled.
#[test]
fn dial_side_handshake_failure_names_the_dialled_peer() {
    fn dial_with_write_half_closed(
        address: &SocketAddr,
        left: Duration,
    ) -> io::Result<std::net::TcpStream> {
        let stream = std::net::TcpStream::connect_timeout(address, left)?;
        stream.shutdown(std::net::Shutdown::Write)?;
        Ok(stream)
    }

    let peer_listener = std::net::TcpListener::bind(LOOPBACK).unwrap();
    let peer_address = peer_listener.local_addr().unwrap();
    let acceptor = thread::spawn(move || peer_listener.accept().map(drop));

    let runtime = TcpMesh::runtime(0).unwrap();
    let listener = TcpMesh::bind(&runtime, 0, LOOPBACK).unwrap();
    let addresses = [listener.local_addr().unwrap(), peer_address];
    let outcome = TcpMesh::from_listener(
        0,
        2,
        &addresses,
        &listener,
        runtime,
        MeshDeadlines::DEFAULT.with_setup(Duration::from_secs(10)),
        dial_with_write_half_closed,
    );
    acceptor.join().unwrap().unwrap();

    match outcome {
        Err(TcpMeshError::StreamSetup {
            rank,
            peer,
            address,
            step: StreamStep::Handshake,
            source,
        }) => {
            assert_eq!(rank, 0);
            assert_eq!(peer, Some(1));
            assert_eq!(address, peer_address);
            // Writing after a local write shutdown: EPIPE on Unix, WSAESHUTDOWN on
            // Windows; std maps both to BrokenPipe.
            assert_eq!(source.kind(), io::ErrorKind::BrokenPipe);
        }
        Err(other) => panic!("expected a Handshake StreamSetup, got {other:?}"),
        Ok(_) => panic!("a handshake over a closed write half must fail"),
    }
}

/// Poisoning closes the socket, so a peer blocked in `recv` on that link sees
/// end of stream at once rather than timing out at its own I/O deadline.
#[test]
fn poisoning_a_link_ends_the_peer_s_pending_recv() {
    /// Rank 1's I/O bound: long enough that loopback delivers the close first
    /// on any runner, short enough that a missing close fails this test with
    /// `RecvTimedOut` before the 60 s nextest termination.
    const PEER_IO: Duration = Duration::from_secs(20);
    let (mut rank_0, mut rank_1, addresses) = two_ranks_with([
        MeshDeadlines::DEFAULT.with_io(SHORT_IO),
        MeshDeadlines::DEFAULT.with_io(PEER_IO),
    ]);

    rank_1.send(0, &[1, 2, 3, 4]).unwrap();
    let peer_outcome = thread::scope(|scope| {
        let rank_1 = &rank_1;
        let pending = scope.spawn(move || {
            let mut frame = [0u8; 8];
            rank_1.recv(0, &mut frame)
        });
        let mut frame = [0u8; 8];
        assert!(matches!(
            rank_0.recv(1, &mut frame),
            Err(TcpMeshError::RecvTimedOut { peer: 1, .. })
        ));
        pending.join().unwrap()
    });
    // Without the close, rank 1 would report `RecvTimedOut` after its bound.
    match peer_outcome {
        Err(TcpMeshError::Recv {
            rank,
            peer,
            address,
            source,
        }) => {
            assert_eq!((rank, peer), (1, 0));
            // Rank 1 accepted rank 0, so the address is rank 0's outgoing
            // loopback endpoint.
            assert_eq!(address.ip(), addresses[0].ip());
            // Rank 0 had read everything sent to it, so closing sends FIN.
            assert_eq!(source.kind(), io::ErrorKind::UnexpectedEof);
        }
        other => panic!("expected rank 1's Recv to end with the close, got {other:?}"),
    }
    rank_1.shutdown();
    rank_0.shutdown();
}
