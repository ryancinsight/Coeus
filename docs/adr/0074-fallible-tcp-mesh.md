# ADR 0074: Fallible, deadline-bounded TCP mesh

Status: Accepted  \
Date: 2026-09-23  \
Change class: [major]  \
Board item: COEUS-TCPMESH-FALLIBLE-SETUP (deleted by its delivering PR);
successor COEUS-COMMUNICATOR-FALLIBLE-COLLECTIVES, delivered with
[ADR 0075](0075-fallible-communicator-collectives.md)

## Context

`TcpMesh::new` and `TcpMesh::create_loopback_cluster` panicked on runtime
start, bind, accept, `TCP_NODELAY`, and rank-handshake failures, and asserted
on an out-of-range or duplicate rank announced by a peer. Connecting to a
higher rank retried until it succeeded; only debug builds bounded it, with a
45 s panic. `send` and `recv` panicked on I/O failure. Each of these is a
runtime condition of the network or of a peer, not a caller programming error.

The pinned Moirai `TcpStream::connect` performs a blocking `std` connect
inside its future, so a timeout future cannot interrupt an attempt: a refused
loopback connect takes about 2 s on Windows regardless of any deadline.

## Decision

- `TcpMesh::new` and `create_loopback_cluster` take a `MeshDeadlines` and
  return `Result<_, TcpMeshError>`. `send` and `recv` return
  `Result<(), TcpMeshError>`.
- `TcpMeshError` is a `#[non_exhaustive]` `thiserror` enum. Each variant names
  the local rank; peer-stream variants also carry the peer address and the
  peer rank wherever it is known: always after setup and on the dialling side,
  `None` for an accepted stream before its handshake announces the rank.
  Socket variants carry the `io::Error` as `#[source]` without restating it.
  A peer's invalid rank announcement is `PeerRank { claimed, .. }`; a send or
  receive past the I/O deadline is `SendTimedOut`/`RecvTimedOut`. The
  per-stream setup steps, `TCP_NODELAY` and the rank handshake, share one
  `StreamSetup { step, .. }` variant, so each side builds its peer rank in a
  single constructor.
- Any failed or timed-out send or receive poisons that peer link: either can
  stop mid-frame, and a later receive would then return misaligned bytes as a
  frame. Every later operation on a poisoned link returns `LinkPoisoned`.
  Poisoning also closes the link's socket, so a peer blocked receiving on it
  fails at once with end of stream or a reset instead of waiting out its own
  I/O deadline or reading a truncated frame. The pinned Moirai stream exposes
  only a write-half shutdown, so the close is the stream's drop.
- `MeshDeadlines` holds two bounds, applied in every build profile. The setup
  bound (default 45 s) covers a rank's whole setup. Each connection attempt is
  `std::net::TcpStream::connect_timeout` with the time left, so the bound holds
  per attempt. Only transient failures (refused, timed out, interrupted, would
  block) are retried, after 5 ms doubling to 500 ms; any other error ends setup
  at once. Attempts start at 0, 5, 15, 35, 75, 155, 315 and 635 ms, then every
  500 ms, only before the deadline, so a deadline `d` above 635 ms admits at
  most `7 + ⌈(d − 635 ms) / 500 ms⌉` attempts per peer (96 at 45 s).
  Accept and the rank handshake use the time left as their timeout. Expiry
  reports an `io::ErrorKind::TimedOut` source, or the last transient error.
  The I/O bound (default 300 s: one call moves a whole tensor, and 300 s
  carries 30 GB at 100 MB/s) limits each `send` and `recv` call.
- Configuration errors (`size == 0`, `rank >= size`, address-count mismatch)
  and peer-index misuse in `send`/`recv` stay panics: they are caller bugs.
- `TcpCommunicator` keeps an infallible constructor over an established mesh.
  Revised 2026-09-24: collectives now return `TcpMeshError` instead of
  panicking, and a failed collective poisons every link of its rank; see
  [ADR 0075](0075-fallible-communicator-collectives.md).

Rejected: a separate `connect` constructor on `TcpCommunicator` (a second
construction path for the same mesh), and a per-attempt retry count (the
wall-clock deadline is what a launcher reasons about).

## Migration

- `TcpMesh::new(rank, size, &addresses)` becomes
  `TcpMesh::new(rank, size, &addresses, MeshDeadlines::DEFAULT)?`.
- `TcpMesh::create_loopback_cluster(size)` becomes
  `TcpMesh::create_loopback_cluster(size, MeshDeadlines::DEFAULT)?`.
- `mesh.send(peer, bytes)` and `mesh.recv(peer, bytes)` return a `Result`;
  propagate it with `?`. Release builds previously waited on a silent peer
  forever; they now fail after `MeshDeadlines::io`, which `with_io` raises
  for transfers that need longer.
- Python: mesh setup failures raise `ConnectionError` with the cause chain in
  the message instead of aborting the interpreter thread with a Rust panic.

## Verification and limits

Integration tests force a bind of an address in use (`Bind`, `AddrInUse`),
a connect to a closed loopback port under a 300 ms deadline (`Connect`,
`ConnectionRefused` or `TimedOut` by platform), and an accept with no dialler
(`Accept`, `TimedOut`). Unit tests reject a peer announcing rank 7 to rank 1
(`PeerRank`) and a handshake cut off after 3 of 8 bytes (`StreamSetup`,
`Handshake`, `UnexpectedEof`); receive from a shut-down peer (`Recv`,
`UnexpectedEof`), send to a closed peer (`Send`, reset or abort), and receive
from a silent peer under a 200 ms I/O bound (`RecvTimedOut`, reported within
25 times the bound); send to a peer that never reads (`SendTimedOut`, then
`LinkPoisoned`); receive after a timeout mid-frame (`LinkPoisoned`); fail the
dial-side handshake over a closed write half (`StreamSetup`, `Handshake`,
`peer: Some(1)`, `BrokenPipe`), which also pins the `NoDelay` peer rank because
both dial-side steps come from one constructor; end a peer's pending receive
under a 20 s I/O bound with `Recv` and `UnexpectedEof`, not `RecvTimedOut`,
when its link is poisoned; retry a transient error until the
peer listens, stop at the deadline with the last error, and return a
permanent error after one attempt. The attempt bound is checked against the
backoff schedule the code uses. The connect test's closed port is freed
before dialling, so another process could claim it in between; std has no
bound-but-unlistened TCP socket to hold it. The `NoDelay` step itself has no
forcing test: `setsockopt(TCP_NODELAY)` on a connected socket has no
deterministic failure to provoke. The blocking connect attempt
occupies the mesh's dedicated setup runtime, which runs no other work; an
async connect in Moirai would remove it.
