# ADR 0074: Fallible, deadline-bounded TCP mesh

Status: Accepted  \
Date: 2026-09-23  \
Change class: [major]  \
Board item: COEUS-TCPMESH-FALLIBLE-SETUP (deleted by its delivering PR; successor
[COEUS-COMMUNICATOR-FALLIBLE-COLLECTIVES](../backlog.md#coeus-communicator-fallible-collectives))

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

- `TcpMesh::new` and `create_loopback_cluster` take a `SetupDeadline` and
  return `Result<_, TcpMeshError>`. `send` and `recv` return
  `Result<(), TcpMeshError>`.
- `TcpMeshError` is a `#[non_exhaustive]` `thiserror` enum. Each variant names
  the local rank and, where one exists, the peer rank and address; socket
  variants carry the `io::Error` as `#[source]` without restating it. A peer's
  invalid rank announcement is `PeerRank { claimed, .. }`.
- One `SetupDeadline` (default 45 s, all build profiles) bounds a rank's
  whole setup. Each connection attempt is `std::net::TcpStream::connect_timeout`
  with the time left, so the bound holds per attempt. Refused attempts back off
  5 ms doubling to 500 ms, admitting at most `8 + ⌈(d − 635 ms) / 500 ms⌉`
  attempts per peer for deadline `d`. Accept and the rank handshake use the
  time left as their timeout. Expiry reports an `io::ErrorKind::TimedOut`
  source, or the last refused attempt's error.
- Configuration errors (`size == 0`, `rank >= size`, address-count mismatch)
  and peer-index misuse in `send`/`recv` stay panics: they are caller bugs.
- `TcpCommunicator` keeps an infallible constructor over an established mesh.
  The `Communicator` trait has no error channel, so a collective's peer I/O
  failure panics with the full error chain; making collectives fallible is the
  successor item.

Rejected: a separate `connect` constructor on `TcpCommunicator` (a second
construction path for the same mesh), and a per-attempt retry count (the
wall-clock deadline is what a launcher reasons about).

## Migration

- `TcpMesh::new(rank, size, &addresses)` becomes
  `TcpMesh::new(rank, size, &addresses, SetupDeadline::default())?`.
- `TcpMesh::create_loopback_cluster(size)` becomes
  `TcpMesh::create_loopback_cluster(size, SetupDeadline::default())?`.
- `mesh.send(peer, bytes)` and `mesh.recv(peer, bytes)` return a `Result`;
  propagate it with `?`.
- Python: mesh setup failures raise `ConnectionError` with the cause chain in
  the message instead of aborting the interpreter thread with a Rust panic.

## Verification and limits

Integration tests force a bind of an address in use (`Bind`, `AddrInUse`),
a connect to a closed loopback port under a 300 ms deadline (`Connect`,
`ConnectionRefused` or `TimedOut` by platform), and an accept with no dialler
(`Accept`, `TimedOut`); a unit test rejects a peer announcing rank 7 to rank 1
(`PeerRank`). The blocking connect attempt occupies the mesh's dedicated setup
runtime, which runs no other work; an async connect in Moirai would remove it.
