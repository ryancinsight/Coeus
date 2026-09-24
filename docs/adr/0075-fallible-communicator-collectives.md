# ADR 0075: Fallible communicator collectives

Status: Accepted  \
Date: 2026-09-24  \
Change class: [major]  \
Board item: COEUS-COMMUNICATOR-FALLIBLE-COLLECTIVES (deleted by its delivering
PR); builds on [ADR 0074](0074-fallible-tcp-mesh.md)

## Context

After ADR 0074, `TcpMesh::send` and `recv` return `TcpMeshError`, but the
`Communicator` trait's collectives returned `()`. `TcpCommunicator` therefore
turned every peer I/O failure inside a collective into a panic. A rank whose
collective failed also kept its other links open, so peers still in the
collective waited out their own I/O deadline (300 s by default).

`LocalCommunicator` exchanges data through shared memory and a `Barrier`
between threads of one process; nothing in it performs I/O that can fail.

## Decision

- `Communicator` gains an associated `type Error: Error + Send + Sync +
  'static`, and every collective (`barrier`, `all_reduce`, `broadcast`,
  `all_gather`, `reduce`, `gather`, `scatter`) returns `Result<(),
  Self::Error>`. `TcpCommunicator::Error` is `TcpMeshError`;
  `LocalCommunicator::Error` is `std::convert::Infallible`.
- A TCP collective that fails on any link poisons every link of that rank
  before returning the error. Its streams are at unknown frame positions, and
  closing them ends the waits of the peers still in the collective, so every
  surviving rank returns a typed error at once rather than after its I/O
  deadline. Later collectives on that communicator return `LinkPoisoned`.
- `synchronize_gradients` returns `GradientSyncError<B::Error, C::Error>`, a
  `#[non_exhaustive]` enum with `Communicator` and `Backend` variants that keep
  the cause as `#[source]`.
- Caller contract violations (a root out of range, mismatched element counts,
  a failed element-count handshake) stay panics, as before.

Rejected: a crate-level `CommunicatorError` wrapping `TcpMeshError`. It would
force `LocalCommunicator` to declare failures it cannot produce, and a closed
error enum owned by the trait's crate would make a communicator implemented
elsewhere wrap its failures in someone else's type. The associated type lets
each implementation state exactly what it can fail with, and `Infallible`
makes the local results irrefutable (`let Ok(()) = ...`).

## Migration

- Handle the `Result` of every collective: `comm.all_reduce::<T, B, Sum>(&mut
  t, &b)?` for `TcpCommunicator`, `let Ok(()) = comm.barrier();` for
  `LocalCommunicator`.
- Code generic over `C: Communicator` propagates `C::Error`.
- `synchronize_gradients` callers match `GradientSyncError::Backend(source)`
  for the error previously returned directly.
- After any collective error on a `TcpCommunicator`, rebuild the mesh; the old
  one fails every further collective with `LinkPoisoned`.
- Python: TCP collectives raise `ConnectionError` with the cause chain in the
  message instead of aborting the interpreter thread with a Rust panic.

## Verification and limits

`tcp::errors::peer_loss` shuts down rank 2 of 3, runs an all-reduce on the
two survivors under a 20 s I/O bound, and asserts that rank 0 reports
`Recv { peer: 2, UnexpectedEof }`, rank 1 reports `Recv { peer: 0,
UnexpectedEof }`, and rank 0's next barrier returns `LinkPoisoned`. Without
the poison-on-failure step rank 1 reports `RecvTimedOut` after 20 s instead.
A Python test drops one rank of a two-rank cluster and checks that
`all_reduce` and the following `barrier` raise `ConnectionError`. A peer
lost in the middle of a payload transfer, rather than before the collective,
is covered by the same poisoning path but has no dedicated test.
