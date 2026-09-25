# ADR 0075: Fallible communicator collectives

Status: Accepted  \
Date: 2026-09-24  \
Change class: [major]  \
Board item: COEUS-COMMUNICATOR-FALLIBLE-COLLECTIVES (deleted by its delivering
PR); builds on [ADR 0074](0074-fallible-tcp-mesh.md)

Revision (2026-09-24): an independent review of the delivering PR found that
element-count mismatches were still panics, contradicting the "hostile peer
cannot crash a rank" decision below, and that integer `Sum`/`Product` could
still overflow-panic on a peer-supplied value. Both are now typed errors /
wrapping arithmetic; see the mismatch and overflow bullets in Decision below.

Revision (2026-09-24, second pass): a further review found the address
fields on those typed errors were tested only for "is loopback", which a
wrong-peer mutant survives; `Scalar::wrapping_add_val`/`wrapping_mul_val`
renamed to `total_add`/`total_mul` (defined for every input, echoing
`total_cmp`) with their default body removed so every `Scalar` implementor
states its overflow behavior explicitly; and this record's own citations
(below) named a test module that does not exist. All three are fixed in the
same revision as the tests and citations they describe.

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
- A handshake status byte other than 0 or 1 from a collective's root is
  untrusted peer data, not a caller error: it returns
  `TcpMeshError::InvalidStatus { rank, peer, address, status }` and poisons
  the links like an I/O failure, so a malformed or hostile peer cannot crash
  a rank.
- A root out of range is a caller contract violation and stays a panic. A
  peer-reported element-count mismatch is untrusted peer data, not a caller
  error: the rank that detects the mismatch returns
  `TcpMeshError::NumelMismatch { rank, peer, address, expected, received }`,
  the other ranks return `PeerReportedMismatch { rank, peer, address }` for
  the root's status-0 answer, and both poison the rank's links like any other
  collective failure — a hostile peer announcing any count, valid or not,
  never panics a rank.
- Integer `Sum`/`Product` reductions (`coeus_dist::ops`) fold a peer-supplied
  value through `Scalar::total_add`/`total_mul` (added to `coeus-core`):
  addition/multiplication defined for every input, wrapping on overflow for
  `i8`–`u64` and following IEEE 754 (overflow to ±infinity, the same result
  native `+`/`*` already give) for floats. The name echoes `total_cmp` — "the
  well-defined variant of the usual operator" — and, unlike the
  `wrapping_add_val`/`wrapping_mul_val` names this ADR first shipped with,
  does not imply floats wrap. The trait method has no default body: every
  `Scalar` implementor (native floats, `F16`/`Bf16`, `Complex`, `i8`–`u64`)
  states its own overflow behavior explicitly rather than inheriting one that
  happens to be correct for its type. Overflow checks are enabled in dev/test
  builds, so `a + b` on a hostile or merely large peer value (e.g. `i32::MAX`
  reduced against a local `1`) panicked the root before this change — the
  same class of defect the numel-mismatch fix addresses, and equally a
  violation of "a malformed or hostile peer cannot crash a rank" above.
  Rejected: a checked reduction returning a typed overflow error, which would
  make every local (non-networked) reduction fallible solely to cover the
  peer-input path; wrapping instead matches what release builds already do
  (`+`/`*` wrap silently once checks are off), matches MPI's and NCCL's
  integer-reduction semantics, and needs no new error variant. Float
  reduction semantics are unchanged.
- When a collective returns an error, the contents of the tensors it was
  writing are unspecified: a receive may have filled part of one.

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
`tcp::errors::root_failures` drives rank 0 of a broadcast by hand: a status
byte of 7 yields `InvalidStatus { rank: 1, peer: 0, status: 7 }`, after which
the root reads end of stream and rank 1's next barrier returns
`LinkPoisoned`; a root that sends 8 of 16 payload bytes and shuts down yields
`Recv { peer: 0, UnexpectedEof }`. A Python test drops one rank of a
two-rank cluster and checks that `all_reduce` and the following `barrier`
raise `ConnectionError`.
`tcp::errors::mismatch_cascade` drives a three-rank mesh where a hostile peer
announces `u64::MAX` elements to a broadcast root and where two ranks
all-gather mismatched lengths, asserting the exact `NumelMismatch`
rank/peer/expected/received fields and the `address` field exactly: it
compares the address against the same rank's next `LinkPoisoned { peer,
address }` for that same peer, both read from the same link, which a mutant
reporting a different peer's address fails (the earlier "address is a
loopback IP" check passed such a mutant, since every peer in the test shares
that IP). The same exact-address comparison covers `PeerReportedMismatch`.
`tcp::errors::root_failures` drives a two-rank cluster by hand and covers
`PeerReportedMismatch` and `InvalidStatus`, not `NumelMismatch` (a two-rank
mismatch has no third rank to receive the root's report). `tcp::collectives`
runs `Sum`/`Product` with an `i32::MAX`-valued peer contribution
(`test_tcp_reduce_sum_wraps_on_integer_overflow_from_a_peer` and its
`Product` counterpart) and asserts the exact two's-complement wrapped result
on the root, in place of the panic the unchecked `+`/`*` previously produced
under overflow checks. `core_ops::scalars::scalar_total_ops` runs one test
across every `Scalar` implementor (`i8`–`u64`, `f32`, `f64`, `F16`, `Bf16`,
`Complex`), asserting `MAX.total_add(1) == MIN` and the wrapped product for
integers (cross-verified against `wrapping_mul`), and `MAX.total_add(MAX)`/
`total_mul(MAX)` overflowing to infinity for floats, plus a normal-value case
for each.
