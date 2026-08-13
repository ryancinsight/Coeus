# coeus-dist

Collective communication for [Coeus](../../README.md).

Provides the `Communicator` trait and two host-side implementations for
distributing tensor work.

## What is here

- `Communicator` — the collective-operation trait.
- `LocalCommunicator` — a thread-based implementation for single-process
  multi-worker runs.
- A TCP implementation providing collectives and mesh setup over `std` sockets.
- Reduce-operation tags: `Sum`, `Max`, `Min`, `Product`.
- `synchronize_gradients` for data-parallel training.

## Scope

Collectives are hand-rolled over TCP and run on the host. There is no NCCL,
RCCL, or MPI backend, and no device-to-device transport.

## Documentation

API docs: <https://docs.rs/coeus-dist>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
