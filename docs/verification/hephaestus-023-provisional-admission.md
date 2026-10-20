# Provisional Hephaestus 0.23 source admission

[patch] Admit both Hephaestus 0.22.2 and 0.23 across all direct edges with
`>=0.22.2, <0.24.0`. Actual [Hephaestus PR #504](https://github.com/ryancinsight/hephaestus/pull/504)
merged and its six native 0.23 packages are indexed. Its required EmbeddingOps
method and host AddAssign bound are source breaks; this consumer does not
implement or reference that seam. Test the actual provider graph rather than
inferring compatibility from the version interval.

The graph also requires provisional Leto/LetoOps 0.46.3 from
[Leto PR #376](https://github.com/ryancinsight/leto/pull/376). Its numerical
contract is independently accepted, but performance remains held. It is not
a released native provider or a finalized publication graph.

No published native package version may be reused for a changed manifest.
Any required patch reservations, internal package floors, normalized package
verification, full shared Python FFI acceptance, owner disposition and exact
resulting-tree review remain prerequisites. This is a local proposal only.
Coeus Miri remains deferred and unverified; no owner branch is edited.
