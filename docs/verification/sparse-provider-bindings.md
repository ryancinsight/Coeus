# Sparse provider review bindings — 2026-10-15

Accepted Leto ADR 0036 adds the checked borrowed-CSR capability in 0.47.
Canonical default Git sources remain unpinned in manifests. The review lock
selects the actual independently published branch commits below; Cargo
regenerates it and independently verifies `metadata --offline --locked`.

| Review | Version / contract | Exact source |
| --- | --- | --- |
| [Leto #362](https://github.com/ryancinsight/leto/pull/362) | 0.47.0, signed stored-order views and prepared row tasks | `a277833444ed6d54b5b58cf72bf8911c4c9cbb3a` |
| [Hephaestus #497](https://github.com/ryancinsight/hephaestus/pull/497) | 0.22.1, both Leto layers accept `>=0.46.0, <0.48.0` | `a153f475fbee1042fc323a845400da89c9737c94` |
| [Apollo #674](https://github.com/ryancinsight/apollo/pull/674) | FFT 0.29.2 and affected family patches; same compatible interval | `f8fa292ce86315875adaf2eeb365ef862aba63f4` |
| [Coeus #533](https://github.com/ryancinsight/Coeus/pull/533) | 0.13.0, accepted backend unwind contract and checked tensor adapter | Review head records the current complete candidate |

The compatibility patches preserve the merged 0.46 cohort, all 1337/1064
Hephaestus/Apollo Rust files and every external locked package (228/245).
Their committed locks follow merged defaults, so they can be reviewed before
Leto 0.47 integration. They add no conversion layer, source pin or path patch.
Existing 0.22/0.29 consumer constraints accept the new compatible patches.

The Coeus review seed consists only of actual packages from the provider-owned
locks, assigning their owning workspace packages the actual reviewed Git
source above. Cargo then resolves/prunes all packages from their real manifests
and independently accepts the regenerated lock with `--locked`. The resulting
full graph has ten canonical providers and zero excess identities. This is a
locked review-graph proof; it does not claim unseeded defaults or registry
availability before the provider reviews integrate.

Publication must keep the shared graph coherent: integrate/publish compatible
Hephaestus and Apollo bindings before exposing Leto 0.47 to fresh consumers,
then refresh Coeus defaults and execute unseeded/registry consumer gates before
its release. All merges remain subject to actual checks and authorization.

Workflow/settings audit: Hephaestus uses master-push pending publication and
workflow Pages; Apollo uses main-push pending publication and legacy main/root
Pages. Leto also uses legacy Pages; Coeus uses workflow Pages. Branch/PR checks
publish none of those outputs. No merge, tag, registry publication or Pages
deployment is authorized by this recovery implementation.
