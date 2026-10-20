# Repaired native Leto dependency renewal

[patch] The unpublished Coeus 0.13 cohort requires both Leto packages at
`>=0.46.3, <0.48.0`, excluding the published 0.46.2 finite-input CSR NaN
regression. The selected accepted candidate from Leto PR #376 is
b3d744a91f331061d0e3eb40b5126cc1fa7782c2,
tree 63be55806f50a253718644a9d98e65ea50b4d2fd. Independent receipt SHA256
2c7570ede0b949ec109df7be01968aec7554fef7e676ed4d88cbe29c57580d1c
accepts its numerical contract and matched native guard-cost performance.
Absolute application performance and the Hephaestus 0.25 consumer cohort
remain unverified by that receipt. The superseded 3ebe candidate is not the
input for final reconciliation.

Actual merged Coeus PRs #613 and #614 remain ancestors. The strict Moirai
floor, registry renewal, private 0.13 requirements, numerical/NN repairs and
native Python linkage arrangement are preserved. The exact #614 three-file
FusionScalar migration is retained; see the
[0.25 renewal](hephaestus-025-renewal.md).

Both registry 0.46.3 archives verify against their index checksums and all
519 packaged Rust files/original manifests match the accepted source. The
actual normal merge is 5051cd9577650a5712aa6a7effcd8955c9d8bcc7 with the exact
accepted tree. Ordinary Cargo renewal selects that default plus Hephaestus
0.25 at 36965bf2d775f09f4e5a76daae57d2a64b145e1c; all 235 registry and
36 other Git records, and all sixteen local 0.13 entries, are preserved.
Ordinary package/API checks and this material consumer graph's own exact-head
verification remain required. The separate Leto 0.47 stored-order policy and Coeus PyO3
0.23-to-common-0.30 migration remain unresolved. #497/#500 owners, first
publication, physical GPU/platform and Harmonia performance holds remain.
The denied Coeus push stays paused. Coeus Miri is deferred and unverified.
