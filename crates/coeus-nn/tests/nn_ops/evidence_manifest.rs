use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path},
};

const HEADER: &str = "family\tcriterion_status\tcriterion_path\tcriterion_symbol\trust_status\trust_path\trust_symbol\tpython_status\tpython_path\tpython_symbol\tnote";
const MANIFEST: &str = include_str!("../../benches/nn_bench/evidence.tsv");
const NN_ROOT: &str = include_str!("../../src/lib.rs");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Status {
    Present,
    Partial,
    Missing,
    Inapplicable,
}

impl Status {
    fn parse(value: &str) -> Self {
        match value {
            "present" => Self::Present,
            "partial" => Self::Partial,
            "missing" => Self::Missing,
            "inapplicable" => Self::Inapplicable,
            invalid => panic!("invalid evidence status: {invalid}"),
        }
    }

    const fn requires_locator(self) -> bool {
        matches!(self, Self::Present | Self::Partial)
    }
}

#[derive(Debug)]
struct Evidence<'a> {
    status: Status,
    path: &'a str,
    symbol: &'a str,
}

#[derive(Debug)]
struct FamilyEvidence<'a> {
    criterion: Evidence<'a>,
    rust: Evidence<'a>,
    python: Evidence<'a>,
    note: &'a str,
}

fn parse_evidence<'a>(fields: &[&'a str], offset: usize) -> Evidence<'a> {
    Evidence {
        status: Status::parse(fields[offset]),
        path: fields[offset + 1],
        symbol: fields[offset + 2],
    }
}

fn parse_manifest() -> BTreeMap<&'static str, FamilyEvidence<'static>> {
    let mut lines = MANIFEST.lines();
    assert_eq!(
        lines.next(),
        Some(HEADER),
        "evidence manifest header drifted"
    );

    let mut manifest = BTreeMap::new();
    for (index, line) in lines.enumerate() {
        let fields = line.split('\t').collect::<Vec<_>>();
        assert_eq!(
            fields.len(),
            11,
            "manifest line {} must contain 11 tab-separated fields",
            index + 2
        );
        let family = fields[0];
        assert!(!family.is_empty(), "manifest family cannot be empty");
        let row = FamilyEvidence {
            criterion: parse_evidence(&fields, 1),
            rust: parse_evidence(&fields, 4),
            python: parse_evidence(&fields, 7),
            note: fields[10],
        };
        assert!(
            manifest.insert(family, row).is_none(),
            "duplicate manifest family: {family}"
        );
    }
    manifest
}

fn public_families() -> BTreeSet<&'static str> {
    NN_ROOT
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("pub mod ")
                .and_then(|module| module.strip_suffix(';'))
        })
        .collect()
}

const fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn contains_symbol(source: &str, symbol: &str) -> bool {
    source.match_indices(symbol).any(|(start, _)| {
        let end = start + symbol.len();
        let bytes = source.as_bytes();
        let left_is_boundary = start == 0 || !is_identifier_byte(bytes[start - 1]);
        let right_is_boundary = end == bytes.len() || !is_identifier_byte(bytes[end]);
        left_is_boundary && right_is_boundary
    })
}

fn validate_evidence(workspace_root: &Path, family: &str, lane: &str, evidence: &Evidence<'_>) {
    if !evidence.status.requires_locator() {
        assert!(
            evidence.path.is_empty() && evidence.symbol.is_empty(),
            "{family} {lane} {:#?} must not carry a locator",
            evidence.status
        );
        return;
    }

    assert!(
        !evidence.path.is_empty() && !evidence.symbol.is_empty(),
        "{family} {lane} {:#?} requires a path and symbol",
        evidence.status
    );
    let relative = Path::new(evidence.path);
    assert!(
        relative
            .components()
            .all(|component| matches!(component, Component::Normal(_))),
        "{family} {lane} path must be workspace-relative: {}",
        evidence.path
    );
    let source = fs::read_to_string(workspace_root.join(relative)).unwrap_or_else(|error| {
        panic!(
            "{family} {lane} evidence file {} is unreadable: {error}",
            evidence.path
        )
    });
    assert!(
        contains_symbol(&source, evidence.symbol),
        "{family} {lane} symbol {} is absent from {}",
        evidence.symbol,
        evidence.path
    );
}

#[test]
fn symbol_matching_requires_identifier_boundaries() {
    assert!(contains_symbol(
        "fn bench_linear_forward() {}",
        "bench_linear_forward"
    ));
    assert!(!contains_symbol(
        "fn bench_linear_forward_backward() {}",
        "bench_linear_forward"
    ));
}

#[test]
fn evidence_manifest_covers_every_public_nn_family() {
    let manifest = parse_manifest();
    assert_eq!(
        manifest.keys().copied().collect::<BTreeSet<_>>(),
        public_families(),
        "the evidence manifest must match the public Coeus-NN family inventory"
    );

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = crate_root
        .parent()
        .and_then(Path::parent)
        .expect("invariant: coeus-nn is nested under the workspace crates directory");

    for (family, row) in manifest {
        assert!(!row.note.is_empty(), "{family} requires an evidence note");
        validate_evidence(workspace_root, family, "criterion", &row.criterion);
        validate_evidence(workspace_root, family, "rust", &row.rust);
        validate_evidence(workspace_root, family, "python", &row.python);
    }
}
