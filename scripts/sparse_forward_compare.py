#!/usr/bin/env python3
"""Compare real forward kernels with byte-identical main sources on one CPU graph.

The temporary reference crate contains the complete original sparse module,
without rewriting its mathematics, validation, storage, or backend calls.
The existing benchmark supplies all data, exact oracles, calibration and paired
sampling. This component instrument does not certify workspace/package resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile


def run(arguments: list[str], directory: Path) -> bytes:
    process = subprocess.Popen(arguments, cwd=directory, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    if process.returncode:
        raise RuntimeError(output.decode(errors="replace"))
    return output


def fingerprint(root: Path) -> dict[str, str]:
    result = {}
    for name in ("coeus-core", "coeus-leto", "coeus-tensor", "coeus-sparse", "coeus-ops"):
        package = root / "crates" / name
        for path in sorted((package / "src").rglob("*.rs")) + [package / "Cargo.toml"]:
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ("Cargo.toml", "crates/coeus-ops/benches/sparse_recovery.rs",
                     "scripts/sparse_forward_compare.py"):
        result[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Exact main commit for the reference")
    parser.add_argument("--review-lock", type=Path, required=True,
                        help="A validated lock for this exact CPU provider graph")
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON receipt; log uses .log")
    parser.add_argument("--build-only", action="store_true")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = run(["git", "rev-parse", "--verify", options.base + "^{commit}"], root).decode().strip()
    before = fingerprint(root)
    reference_hashes = {}
    with tempfile.TemporaryDirectory(prefix="csr-forward-comparison-") as temporary:
        work = Path(temporary)
        reference = work / "reference"
        (reference / "src" / "sparse").mkdir(parents=True)
        files = ["ptr.rs", *("sparse/" + name + ".rs" for name in
                             ("mod", "ops", "validation", "conversions"))]
        for relative in files:
            repository_path = "crates/coeus-ops/src/" + relative
            source = run(["git", "show", base + ":" + repository_path], root)
            (reference / "src" / relative).write_bytes(source)
            reference_hashes[repository_path] = hashlib.sha256(source).hexdigest()
        (reference / "src" / "lib.rs").write_text("mod ptr;\npub mod sparse;\n")
        reference_manifest = ('[package]\nname = "sparse-main-reference"\n'
                              'version = "0.1.0"\nedition = "2021"\n[dependencies]\n')
        for name in ("coeus-core", "coeus-tensor", "coeus-sparse"):
            reference_manifest += f'{name} = {{ path = {json.dumps(str(root / "crates" / name))} }}\n'
        reference_manifest += 'smallvec = "1.13"\n'
        (reference / "Cargo.toml").write_text(reference_manifest)

        manifest = ('[package]\nname = "csr-forward-comparison"\nversion = "0.1.0"\n'
                    'edition = "2021"\n[workspace]\nmembers = ["reference"]\n'
                    'default-members = ["."]\n[dependencies]\n'
                    'sparse-main-reference = { path = "reference" }\n')
        for name in ("coeus-core", "coeus-leto", "coeus-tensor", "coeus-sparse", "coeus-ops"):
            manifest += f'{name} = {{ path = {json.dumps(str(root / "crates" / name))} }}\n'
        manifest += ('leto = { git = "https://github.com/ryancinsight/leto.git", version = "0.47.0" }\n'
                     'leto-ops = { git = "https://github.com/ryancinsight/leto.git", version = "0.47.0" }\n'
                     'tracing = { version = "0.1.44", default-features = false, features = ["std"] }\n'
                     'tracing-subscriber = "0.3.23"\n'
                     '[[bench]]\nname = "sparse_compare"\nharness = false\n'
                     'path = "sparse_compare.rs"\n')
        current_manifest = (root / "Cargo.toml").read_text()
        manifest += current_manifest[current_manifest.index("[profile.bench]"):
                                     current_manifest.index("[workspace.lints.clippy]")]
        (work / "Cargo.toml").write_text(manifest)
        (work / "Cargo.lock").write_bytes(options.review_lock.read_bytes())
        benchmark = (root / "crates/coeus-ops/benches/sparse_recovery.rs").read_text()
        # The standalone instrument records its original audit base. This
        # adapter extracts a caller-selected main revision, so label it exactly.
        original_base = 'base = "33f6f1f0596d3e3d914d11e33b79fe48ba16da4c"'
        assert benchmark.count(original_base) == 1, "benchmark base metadata changed"
        benchmark = benchmark.replace(original_base, f'base = "{base}"')
        start = benchmark.index("fn forward_comparison<")
        end = benchmark.index("fn rounding_counterexample", start)
        # Only the comparison adapter changes. The fixture, oracle, workloads,
        # calibration and 21 counterbalanced sample pairs remain authoritative.
        adapter = r'''
fn forward_comparison<B: Backend>(fixture: &Fixture, backend: &B, backend_name: &str)
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    let (a, dense) = fixture.tensor_inputs(backend);
    let baseline = || {
        if fixture.width == 1 {
            sparse_main_reference::sparse::spmv(&a, &dense, backend).expect("main SpMV")
        } else {
            sparse_main_reference::sparse::spmm(&a, &dense, backend).expect("main SpMM")
        }
    };
    assert_eq!(baseline().as_slice(), fixture.expected);
    assert_eq!(forward_product(&a, &dense, fixture.width, backend).as_slice(), fixture.expected);
    paired(&format!("main-forward/{backend_name}/{}x{}/rhs{}", fixture.rows, fixture.shared, fixture.width),
        || { black_box(baseline()); },
        || { black_box(forward_product(&a, &dense, fixture.width, backend)); });
    assert_eq!(baseline().as_slice(), fixture.expected);
    assert_eq!(forward_product(&a, &dense, fixture.width, backend).as_slice(), fixture.expected);
}

'''
        (work / "sparse_compare.rs").write_text(benchmark[:start] + adapter + benchmark[end:])
        # Cargo prunes the imported receipt and adds only the real comparison
        # packages. The subsequent locked command is the measured source graph.
        command = ["cargo", "+1.97.0", "bench", "--offline", "--manifest-path",
                   str(work / "Cargo.toml"), "--target-dir", str(options.target_dir),
                   "-p", "csr-forward-comparison", "--bench", "sparse_compare", "--no-run"]
        build_log = run(command, work)
        lock = (work / "Cargo.lock").read_bytes()
        command.insert(3, "--locked")
        if not options.build_only:
            command.remove("--no-run")
            log = run(command, work)
        else:
            log = build_log
        assert before == fingerprint(root), "candidate source changes during comparison"
        receipt = {"base": base, "candidate_sources": before,
                   "reference_sources": reference_hashes,
                   "lock_sha256": hashlib.sha256(lock).hexdigest(),
                   "command": command, "build_only": options.build_only,
                   "log_sha256": hashlib.sha256(log).hexdigest(),
                   "scope": "same-graph CPU comparison; full workspace/availability gates are separate"}
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(receipt, indent=2) + "\n")
        options.output.with_suffix(".log").write_bytes(log)
        print(options.output)
        print(log.decode(errors="replace")[-12000:])


if __name__ == "__main__":
    main()
