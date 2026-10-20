#!/usr/bin/env python3
"""Exact locked Coeus ROCm API pair; compile-only, with retained JSON evidence.

Uses the committed-snapshot/--locked/prebuilt-JSON contract of Atlas
atlas_semver_locked.py at 7a5a1036973b81dee0e20321942ce0c7e2bf9f9b.
The wrapper binds the approved pair and retains artifacts; it never invokes
the release workflow, a registry baseline, device tests, or provider API pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tomllib

BASE = "5d5979bcd6d33d8965133c619ebf2427cba136a4"
HEPHAESTUS = "https://github.com/ryancinsight/hephaestus.git"
APOLLO = "https://github.com/ryancinsight/apollo.git"
PROVIDERS = {"hephaestus-" + n for n in ("core", "cuda", "metal", "rocm", "wgpu")}
FEATURES = {"default": [], "rocm": ["hephaestus-rocm/rocm"]}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], timeout=60)


def policy(path: Path, expected: str) -> dict:
    raw = path.read_bytes()
    require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "policy digest must be exact SHA256")
    require(digest(raw) == expected, "input policy checksum mismatch")
    data = json.loads(raw)
    require(data["schema"] == "coeus-rocm-locked-api-input/v1", "unsupported policy schema")
    require(data["package"] == "coeus-rocm", "only the consumer ROCm package is permitted")
    require(data["toolchain"] == "1.97.0" and data["checker_version"] == "0.51.0", "tool version drift")
    require(data["features"] == ["default", "rocm"], "all stable ROCm features must remain selected")
    require(data["baseline"]["head"] == BASE, "accepted consumer baseline cannot be replaced")
    require(data["baseline"]["version"] == "0.13.0", "wrong baseline consumer version")
    require(data["candidate"]["version"] == "0.14.0", "wrong candidate consumer version")
    require(data["baseline"]["provider_floor"] == "0.25.0" and data["baseline"]["apollo_floor"] == "0.29.3", "accepted baseline floors cannot change")
    require(data["candidate"]["provider_floor"] == "0.27.1", "candidate repaired provider floor must remain ^0.27.1")
    require(data["candidate"]["apollo_floor"] == "0.30.0", "candidate FFT floor must remain ^0.30.0")
    require(data["candidate"]["provider_version"] == "0.27.1" and data["candidate"]["apollo_version"] == "0.30.0", "wrong candidate provider package versions")
    for arm in ("baseline", "candidate"):
        for key in ("head", "tree", "provider_head", "apollo_head"):
            require(re.fullmatch(r"[0-9a-f]{40}", data[arm][key]) is not None, f"invalid {arm} {key}")
    return data


def snapshot(repo: Path, name: str, expected: dict, out: Path) -> dict:
    head = expected["head"]
    require(git(repo, "rev-parse", head + "^{tree}").decode().strip() == expected["tree"], f"{name} tree drift")
    root = tomllib.loads(git(repo, "show", head + ":Cargo.toml").decode())
    require(root["workspace"]["package"]["version"] == expected["version"], f"{name} version drift")
    member = tomllib.loads(git(repo, "show", head + ":crates/coeus-rocm/Cargo.toml").decode())
    require(member["features"] == FEATURES, f"{name} ROCm feature inventory drift")
    deps = root["workspace"]["dependencies"]
    floor = "0.25.0" if name == "baseline" else "0.27.1"
    fft_floor = "0.29.3" if name == "baseline" else "0.30.0"
    for dep in PROVIDERS:
        require(deps[dep]["git"] == HEPHAESTUS and deps[dep]["version"] == floor, f"{name} provider floor/source drift")
        require(not {"rev", "branch", "tag", "path"}.intersection(deps[dep]), f"{name} provider selector override")
    require(deps["apollo-fft"]["git"] == APOLLO and deps["apollo-fft"]["version"] == fft_floor, f"{name} FFT floor/source drift")
    raw_lock = git(repo, "show", head + ":Cargo.lock")
    require(digest(raw_lock) == expected["lock_sha256"], f"{name} lock checksum drift")
    packages = tomllib.loads(raw_lock.decode())["package"]
    providers = [p for p in packages if p["name"].startswith("hephaestus-")]
    apollo = [p for p in packages if p["name"].startswith("apollo-")]
    require({p["name"] for p in providers} == PROVIDERS, f"{name} provider package identity drift")
    require({p["name"] for p in apollo} == {"apollo-fft", "apollo-fft-macros", "apollo-leto-interop"}, f"{name} Apollo package identity drift")
    require(all(p["source"] == "git+" + HEPHAESTUS + "#" + expected["provider_head"] and p["version"] == expected["provider_version"] for p in providers), f"{name} provider lock source drift")
    require(all(p["source"] == "git+" + APOLLO + "#" + expected["apollo_head"] for p in apollo), f"{name} Apollo lock source drift")
    require(next(p["version"] for p in apollo if p["name"] == "apollo-fft") == expected["apollo_version"], f"{name} FFT lock version drift")
    require(not any(p.get("source") is None and not p["name"].startswith("coeus-") for p in packages), f"{name} source-less external lock entry")
    destination = out / "work" / name
    require(not destination.exists(), "output must be a fresh directory")
    destination.mkdir(parents=True)
    archive = git(repo, "archive", "--format=tar", head)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(destination, filter="data")
    return {"name": name, "root": str(destination), "head": head, "tree": expected["tree"], "lock_sha256": digest(raw_lock), "provider_head": expected["provider_head"], "apollo_head": expected["apollo_head"]}


def run(command: list[str], cwd: Path, env: dict[str, str], out: Path, label: str, report: dict) -> str:
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / f"{label}.stdout.log").open("w") as stdout, (logs / f"{label}.stderr.log").open("w") as stderr:
        try:
            completed = subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=1200)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = 124
    report["commands"].append({"command": command, "cwd": str(cwd), "exit_code": code, "label": label, "timeout_seconds": 1200})
    require(code == 0, f"{label} failed with status {code}; see raw logs")
    return (logs / f"{label}.stdout.log").read_text()


def execute(repo: Path, data: dict, arms: list[dict], cargo_home: Path, checker: Path, out: Path, report: dict) -> None:
    require(platform.system() == "Linux", "full-feature ROCm API execution requires the approved Linux SDK environment")
    require((Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "include/hip/hip_runtime.h").is_file(), "ROCm SDK headers unavailable")
    for name in ("CARGO_BUILD_TARGET", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTDOCFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"):
        require(not os.environ.get(name), f"unexpected build override {name}")
    require(cargo_home.is_absolute(), "Cargo home must be an explicit isolated directory")
    for filename in ("config", "config.toml", "credentials", "credentials.toml"):
        require(not (cargo_home / filename).exists(), "Cargo home must contain no settings, overlays or credentials")
    env = dict(os.environ, CARGO_HOME=str(cargo_home), CARGO_BUILD_JOBS="2", CARGO_INCREMENTAL="0", CARGO_TERM_COLOR="never", RUSTUP_TOOLCHAIN="1.97.0")
    version = run([str(checker), "semver-checks", "--version"], repo, env, out, "checker-version", report)
    require(version.strip() == "cargo-semver-checks 0.51.0", "API checker version mismatch")
    run(["cargo", "+1.97.0", "--version"], repo, env, out, "cargo-version", report)
    docs = []
    for arm in arms:
        name, root = arm["name"], Path(arm["root"])
        run(["cargo", "+1.97.0", "fetch", "--locked"], root, env, out, name + "-fetch", report)
        metadata = json.loads(run(["cargo", "+1.97.0", "metadata", "--locked", "--offline", "--all-features", "--format-version", "1"], root, env, out, name + "-metadata", report))
        for package in ("aequitas", "eunomia", "leto", "leto-ops", "hephaestus-core", "hephaestus-rocm", "apollo-fft"):
            require(len([p for p in metadata["packages"] if p["name"] == package]) == 1, f"{name} duplicate type-bearing identity {package}")
        require(digest((root / "Cargo.lock").read_bytes()) == arm["lock_sha256"], f"{name} lock mutated while hydrating")
        target = out / "work" / (name + "-target")
        arm_env = dict(env, CARGO_TARGET_DIR=str(target), RUSTC_BOOTSTRAP="1")
        command = ["cargo", "+1.97.0", "rustdoc", "--locked", "--offline", "--package", "coeus-rocm", "--lib", "--all-features", "--", "-Z", "unstable-options", "--output-format=json", "--document-private-items", "--document-hidden-items"]
        run(command, root, arm_env, out, name + "-rustdoc", report)
        doc = target / "doc/coeus_rocm.json"
        document = json.loads(doc.read_text())
        require(document["crate_version"] == data[name]["version"] and document["includes_private"], f"{name} wrong/incomplete consumer Rustdoc")
        require(document["index"][str(document["root"])]["name"] == "coeus_rocm", f"{name} wrong crate API JSON")
        require(digest((root / "Cargo.lock").read_bytes()) == arm["lock_sha256"], f"{name} lock mutated during rustdoc")
        retained = out / (name + "-coeus-rocm.json")
        shutil.copyfile(doc, retained)
        arm["rustdoc_sha256"] = digest(retained.read_bytes())
        docs.append(retained)
    run([str(checker), "semver-checks", "check-release", "--baseline-rustdoc", str(docs[0]), "--current-rustdoc", str(docs[1]), "--release-type", "patch", "--color", "never"], repo, env, out, "consumer-structural-comparison", report)
    report["completed"] = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cargo-home", type=Path)
    parser.add_argument("--checker", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.exists(), "output directory must not preexist")
    out.mkdir(parents=True)
    report = {"schema": "coeus-rocm-locked-api-result/v1", "completed": False, "preflight_only": args.preflight_only, "policy_sha256": args.policy_sha256, "commands": [], "scope": "Exact full-feature consumer source API only; no provider API package comparisons, registry baseline, publication, device tests or performance claim."}
    try:
        repo = args.repo.resolve()
        require(not git(repo, "status", "--porcelain=v1").strip(), "source repository must be clean")
        data = policy(args.policy, args.policy_sha256)
        shutil.copyfile(args.policy, out / "input-policy.json")
        report["snapshots"] = [snapshot(repo, name, data[name], out) for name in ("baseline", "candidate")]
        report["preflight_passed"] = True
        if not args.preflight_only:
            require(args.cargo_home is not None and args.checker is not None, "execution needs isolated Cargo home and exact checker")
            execute(repo, data, report["snapshots"], args.cargo_home.resolve(), args.checker.resolve(), out, report)
        return 0
    except Exception as error:
        report["error"] = str(error)
        return 1
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
