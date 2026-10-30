#!/usr/bin/env python3
"""compare_images.py — compare what two image builds produced.

Give it the manifest folders Packer saved for two builds of the same target
(normally /tmp/image_manifests, copied after each build), typically one built
with bash and one with the Python CLI:

    python3 tools/compare_images.py bash-a100/ python-a100/

Each folder may hold:

    build-evidence.tgz           every build (packages, holds, repositories,
                                 service states, dkms status, component versions)
    component-versions.json      every build
    trivy-report-rootfs.json     every build (vulnerabilities found)
    image-fingerprint.json       with CAPTURE_IMAGE_FINGERPRINT=true (config file
                                 hashes, /opt layout, modulefiles, enabled units,
                                 pip packages; see tools/image_fingerprint.py)

It compares every section both folders have and prints what differs: things
only in the first build, only in the second, and things in both that differ.
Expected noise is ignored (ImageVersion, timestamps, journal/dmesg). Exit
status: 0 if nothing differs, 1 if something does, 2 if the inputs are
unusable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path

# component_versions.txt keys that always differ between two builds.
IGNORED_COMPONENTS = {"ImageVersion"}


# --- reading the inputs ------------------------------------------------------

def evidence_files(tgz: Path) -> dict[str, str]:
    """{path inside the run folder: text} for build-evidence.tgz.

    The archive holds one folder per run (<run name>/build/packages.tsv ...);
    the run name is dropped so two builds' paths line up.
    """
    files = {}
    with tarfile.open(tgz) as archive:
        for member in archive.getmembers():
            if member.isfile():
                inner = member.name.split("/", 1)[-1]
                data = archive.extractfile(member).read()
                files[inner] = data.decode("utf-8", errors="replace")
    return files


def load_build(folder: Path) -> dict:
    """Everything comparable in one build's manifest folder."""
    build = {}
    evidence = folder / "build-evidence.tgz"
    if evidence.is_file():
        build["evidence"] = evidence_files(evidence)
    for name, key in (("component-versions.json", "components"),
                      ("trivy-report-rootfs.json", "trivy"),
                      ("image-fingerprint.json", "fingerprint")):
        path = folder / name
        if path.is_file():
            build[key] = json.loads(path.read_text(encoding="utf-8"))
    return build


# --- turning each section into a comparable dict -----------------------------

def parse_packages(tsv: str) -> dict[str, str]:
    """packages.tsv (name<TAB>version) -> {name: version}; repeats joined (kernels)."""
    packages: dict[str, list[str]] = {}
    for line in tsv.splitlines():
        if "\t" in line:
            name, version = line.split("\t", 1)
            packages.setdefault(name, []).append(version)
    return {name: ", ".join(sorted(versions)) for name, versions in packages.items()}


def parse_services(text: str) -> dict[str, str]:
    """`systemctl list-units --type=service --all` -> {unit: "load active sub"}."""
    services = {}
    for line in text.splitlines():
        fields = line.replace("●", " ").split()
        if len(fields) >= 4 and fields[0].endswith(".service"):
            services[fields[0]] = " ".join(fields[1:4])
    return services


def as_lines(text: str) -> dict[str, str]:
    """Any line-per-item text -> {line: ""}, so it compares as a set."""
    return {line.strip(): "" for line in text.splitlines() if line.strip()}


def trivy_findings(report: dict) -> dict[str, str]:
    """{"CVE-x pkg": severity} for every vulnerability in a Trivy JSON report."""
    findings = {}
    for result in report.get("Results") or []:
        for vuln in result.get("Vulnerabilities") or []:
            key = f"{vuln.get('VulnerabilityID')} {vuln.get('PkgName')}"
            findings[key] = vuln.get("Severity", "")
    return findings


def fingerprint_sections(fp: dict) -> dict[str, dict[str, str]]:
    """image-fingerprint.json -> {section: {item: value}}."""
    def files(tree):
        return {path: " ".join(f"{k}={v}" for k, v in sorted(entry.items()))
                for path, entry in tree.items()}
    sections = {
        "config files": files(fp.get("config_files", {})),
        "modulefiles": files(fp.get("modulefiles", {})),
        "enabled units": {unit: "" for unit in fp.get("enabled_units", [])},
        "pip packages": {pkg: "" for pkg in fp.get("pip_packages", [])},
    }
    for root, entries in (fp.get("layout") or {}).items():
        sections[f"layout of {root}"] = entries
    return sections


def sections(build: dict) -> dict[str, dict[str, str]]:
    """Every comparable section of one build, as {section: {item: value}}."""
    out = {}
    if "components" in build:
        out["component versions"] = {k: str(v) for k, v in build["components"].items()
                                     if k not in IGNORED_COMPONENTS}
    evidence = build.get("evidence", {})
    readers = {"build/packages.tsv": ("packages", parse_packages),
               "build/package-holds.txt": ("package holds", as_lines),
               "build/repositories.txt": ("repositories", as_lines),
               "build/dkms.txt": ("dkms modules", as_lines),
               "build/service-state.txt": ("services", parse_services)}
    for path, (name, reader) in readers.items():
        if path in evidence:
            out[name] = reader(evidence[path])
    if "trivy" in build:
        out["vulnerabilities (Trivy)"] = trivy_findings(build["trivy"])
    if "fingerprint" in build:
        out.update(fingerprint_sections(build["fingerprint"]))
    return out


# --- comparing ---------------------------------------------------------------

def diff(a: dict[str, str], b: dict[str, str], ignore: re.Pattern | None = None):
    """(only in a, only in b, {item: (a value, b value)} that differ)."""
    keep = (lambda item: not ignore.search(item)) if ignore else (lambda item: True)
    only_a = sorted(item for item in a.keys() - b.keys() if keep(item))
    only_b = sorted(item for item in b.keys() - a.keys() if keep(item))
    changed = {item: (a[item], b[item]) for item in sorted(a.keys() & b.keys())
               if a[item] != b[item] and keep(item)}
    return only_a, only_b, changed


def report(a: dict, b: dict, labels: tuple[str, str], ignore=None, out=sys.stdout) -> int:
    """Print the differences between two builds; the number of differing items."""
    sa, sb = sections(a), sections(b)
    total = 0
    for name in sorted(sa.keys() | sb.keys()):
        if name not in sa or name not in sb:
            missing = labels[0] if name not in sa else labels[1]
            print(f"\n== {name}: not captured for {missing}; skipped", file=out)
            continue
        only_a, only_b, changed = diff(sa[name], sb[name], ignore)
        count = len(only_a) + len(only_b) + len(changed)
        total += count
        if not count:
            print(f"== {name}: same ({len(sa[name])} items)", file=out)
            continue
        print(f"\n== {name}: {count} difference(s)", file=out)
        for item in only_a:
            print(f"   only in {labels[0]}: {item} {sa[name][item]}".rstrip(), file=out)
        for item in only_b:
            print(f"   only in {labels[1]}: {item} {sb[name][item]}".rstrip(), file=out)
        for item, (va, vb) in changed.items():
            print(f"   differs: {item}\n      {labels[0]}: {va}\n      {labels[1]}: {vb}", file=out)
    print(f"\n{total} difference(s) in total", file=out)
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare what two image builds produced.")
    parser.add_argument("first", type=Path, help="manifest folder of the first build")
    parser.add_argument("second", type=Path, help="manifest folder of the second build")
    parser.add_argument("--labels", nargs=2, default=["bash", "python"],
                        help="names for the two builds in the report (default: bash python)")
    parser.add_argument("--ignore", help="regular expression: skip items that match")
    args = parser.parse_args(argv)

    builds = []
    for folder in (args.first, args.second):
        if not folder.is_dir():
            print(f"error: {folder} is not a folder", file=sys.stderr)
            return 2
        build = load_build(folder)
        if not build:
            print(f"error: nothing to compare in {folder}", file=sys.stderr)
            return 2
        builds.append(build)
    ignore = re.compile(args.ignore) if args.ignore else None
    return 1 if report(builds[0], builds[1], tuple(args.labels), ignore) else 0


if __name__ == "__main__":
    sys.exit(main())
