#!/usr/bin/env python3
"""Reject Pillow/PyArrow builds whose Brotli decoder resolves below 1.2.1."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

MINIMUM = (1, 2, 1)
BROTLI = re.compile(r"libbrotli(?:dec|common)")
STORE_COMPONENT = re.compile(r"(?:^|-)brotli-(\d+(?:\.\d+)+)(?:-|$)")
FILENAME_VERSION = re.compile(r"\.so\.(\d+(?:\.\d+)+)$")
RESOLVED = re.compile(r"=>\s+(/\S+)")

Ldd = Callable[[Path], str]


def run_ldd(path: Path) -> str:
    result = subprocess.run(
        ["ldd", str(path)], capture_output=True, text=True, check=False
    )
    return result.stdout + result.stderr


def elf_files(root: Path) -> list[Path]:
    found = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not (path.name.endswith(".so") or ".so." in path.name):
            continue
        try:
            if path.open("rb").read(4) == b"\x7fELF":
                found.append(path)
        except OSError:
            continue
    return found


def brotli_lines(path: Path, ldd: Ldd) -> list[str]:
    return [line.strip() for line in ldd(path).splitlines() if BROTLI.search(line)]


def parse_version(resolved: str) -> tuple[int, ...] | None:
    versions: set[tuple[int, ...]] = set()
    for part in Path(resolved).parts:
        match = STORE_COMPONENT.search(part)
        if match:
            versions.add(tuple(map(int, match.group(1).split("."))))
    match = FILENAME_VERSION.search(Path(resolved).name)
    if match:
        versions.add(tuple(map(int, match.group(1).split("."))))
    if len(versions) != 1:
        return None
    return versions.pop()


def dependency_failure(line: str) -> str | None:
    if "not found" in line:
        return "unresolved Brotli dependency"
    match = RESOLVED.search(line)
    if not match:
        return "Brotli dependency without a path"
    resolved = match.group(1)
    version = parse_version(resolved)
    if version is None:
        return f"unrecognised Brotli provenance: {resolved}"
    if version < MINIMUM:
        return f"Brotli {'.'.join(map(str, version))} below 1.2.1: {resolved}"
    return None


def package_for(path: Path, root: Path) -> str | None:
    parts = path.relative_to(root).parts
    if not parts:
        return None
    if parts[0] == "PIL" or parts[0].startswith("pillow.libs"):
        return "pillow"
    if parts[0] == "pyarrow":
        return "pyarrow"
    return None


def is_anchor(package: str, path: Path, root: Path) -> bool:
    if package == "pillow":
        relative = path.relative_to(root).as_posix()
        return relative.startswith("PIL/_imagingft")
    if package == "pyarrow":
        name = path.name
        return name.startswith(("lib.cpython", "libarrow.so", "libparquet.so"))
    return False


def audit_tree(
    root: Path,
    label: str,
    ldd: Ldd = run_ldd,
    required: str | None = None,
) -> list[str]:
    failures = []
    groups: dict[str, list[Path]] = {}
    for path in elf_files(root):
        package = package_for(path, root)
        if package:
            groups.setdefault(package, []).append(path)

    if required and required not in groups:
        failures.append(f"{label}: no {required} ELF found")
    if not groups:
        failures.append(f"{label}: no Pillow or PyArrow ELF found")
        return failures

    for package, files in sorted(groups.items()):
        anchors = [path for path in files if is_anchor(package, path, root)]
        if not anchors and package == "pyarrow":
            failures.append(f"{label}: {package}: no auditable extension")
        for anchor in anchors:
            lines = brotli_lines(anchor, ldd)
            if not lines:
                failures.append(
                    f"{label}: {package} {anchor.name}: no resolved Brotli decoder"
                )
            for line in lines:
                failure = dependency_failure(line)
                if failure:
                    failures.append(f"{label}: {package} {anchor.name}: {failure}")
        for path in files:
            if "libbrotli" not in path.name:
                continue
            version = parse_version(path.name)
            if version is None or version < MINIMUM:
                failures.append(f"{label}: {package}: bundled {path.name}")
    return failures


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: check_bundled_brotli.py WHEEL_OR_DIR [...]", file=sys.stderr)
        return 2
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for argument in argv:
            source = Path(argument)
            if source.suffix == ".whl":
                root = Path(tmp) / source.stem
                root.mkdir()
                with zipfile.ZipFile(source) as archive:
                    archive.extractall(root)
                required = (
                    "pillow"
                    if source.name.startswith("pillow-")
                    else "pyarrow"
                    if source.name.startswith("pyarrow-")
                    else None
                )
                failures.extend(audit_tree(root, source.name, required=required))
            elif source.is_dir():
                failures.extend(audit_tree(source, source.name))
            else:
                failures.append(f"{source}: not a wheel or directory")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
