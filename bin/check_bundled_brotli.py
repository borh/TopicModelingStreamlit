#!/usr/bin/env python3
"""Reject Pillow/PyArrow builds whose Brotli decoder resolves below 1.2.1."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

BROTLI_FILE = re.compile(r"libbrotli(?:dec|common)[^/]*\.so[^/]*")
STORE_VERSION = re.compile(r"/brotli-(\d+\.\d+\.\d+)/")
RESOLVED = re.compile(r"=>\s+(/\S+)")


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


def brotli_dependencies(path: Path) -> list[str]:
    result = subprocess.run(
        ["ldd", str(path)], capture_output=True, text=True, check=False
    )
    return [
        line.strip()
        for line in (result.stdout + result.stderr).splitlines()
        if BROTLI_FILE.search(line)
    ]


def audit(root: Path, label: str) -> list[str]:
    failures = []
    bundled = {path.resolve() for path in root.rglob("*") if "libbrotli" in path.name}
    for path in sorted(bundled):
        if path.name.endswith(".so.1.2.0"):
            failures.append(f"{label}: bundled vulnerable {path.name}")

    dependencies = []
    for path in elf_files(root):
        dependencies.extend(brotli_dependencies(path))

    if not dependencies:
        failures.append(f"{label}: no resolved Brotli decoder")
        return failures

    seen = set()
    for line in dependencies:
        match = RESOLVED.search(line)
        resolved = str(Path(match.group(1)).resolve()) if match else line
        if resolved in seen:
            continue
        seen.add(resolved)
        version = STORE_VERSION.search(resolved)
        if (
            version and tuple(map(int, version.group(1).split("."))) < (1, 2, 1)
        ) or resolved.endswith(".so.1.2.0"):
            failures.append(f"{label}: {resolved}")
        else:
            print(f"{label}: {resolved}")
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
                failures.extend(audit(root, source.name))
            elif source.is_dir():
                failures.extend(audit(source, source.name))
            else:
                failures.append(f"{source}: not a wheel or directory")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
