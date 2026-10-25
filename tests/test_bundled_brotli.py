import runpy
from pathlib import Path

import pytest

CHECKER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "bin/check_bundled_brotli.py")
)
parse_version = CHECKER["parse_version"]
dependency_failure = CHECKER["dependency_failure"]
audit_tree = CHECKER["audit_tree"]

HASH = "a" * 32


def nix_path(version: str, filename: str) -> str:
    return f"/nix/store/{HASH}-brotli-{version}-lib/lib/{filename}"


def nix_line(version: str, filename: str) -> str:
    return f"libbrotlidec.so.1 => {nix_path(version, filename)} (0x0000)"


FIXED = nix_line("1.2.1", "libbrotlidec.so.1.2.1")


def _elf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF")


@pytest.mark.parametrize(
    "resolved, expected",
    [
        (nix_path("1.0.9", "libbrotlidec.so.1.0.9"), (1, 0, 9)),
        (nix_path("1.2.0", "libbrotlidec.so.1.2.0"), (1, 2, 0)),
        (nix_path("1.2.1", "libbrotlidec.so.1.2.1"), (1, 2, 1)),
        (nix_path("1.3.0", "libbrotlidec.so.1.3.0"), (1, 3, 0)),
        ("/usr/lib/libbrotlidec.so.1.2.1", (1, 2, 1)),
        ("/usr/lib/libbrotlidec.so.1", None),
        (f"/nix/store/{HASH}-brotli-lib/lib/libbrotlidec.so.1", None),
        (nix_path("1.2.0", "libbrotlidec.so.1.2.1"), None),
    ],
)
def test_parse_version(resolved: str, expected: tuple[int, ...] | None) -> None:
    assert parse_version(resolved) == expected


@pytest.mark.parametrize(
    "line, rejected",
    [
        (nix_line("1.0.9", "libbrotlidec.so.1.0.9"), True),
        ("libbrotlidec.so.1 => not found", True),
        (nix_line("1.2.0", "libbrotlidec.so.1.2.0"), True),
        (FIXED, False),
        (nix_line("1.3.0", "libbrotlidec.so.1.3.0"), False),
        ("libbrotlidec.so.1 => /usr/lib/libbrotlidec.so.1", True),
        ("libbrotlidec.so.1 => /usr/lib/libbrotlidec.so.1.2.1", False),
    ],
)
def test_dependency_failure(line: str, rejected: bool) -> None:
    assert (dependency_failure(line) is not None) is rejected


def test_bundled_old_decoder_is_rejected(tmp_path: Path) -> None:
    _elf(tmp_path / "PIL/_imagingft.cpython-313.so")
    _elf(tmp_path / "pillow.libs/libbrotlidec-7e5462ba.so.1.2.0")
    line = (
        "libbrotlidec-7e5462ba.so.1.2.0 => "
        "/tmp/x/pillow.libs/libbrotlidec-7e5462ba.so.1.2.0 (0x0000)"
    )
    assert audit_tree(tmp_path, "fixture", ldd=lambda path: line)


def test_static_arrow_is_not_masked_by_pillow_decoder(tmp_path: Path) -> None:
    _elf(tmp_path / "PIL/_imagingft.cpython-313.so")
    _elf(tmp_path / "pyarrow/lib.cpython-313.so")
    _elf(tmp_path / "pyarrow/libarrow.so.2400")
    traces = {
        "_imagingft.cpython-313.so": FIXED,
        "lib.cpython-313.so": "libarrow.so.2400 => /tmp/x/pyarrow/libarrow.so.2400",
        "libarrow.so.2400": "libstdc++.so.6 => /usr/lib/libstdc++.so.6",
    }
    failures = audit_tree(tmp_path, "fixture", ldd=lambda path: traces[path.name])
    assert any(
        "pyarrow" in failure and "no resolved Brotli decoder" in failure
        for failure in failures
    )


def test_dynamic_arrow_decoder_passes(tmp_path: Path) -> None:
    _elf(tmp_path / "pyarrow/lib.cpython-313.so")
    _elf(tmp_path / "pyarrow/libarrow.so.2400")
    failures = audit_tree(tmp_path, "fixture", ldd=lambda path: FIXED)
    assert failures == []


def test_required_package_must_be_present(tmp_path: Path) -> None:
    _elf(tmp_path / "PIL/_imagingft.cpython-313.so")
    failures = audit_tree(
        tmp_path, "fixture", ldd=lambda path: FIXED, required="pyarrow"
    )
    assert any("no pyarrow ELF" in failure for failure in failures)
