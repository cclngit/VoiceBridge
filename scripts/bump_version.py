#!/usr/bin/env python3
"""Bump VoiceBridge's version in Cargo.toml and Cargo.lock.

Usage:
    python scripts/bump_version.py            # patch: 0.3.0 -> 0.3.1
    python scripts/bump_version.py minor      # 0.3.0 -> 0.4.0
    python scripts/bump_version.py major      # 0.3.0 -> 1.0.0
    python scripts/bump_version.py 0.5.2      # explicit version
    python scripts/bump_version.py --current  # print the current version only

Prints the new version on stdout (nothing else), so CI can capture it.
Used by the `version` job of .github/workflows/ci.yml on every push to main.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CARGO_TOML = ROOT / "Cargo.toml"
CARGO_LOCK = ROOT / "Cargo.lock"
PACKAGE = "VoiceBridge"

# `version = "x.y.z"` inside the [package] table (first table of Cargo.toml).
PKG_VERSION_RE = re.compile(r'(\[package\][^\[]*?\nversion\s*=\s*")(\d+)\.(\d+)\.(\d+)(")', re.S)
# The VoiceBridge entry of Cargo.lock.
LOCK_RE = re.compile(r'(\[\[package\]\]\nname = "' + PACKAGE + r'"\nversion = ")[^"]+(")')


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: pathlib.Path, text: str) -> None:
    # Keep the file's existing line endings (newline="" writes "\n" as-is).
    path.write_text(text, encoding="utf-8", newline="")


def current_version(toml: str) -> tuple[int, int, int]:
    m = PKG_VERSION_RE.search(toml.replace("\r\n", "\n"))
    if not m:
        sys.exit("bump_version: [package] version not found in Cargo.toml")
    return int(m.group(2)), int(m.group(3)), int(m.group(4))


def next_version(cur: tuple[int, int, int], kind: str) -> str:
    major, minor, patch = cur
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    if kind == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if re.fullmatch(r"\d+\.\d+\.\d+", kind):
        return kind
    sys.exit(f"bump_version: unknown bump '{kind}' (patch | minor | major | X.Y.Z)")


def replace_crlf_safe(text: str, pattern: re.Pattern, repl) -> tuple[str, int]:
    crlf = "\r\n" in text
    normalized = text.replace("\r\n", "\n")
    out, n = pattern.subn(repl, normalized, count=1)
    return (out.replace("\n", "\r\n") if crlf else out), n


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "patch"
    toml = read(CARGO_TOML)
    cur = current_version(toml)
    if arg == "--current":
        print("{}.{}.{}".format(*cur))
        return

    new = next_version(cur, arg)
    toml, n = replace_crlf_safe(toml, PKG_VERSION_RE, lambda m: f"{m.group(1)}{new}{m.group(5)}")
    if n != 1:
        sys.exit("bump_version: failed to update Cargo.toml")
    write(CARGO_TOML, toml)

    if CARGO_LOCK.exists():
        lock, n = replace_crlf_safe(read(CARGO_LOCK), LOCK_RE, lambda m: f"{m.group(1)}{new}{m.group(2)}")
        if n != 1:
            sys.exit(f"bump_version: package '{PACKAGE}' not found in Cargo.lock")
        write(CARGO_LOCK, lock)

    print(new)


if __name__ == "__main__":
    main()
