"""
Release pipeline steps for .github/workflows/release.yaml. 

    python3 .github/release.py resolve-symlinks
    python3 .github/release.py set-version <path/to/pyproject.toml> [...]
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

TAG_PATTERN = re.compile(r"v([0-9]+\.[0-9]+\.[0-9]+)")
GIT_SYMLINK_MODE = "120000"


def resolve_symlinks(args):
    """Replaces every symlink tracked by git with a copy of its target."""
    entries = subprocess.run(
        ["git", "ls-files", "-s", "-z"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")

    for entry in filter(None, entries):
        info, path = entry.split("\t", 1)
        if info.split()[0] == GIT_SYMLINK_MODE:
            link = Path(path)
            target = link.resolve()
            link.unlink()
            if target.is_dir():
                shutil.copytree(target, link)
            else:
                shutil.copy2(target, link)
            print(f"Resolved symlink {link} -> {target}")


def get_version_from_tag():
    tags = subprocess.run(
        ["git", "tag", "--points-at", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    versions = [m.group(1) for m in map(TAG_PATTERN.fullmatch, tags) if m]
    assert len(versions) == 1
    return versions[0]


def set_version(args):
    """Sets the version in each pyproject.toml from the git tag of the form
    vX.Y.Z on HEAD."""
    import tomlkit

    version = get_version_from_tag()
    for pyproject in args.pyprojects:
        doc = tomlkit.parse(pyproject.read_text())
        assert "version" in doc["project"]
        doc["project"]["version"] = version
        pyproject.write_text(tomlkit.dumps(doc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    steps = parser.add_subparsers(dest="step", required=True)

    step = steps.add_parser("resolve-symlinks", help=resolve_symlinks.__doc__)
    step.set_defaults(func=resolve_symlinks)

    step = steps.add_parser("set-version", help=set_version.__doc__)
    step.add_argument("pyprojects", nargs="+", type=Path)
    step.set_defaults(func=set_version)

    args = parser.parse_args()
    args.func(args)
