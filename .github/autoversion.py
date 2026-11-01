"""
Sets the version in a pyproject.toml from git tags of the form 
vX.Y.Z on HEAD.

Usage: python3 .github/autoversion.py <path/to/pyproject.toml> [...]
"""

import re
import subprocess
import sys
from pathlib import Path

import tomlkit

TAG_PATTERN = re.compile(r"v([0-9]+\.[0-9]+\.[0-9]+)")


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


def set_version(pyproject, version):
    doc = tomlkit.parse(pyproject.read_text())
    assert "version" in doc["project"]
    doc["project"]["version"] = version
    pyproject.write_text(tomlkit.dumps(doc))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    version = get_version_from_tag()
    for path in sys.argv[1:]:
        set_version(Path(path), version)
