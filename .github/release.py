"""
Release pipeline steps for .github/workflows/release.yaml.

    python3 .github/release.py resolve-symlinks
    python3 .github/release.py set-version <path/to/pyproject.toml> [...]
    python3 .github/release.py install-gpu-toolkits <cuda-root> [...]
"""

import argparse
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TAG_PATTERN = re.compile(r"v([0-9]+\.[0-9]+\.[0-9]+)")
GIT_SYMLINK_MODE = "120000"
CUDA_ROOT_PATTERN = re.compile(r"/usr/local/cuda-([0-9]+)\.([0-9]+)")
CUDA_REPO_ARCH = {"x86_64": "x86_64", "aarch64": "sbsa"}
ROCM_REQUIREMENTS = """\
--index-url https://download.pytorch.org/whl/rocm10.0
rocm-sdk-core==10.0.0 --hash=sha256:3bf4a72d11aa2a4ee1e90572c73630f937d38c1b7af4dda45b483b54655138a7
"""


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


def install_gpu_toolkits(args):
    """Installs the CUDA and HIP headers and link-time libraries inside a 
    manylinux container.""" 
    arch = platform.machine()
    subprocess.run(
        [
            "dnf",
            "config-manager",
            "--add-repo",
            "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/"
            f"{CUDA_REPO_ARCH[arch]}/cuda-rhel8.repo",
        ],
        check=True,
    )

    for root in args.cuda_roots:
        match = CUDA_ROOT_PATTERN.fullmatch(root)
        assert match, f"expected a CUDA root like /usr/local/cuda-12.6, got {root!r}"
        pkg_version = "-".join(match.groups())
        packages = [
            "cuda-nvcc",
            "cuda-cudart-devel",
            "cuda-nvrtc-devel",
            "cuda-driver-devel",
        ]
        subprocess.run(
            ["dnf", "install", "-y", *(f"{pkg}-{pkg_version}" for pkg in packages)],
            check=True,
        )

    if arch == "x86_64":
        with tempfile.TemporaryDirectory() as tmp:
            requirements = Path(tmp) / "rocm_requirements.txt"
            requirements.write_text(ROCM_REQUIREMENTS)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--require-hashes",
                    "--target",
                    "/opt/rocm-sdk-core",
                    "-r",
                    str(requirements),
                ],
                check=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    steps = parser.add_subparsers(dest="step", required=True)

    step = steps.add_parser("resolve-symlinks", help=resolve_symlinks.__doc__)
    step.set_defaults(func=resolve_symlinks)

    step = steps.add_parser("set-version", help=set_version.__doc__)
    step.add_argument("pyprojects", nargs="+", type=Path)
    step.set_defaults(func=set_version)

    step = steps.add_parser("install-gpu-toolkits", help=install_gpu_toolkits.__doc__)
    step.add_argument("cuda_roots", nargs="+")
    step.set_defaults(func=install_gpu_toolkits)

    args = parser.parse_args()
    args.func(args)
