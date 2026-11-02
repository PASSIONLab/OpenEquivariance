#!/bin/bash
# Runs inside the manylinux container before cibuildwheel builds
# openequivariance. Installs only the headers and link-time libraries
# for CUDA and HIP; at runtime, the user's PyTorch install provides them.
#
# Configured from [tool.cibuildwheel.linux.environment] in pyproject.toml:
#   OEQ_CUDA_ROOTS  space-separated CUDA toolkits to install, e.g.
#                   "/usr/local/cuda-12.6 /usr/local/cuda-13.0"
# and the pinned rocm-sdk-core in .github/cibw_rocm_requirements.txt.
set -euo pipefail

ARCH=$(uname -m)

case "$ARCH" in
  x86_64)  CUDA_REPO_ARCH=x86_64 ;;
  aarch64) CUDA_REPO_ARCH=sbsa ;;
esac

dnf config-manager --add-repo \
  "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${CUDA_REPO_ARCH}/cuda-rhel8.repo"

for root in $OEQ_CUDA_ROOTS; do
  version=${root##*/cuda-}
  pkg_version=${version//./-}
  dnf install -y \
    cuda-nvcc-${pkg_version} \
    cuda-cudart-devel-${pkg_version} \
    cuda-nvrtc-devel-${pkg_version} \
    cuda-driver-devel-${pkg_version}
done

# rocm-sdk-core (~415 MB) has the HIP headers, libamdhip64 and libhiprtc,
# without the rest of the ROCm toolkit. AMD publishes no aarch64 builds.
if [ "$ARCH" = x86_64 ]; then
  /opt/python/cp312-cp312/bin/python -m pip install \
    --no-deps --require-hashes --target /opt/rocm-sdk-core \
    -r "$(dirname "$0")/cibw_rocm_requirements.txt"
fi
