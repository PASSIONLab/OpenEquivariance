#!/bin/bash
# Runs inside the manylinux container before cibuildwheel builds
# openequivariance. Installs only the headers and link-time libraries
# for CUDA and HIP; at runtime, the user's PyTorch install provides them.
set -euo pipefail

CUDA_VERSION=13-0
ARCH=$(uname -m)

case "$ARCH" in
  x86_64)  CUDA_REPO_ARCH=x86_64 ;;
  aarch64) CUDA_REPO_ARCH=sbsa ;;
esac

dnf config-manager --add-repo \
  "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${CUDA_REPO_ARCH}/cuda-rhel8.repo"
dnf install -y \
  cuda-nvcc-${CUDA_VERSION} \
  cuda-cudart-devel-${CUDA_VERSION} \
  cuda-nvrtc-devel-${CUDA_VERSION} \
  cuda-driver-devel-${CUDA_VERSION}

# rocm-sdk-core (~415 MB) has the HIP headers, libamdhip64 and libhiprtc,
# without the rest of the ROCm toolkit. AMD publishes no aarch64 builds.
if [ "$ARCH" = x86_64 ]; then
  echo "rocm-sdk-core==10.0.0 --hash=sha256:3bf4a72d11aa2a4ee1e90572c73630f937d38c1b7af4dda45b483b54655138a7" \
    > /tmp/rocm-requirements.txt
  /opt/python/cp312-cp312/bin/python -m pip install \
    --no-deps --require-hashes --target /opt/rocm-sdk-core \
    --index-url https://download.pytorch.org/whl/rocm10.0 \
    -r /tmp/rocm-requirements.txt
fi
