# AMD production validation

ROCm validation passed on 2026-09-13 without changing the GEMM implementation. The final matrix contains 192 successful checks and three skips for the device-guard test requiring two GPUs. The normal ROCm JIT path passed with Torch 2.10 and 2.7. The stable HIP artifacts also passed with the isolated loader override described below.

## Source and environment

- Branch: `move-bmm-calls-to-stable-cshim`, tested at `4e3dcc8`.
- GPU: one AMD Instinct MI300X VF, 192 GB, `gfx942`.
- Driver reported by `rocm-smi`: `6.19.14.31400000`; Linux kernel `6.8.0-138-generic`.
- Python: 3.12.3.
- Build SDK: `/opt/rocm/core-10.0`, reporting HIP `7.15.26333`; host compiler GCC 13.3.0.
- Test environments: official Torch `2.10.0+rocm7.1` and `2.7.0+rocm6.3` wheels in separate virtual environments.
- Checkout, environments, caches, and logs: `/tmp/oeq-cshim-amd/` on the AMD machine.

The NVIDIA machine was not used during this validation.

## Results

The first production grouped-GEMM case passed before running the focused suites.

| Configuration | Import and grouped-GEMM tests | Symmetric-contraction tests |
| --- | --- | --- |
| Normal JIT path, Torch 2.10 / ROCm 7.1 | 48 passed, 1 skipped | 24 passed |
| Stable HIP artifacts, Torch 2.10 / ROCm 7.1, explicit test loader override | 48 passed, 1 skipped | 24 passed |
| Normal JIT path, Torch 2.7 / ROCm 6.3 | 48 passed, 1 skipped | Not run in this environment |

The focused tests cover both GEMM modes, float32/float64, noncontiguous inputs and counts, storage offsets, empty groups and dimensions, backward and double backward, current-stream behavior, graph capture/replay, compiled training with changing counts, and AOTI inference in a fresh process. The model suite compares with MACE for forward, backward, and double backward across three configurations and both dtypes, and checks compile and export.

Additional checks passed outside this matrix: the initial single GPU case, Torch 2.7 JIT compilation/import with GPU visibility disabled, and three editable-install import/dependency tests with GPU visibility disabled. They are not counted again in the 192-check total.

## Build, packaging, and BLAS dependency checks

The HIP wheel built successfully using the branch's CMake configuration:

- Artifact: `openequivariance-0.7.0-cp312-cp312-linux_x86_64.whl`.
- SHA256: `8ac6dc8cb3313298e613e7425e255fc369f89c797fbdefa1126778932fe76fb8`.
- Packaged libraries: `oeq_stable_hip.cpython-312-x86_64-linux-gnu.so` and `liboeq_stable_hip_aoti.so`.
- The wheel includes the JIT sources and excludes the `libtorch_hip.so` build stub.

ELF inspection of both stable libraries and both JIT libraries found no direct cuBLAS, hipBLAS, or rocBLAS library dependency and no undefined vendor BLAS functions. All four libraries reference `aoti_torch_cuda_bmm_out`. Both installed ROCm Torch libraries export that symbol and `aoti_torch_get_current_cuda_stream`, retaining the CUDA spelling on AMD.

The stable libraries depend on `libtorch_hip.so`, `libhiprtc.so.7`, and `libamdhip64.so.7`, along with CPU Torch and standard system libraries. The JIT builds link the Torch GPU library and the HIPRTC library found for their environments: `.so.7` for Torch 2.10 and `.so.6` for Torch 2.7. BLAS selection and handle ownership remain inside Torch.

The actual editable installation also passed. Its Python source resolved to the checkout, its compiled HIP extension resolved to the virtual environment, and `_has_precompiled_extension()` found the compiled artifacts. The existing ROCm policy selected JIT as intended. The normal wheel was restored afterward.

## Stable HIP loader policy

The production loader still unconditionally selects JIT on ROCm. This validation does not enable automatic stable HIP loading.

To exercise the compiled stable implementation, the wheel was extracted into `/tmp/oeq-cshim-amd/stable210-package`. Only the unconditional HIP precompiled-disable block was removed from that copy of the Python loader. Both shared libraries were verified byte-for-byte against the wheel. The tests asserted that the stable extension was selected and exercised the real registered operator, its existing autograd registration, and the companion AOTI library. No numerical operations were mocked.

This establishes hardware correctness for the tested stable artifacts while keeping the production loader policy unchanged.

## Setup and reproduction

The new machine lacked Python packaging bootstrap support and Python development headers. Pip was bootstrapped inside the isolated environments, and `python3.12-dev` was installed for extension compilation. The initial CMake attempt stopped at the missing Python headers; compilation succeeded after installing them. No GPU test failed, and no production source changes were needed.

The model-test environment used MACE 0.3.16, its pinned e3nn 0.4.4, NumPy 1.26.4, and pytest 9.1.1. As in the CUDA run, the test process allowed the built-in `slice` type for e3nn's packaged constants. The Torch 2.7 environment used e3nn 0.6.0 and NumPy 1.26.4.

With the wheel and dependencies installed, the normal ROCm path is exercised by:

```sh
ROCM_HOME=/opt/rocm/core-10.0 python -c 'import torch; torch.serialization.add_safe_globals([slice]); import pytest; raise SystemExit(pytest.main())' -q tests/import_test.py tests/group_gemm_test.py tests/symmetric_contraction_test.py
```

The actual runs used separate JIT and Inductor caches for each Torch version, and ran the focused and model suites separately. For stable-artifact validation, `PYTHONPATH` selected the isolated wheel copy described above. The older Torch environment ran the focused suite without the optional MACE dependency. Full environment snapshots, build logs, binary inspection output, and JUnit XML are retained in `/tmp/oeq-cshim-amd/logs/`.

## Scope limits

- Only one AMD GPU was available, so multi-GPU device guarding remains unverified.
- Automatic stable HIP loading remains disabled by existing policy.
- These are correctness and packaging results, with no performance claim.
- Direct GPU graph replay requires fixed ragged counts.

At the end of GPU testing, `rocm-smi` reported zero GPU utilization, zero allocated VRAM percentage, and no KFD processes. The GPU reservation was released; remaining installation checks used disabled GPU visibility.
