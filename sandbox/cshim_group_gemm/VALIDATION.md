# Production validation

The production implementation passed CUDA validation on 2026-09-13. The final test matrix contains 192 successful checks and three skips, all for the same device-guard test requiring two GPUs. These results cover the registered production operator and installed artifacts, not the earlier standalone prototype.

Subsequent ROCm hardware validation is recorded in [the AMD report](AMD_VALIDATION.md).

## Source and artifact

- Branch: `move-bmm-calls-to-stable-cshim`, based on main at `dbe854415da7771eba33195534c171adbca5677b`.
- Production source: `d2eac4e`; integration tests: `e4df36c`.
- Wheel: `openequivariance-0.7.0-cp310-cp310-linux_x86_64.whl`.
- Wheel SHA256: `f536c53f1ef2358b6f562931c30c3f55823c32123424c153c96cb808250ea97a`.
- Remote checkout and environments: `/tmp/oeq-cshim-production/`, separate from the workspace-allocation checkout and user-installed packages.

## Build and packaging results

All checks in this table ran with `CUDA_VISIBLE_DEVICES=""`. They establish successful compilation, loading, and packaging; they do not establish GPU numerical correctness.

| Configuration | Result |
| --- | --- |
| Stable wheel, Torch 2.10.0+cu128, Python 3.10 | Both production libraries built; 3 import/dependency tests passed |
| Editable install, Torch 2.10.0+cu128 | Stable extension selected with Python sources and libraries in separate directories; 3 import/dependency tests passed |
| Source/JIT extension, Torch 2.10.0+cu128 | Compiled against the installed Torch; 3 import/dependency tests passed |
| Source/JIT extension, Torch 2.7.0, CUDA 12.8 | Compiled against the installed Torch; 3 import/dependency tests passed |

The wheel contains its Python extension, companion AOTI library, and JIT sources. It contains no `libtorch_cuda.so` build stub. ELF checks found no direct vendor BLAS library dependency or undefined vendor BLAS function in either production library or either JIT build. Both installed Torch GPU libraries export `aoti_torch_cuda_bmm_out`.

Validation found and fixed a packaging defect: the absolute CMake install destination placed libraries in the source tree, where wheel ignore rules excluded them. The relative install destination includes the libraries in the wheel. Resolving the extension's actual import location also fixes stable-library discovery for editable installations.

## GPU validation

Hardware: one NVIDIA H100 PCIe, driver 580.105.08. Both Torch environments report CUDA 12.8. The first production-wheel forward case passed before running the larger suites.

| Configuration | Import and grouped-GEMM tests | Symmetric-contraction tests |
| --- | --- | --- |
| Stable wheel, Torch 2.10.0+cu128 | 48 passed, 1 skipped | 24 passed |
| Source/JIT, Torch 2.10.0+cu128 | 48 passed, 1 skipped | 24 passed |
| Source/JIT, installed Torch 2.7.0 / CUDA 12.8 | 48 passed, 1 skipped | Not run in this environment |

The stable-wheel result combines 46 initial passes with two successful AOTI reruns after correcting fresh-process loader initialization. The Torch 2.7 result is a complete rerun after correcting test-environment dependencies. The table reports final per-case outcomes; it does not count preliminary or repeated passes twice.

The 46 grouped-GEMM cases cover layouts, empty groups and dimensions, both modes and dtypes, backward and double backward, the current stream, device guarding, invalid counts, graph capture/replay, compiled training with changing counts, and AOTI inference. Each AOTI case loads the exported OEQ library in a new Python process without importing the OEQ Python package, then compares against a CPU reference. Three additional tests check import, successful extension loading, and binary dependencies.

The existing symmetric-contraction suite compares with MACE for float32/float64 forward, backward, and double backward across three configurations. It also checks compile and export. Both the stable and JIT extensions passed all 24 cases.

## Issues resolved during validation

The wheel and editable-discovery fixes above were production changes. GPU execution required no further changes to the GEMM implementation.

The fresh-process AOTI test exposed a Torch 2.10 loader initialization issue: its package loader accesses `torch._inductor.codecache` before importing it. The subprocess now explicitly imports that module before loading the package. Both AOTI modes then passed for the stable wheel and both JIT builds.

The Torch 2.10 model-test environment uses MACE 0.3.16 and its pinned e3nn 0.4.4. That e3nn version loads packaged constants containing Python `slice` objects. The test process allowed that specific built-in type with `torch.serialization.add_safe_globals([slice])`; no production code or global loading policy changed. NumPy was 1.26.4 and pytest was 9.1.1.

The Torch 2.7 environment initially inherited NetworkX 2.4 from the system; its import failed on NumPy's removed `np.int` alias before compilation. Installing NetworkX 3.4.2, SciPy 1.15.3, and SymPy 1.13.3 inside the isolated environment corrected its compiler dependencies. Its final environment used e3nn 0.6.0 and NumPy 1.26.4. System packages and the other checkout were untouched.

## Reproduction

After installing the wheel and test dependencies in a matching GPU Torch environment, run:

```sh
python -c 'import torch; torch.serialization.add_safe_globals([slice]); import pytest; raise SystemExit(pytest.main())' -q tests/import_test.py tests/group_gemm_test.py tests/symmetric_contraction_test.py
OEQ_JIT_EXTENSION=1 python -c 'import torch; torch.serialization.add_safe_globals([slice]); import pytest; raise SystemExit(pytest.main())' -q tests/import_test.py tests/group_gemm_test.py tests/symmetric_contraction_test.py
```

The allowlist is only needed for the older e3nn dependency described above. The actual runs used separate JIT and Inductor cache directories for each Torch environment and ran the focused and model suites separately. The Torch 2.7 environment also set `PYTHONNOUSERSITE=1` to exclude the shared machine's user packages.

## Scope limits

- Multi-GPU device guarding remains unverified on hardware because only one GPU was available.
- Torch 2.4 received header checks only; GPU execution covered Torch 2.7 and 2.10. CUDA versions other than 12.8 were not exercised by this production test matrix.
- Direct GPU graph replay requires fixed ragged counts, as described in the design.
- No end-to-end performance claim is made by these correctness and packaging checks.

Detailed logs, environment snapshots, and JUnit XML are retained under `/tmp/oeq-cshim-production/logs/` on the validation machine. At the end of testing, the GPU reported zero memory use and no compute processes; its status reservation was released.
