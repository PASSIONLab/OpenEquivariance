# Move Torch cuBLAS calls to the stable C shim

The production Torch extension now implements `libtorch_tp_jit::group_gemm` through `aoti_torch_cuda_bmm_out` on CUDA and ROCm. Both the stable extension and the source/JIT extension use this C entry point. The operator schema, fake implementation, output layouts, and custom backward formulas remain compatible.

GPU execution of this implementation is pending. GPU access is paused at the user's request; the earlier prototype results below are not results for this implementation.

## Implementation

The implementation lives in [group_mm.hpp](../../openequivariance/openequivariance/extension/group_mm.hpp), with input validation and output allocation in [torch_core.hpp](../../openequivariance/openequivariance/extension/torch_core.hpp). The algorithm is one BMM-out invocation per nonempty ragged group, using the same interleaved layout as the previous strided batched GEMMs. It does not need a vendor's heterogeneous grouped-GEMM API.

Inputs are made contiguous by the registered operator, then passed as borrowed `AtenTensorHandle` values to the shared helper. The stable extension gets these handles from `torch::stable::Tensor::get()`. The source/JIT extension uses Torch's `tensor_pointer_to_tensor_handle` utility to borrow a handle to its local `at::Tensor`; that bridge is compiled against the installed Torch, as the existing JIT extension already is. ATen C++ objects do not cross the stable extension's ABI boundary.

The helper creates temporary views with `aoti_torch__reinterpret_tensor`. These views retain the original storage and add their element offsets to the original tensor's storage offset. Their handles are released through RAII after each call, including when an error is returned. Original input and output handles remain owned by the caller. This replaces the prototype's `from_blob` construction and avoids reconstructing storage or querying pointer devices for every group. [Torch reinterpretation API](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/inductor/aoti_torch/shim_common.cpp#L435).

The operation checks GPU placement, matching devices and float32/float64 dtypes, tensor shapes, nonnegative dimensions, and mode 0 or 1. Ragged counts must be a CPU int64 vector with one entry per group. Counts may be noncontiguous; the operator makes a contiguous CPU copy before reading them. Counts must be nonnegative and sum to the input row count. Incremental bounds checking avoids overflowing the count sum. These checks establish the bounds needed by the storage-view helper.

Output allocation uses Torch and starts at zero. Empty groups are skipped; in mode 1 this leaves the corresponding weight-gradient block zero. Zero batch, output, or contraction dimensions return the zero-initialized output without invoking BMM. Dimensions and offsets remain int64 throughout the grouped-GEMM implementation.

## Layout mapping

Let `b = batch_size`, `n = ragged_counts[i]`, and `o` be the sum of preceding counts. Offsets are relative to the logical beginning of each original tensor; strides and offsets are in elements.

| Mode | Tensor | Element offset | View shape | View strides |
| --- | --- | --- | --- | --- |
| 0 | B / input | `o*b*k` | `[b,n,k]` | `[k,b*k,1]` |
| 0 | A / weights | `i*b*m*k` | `[b,k,m]` | `[m*k,1,k]` |
| 0 | C / output | `o*b*m` | `[b,n,m]` | `[m,b*m,1]` |
| 1 | A / left | `o*b*m` | `[b,m,n]` | `[m,1,b*m]` |
| 1 | B / right | `o*b*k` | `[b,n,k]` | `[k,b*k,1]` |
| 1 | C / output | `i*b*m*k` | `[b,m,k]` | `[m*k,k,1]` |

The final operation for each group is `aoti_torch_cuda_bmm_out(output, left, right)`. The API overwrites the output with alpha 1 and beta 0. Torch may choose its own BLAS kernel or backend. The layout code requests views, but makes no unconditional promise about copies inside Torch's BMM implementation. [Torch BMM implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/Blas.cpp#L536).

## Device, stream, precision, and autograd

The registered operator guards A's device before making GPU contiguous copies or allocating the output. The stable build uses `torch::stable::accelerator::DeviceGuard`; the JIT build uses `c10::DeviceGuard`. Each restores the caller's current device on return.

BMM uses Torch's current stream on that device. OEQ neither creates a BLAS handle nor changes a BLAS stream. Calls remain asynchronous and follow PyTorch's normal storage/stream lifetime contract.

GPU graph capture records the group sizes read by the host loop during capture. Direct graph replay requires those counts to remain unchanged; changing group sizes requires recapture. Ordinary calls and compiled execution without graph replay read the counts on each invocation.

Following Torch's precision, determinism, and preferred-BLAS settings is an intentional behavior change relative to main's independently created handle. There is no OEQ-specific precision override. Custom backward registration remains responsible for differentiation through the grouped operator; the internal BMM-out invocation does not replace that registration.

## ABI and build paths

The official declaration is in `torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h`. The backend-specific spelling is used instead of the deprecated `aoti_torch_bmm_out`. [Declaration](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h#L74), [stable ABI policy](https://docs.pytorch.org/docs/main/notes/libtorch_stable_abi.html).

Both `aoti_torch_cuda_bmm_out` and `aoti_torch__reinterpret_tensor` are present in the inspected PyTorch 2.4 headers, matching OEQ's documented source/JIT baseline. The stable build explicitly targets the 2.10 ABI and continues to use the pinned LibTorch headers/libraries. This uses Torch's stated ABI guarantees; it does not try to select cuBLAS versions or infer handle ownership from version queries.

The source/JIT loader explicitly links the installed `torch_cuda` or `torch_hip` library supplying the BMM shim. CUDA also retains its driver/runtime/NVRTC dependencies; ROCm links hipRTC. All direct cuBLAS/rocBLAS calls, handle management, and corresponding OEQ link dependencies have been removed.

The stable wheel build uses CPU LibTorch plus small GPU link stubs. [The stub](../../openequivariance/openequivariance/extension/stubs/stream.cpp) now declares and defines BMM-out using the official generated header, alongside the existing stream symbol. Stub bodies return failure if accidentally invoked. Only the real Torch GPU libraries are intended at runtime; CMake installs the OEQ targets, not the stubs. Both the Python extension and AOTI targets use this arrangement.

CMake installs the stable libraries into the wheel's package directory. The previous absolute destination wrote them into the source tree, where the wheel's ignore rules excluded them. The loader now resolves the compiled extension's location to find its companion AOTI library; this also supports editable installations where Python sources and compiled libraries live in different directories. Build CI installs a normal wheel and asserts that the stable extension is selected before running the import checks.

The HIP CMake target is named `oeq_stable_hip`, matching its module entry point and expected artifact name. It explicitly links `hiprtc::hiprtc`. The existing Python loader still selects JIT compilation for HIP; enabling precompiled HIP loading is outside this change. The ROCm JIT path uses the same BMM helper as the stable CUDA build.

## ROCm evidence

PyTorch v2.10 adds the generated `c_shim_cuda.cpp` to `torch_hip` under `USE_ROCM`. The C symbol retains the `cuda` spelling on ROCm. The underlying GEMM implementations are converted to HIP BLAS calls, and Torch handles backend selection, including the double-precision fallback from hipBLASLt. [ROCm library construction](https://github.com/pytorch/pytorch/blob/v2.10.0/caffe2/CMakeLists.txt#L941), [HIP BLAS mappings](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/hipify/cuda_to_hip_mappings.py#L6826), [backend selection](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CUDABlas.cpp#L779).

This supports the implementation choice but does not establish AMD hardware correctness. ROCm execution remains pending.

## JAX scope

JAX does not call this grouped-GEMM helper and has no BLAS dependency in its extension target. The replacement is entirely within the Torch frontend. The common CUDA/HIP kernel compilation backend gains no Torch dependency.

JAX's public FFI provides buffers and a GPU stream, without an equivalent BMM C shim. Future JAX grouped GEMM would require native JAX graph operations where shapes permit, or a separate FFI BLAS integration. [JAX FFI](https://docs.jax.dev/en/latest/ffi.html#ffi-calls-on-a-gpu).

## Validation and next GPU run

Local checks passed for the new operator and both adapters against Torch 2.10 headers, the link stub, and the shared helper against Torch 2.4 headers. These were host syntax checks, not full CUDA/ROCm extension builds. The unmodified production view helper also passed eight cases using real Torch 2.10 CPU tensors with the GPU BMM entry point redirected to CPU BMM for this check: both modes/dtypes, nonzero storage offsets, empty groups, and output guard values. All 46 GPU integration cases collect successfully; none has run on a GPU yet.

[The integration tests](../../tests/group_gemm_test.py) call the real registered operator. They cover both modes/dtypes, noncontiguous inputs and counts, nonzero input storage offsets, empty and singleton dimensions, varied group sizes, backward and double-backward gradients, the current stream, device guarding, invalid counts, graph replay, compiled training, and AOTI inference in a fresh process that loads only the exported OEQ library. The device-guard test requires two GPUs. The same test file supports CUDA and ROCm.

[The import tests](../../tests/import_test.py) also inspect the extension and AOTI library's ELF dependencies and undefined symbols to check that OEQ has no direct vendor BLAS dependency. These checks run in the existing build-verification workflow for precompiled and JIT imports.

After GPU access resumes, start with one case of the production operator:

```sh
pytest -q 'tests/group_gemm_test.py::test_group_gemm_matches_reference[contiguous-0-dtype0]'
```

Then run the focused integration suite against the stable build and the JIT build in separate processes:

```sh
pytest -q tests/import_test.py tests/group_gemm_test.py
OEQ_JIT_EXTENSION=1 pytest -q tests/import_test.py tests/group_gemm_test.py
```

Existing symmetric-contraction integration tests exercise the surrounding model and its higher-order derivatives when the optional MACE dependency is available. No end-to-end performance claim is made before measurement.

## Earlier experiments

The original [standalone prototype](group_gemm.cpp) and its [smoke script](smoke_test.py) remain as historical experiment artifacts. They use blob views rather than the production helper's views of existing storage.

The prototype passed one float32 forward case on an H100 PCIe with Torch 2.7.0 / CUDA 12.8: counts `[2,0,5,1]`, batch 3, m 4, k 5, a nondefault stream, and maximum absolute error `4.37e-7` against a CPU float64 reference. Earlier ctypes calls to the BMM C shim passed 40 small cases across both dtypes/modes. Four selected BMM-only profiles showed no copy or GPU allocation inside the call; they did not measure view creation overhead.

The earlier direct-cuBLAS experiments explain the migration: matching handle/function owners worked, whereas some separately loaded foreign-owner combinations failed or crashed despite successful version queries. They do not establish interchangeability of private BLAS handles. Related: [PR #206](https://github.com/PASSIONLab/OpenEquivariance/pull/206).
