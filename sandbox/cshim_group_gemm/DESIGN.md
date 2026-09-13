# Move Torch cuBLAS calls to the stable C shim

Replace OEQ's direct CUDA and ROCm BLAS calls for `libtorch_tp_jit::group_gemm` with PyTorch's `aoti_torch_cuda_bmm_out` C API. Keep the existing grouped operation, tensor layouts, output allocation, and custom autograd rules. PyTorch will own the underlying BLAS handles, streams, library calls, and backend selection.

Related: [PR #206](https://github.com/PASSIONLab/OpenEquivariance/pull/206).

## Problem and scope

OEQ's grouped GEMM is a host loop over nonempty ragged groups, issuing one strided batched GEMM per group. It does not require a vendor's heterogeneous grouped-GEMM API. The only direct BLAS computation calls found are float32/float64 strided batched GEMMs in `extension/group_mm.hpp`, called from the Torch backend in `extension/torch_core.hpp`.

Using PyTorch's stable ABI to obtain a cuBLAS handle does not make that handle interchangeable with a separately loaded cuBLAS implementation. Earlier H100 experiments passed with matching handle/function owners, while some foreign-owner combinations returned incorrect streams, failed, or crashed. Successful version queries did not establish handle compatibility. Moving the GEMM itself behind PyTorch's C ABI removes this borrowed-handle boundary.

The current checkout at `dbe854415da7771eba33195534c171adbca5677b` creates its own static BLAS handle; the inspected PR snapshot at `b551d7469db6d7ac688859cec46a3bb2b1c71a5b` borrows Torch's handle. The replacement removes either form of OEQ-side handle management. The final patch must be based on the branch being merged.

The stable wheels currently target PyTorch 2.10. Use official stable C declarations, with no dependency on ATen's C++ ABI in that build. The C entry point was also exercised successfully on the available PyTorch 2.7 installation; that observation does not establish support for every older Torch release. [PyTorch stable ABI documentation](https://docs.pytorch.org/docs/main/notes/libtorch_stable_abi.html).

## Public interface

Include `torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h`, which declares:

```cpp
AOTITorchError aoti_torch_cuda_bmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);
```

Use this generated API rather than the deprecated `aoti_torch_bmm_out` spelling. The API takes PyTorch tensor handles, not cuBLAS/hipBLAS handles. OEQ does not call a BLAS version getter or select a vendor library at runtime. [PyTorch 2.10 declaration](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h#L74).

Add a Torch-specific helper, such as `group_mm_torch.hpp`, taking the existing input/output pointers, dimensions, CPU ragged counts, dtype, and explicit device index. Both Torch extension variants should use it where their supported Torch versions expose the required C symbols. Keep this helper outside the framework-independent CUDA/HIP kernel backend.

Create temporary, non-owning tensor views with `aoti_torch_create_tensor_from_blob_v2`. Specify sizes and strides in elements, storage offset zero, and the actual input device. Release each temporary tensor handle with `aoti_torch_delete_tensor_object` through RAII, including error paths. Releasing metadata must not release the caller's input or output storage. [PyTorch's blob implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/inductor/aoti_torch/shim_common.cpp#L525).

## Layout mapping

Let `b = batch_size`, `n = ragged_counts[i]`, and `o` be the sum of preceding counts. Pointer offsets and strides below are measured in elements.

| Mode | View | Base pointer | Shape | Strides |
| --- | --- | --- | --- | --- |
| `ragged_inner == 0` | Left / input | `B + o*b*k` | `[b,n,k]` | `[k,b*k,1]` |
| `ragged_inner == 0` | Right / weights | `A + i*b*m*k` | `[b,k,m]` | `[m*k,1,k]` |
| `ragged_inner == 0` | Output | `C + o*b*m` | `[b,n,m]` | `[m,b*m,1]` |
| `ragged_inner == 1` | Left | `A + o*b*m` | `[b,m,n]` | `[m,1,b*m]` |
| `ragged_inner == 1` | Right | `B + o*b*k` | `[b,n,k]` | `[k,b*k,1]` |
| `ragged_inner == 1` | Output | `C + i*b*m*k` | `[b,m,k]` | `[m*k,k,1]` |

Call BMM-out once for each nonempty group. Preserve the existing zero-initialized output and skip empty groups, including the all-empty case. This matters for the weight-gradient output blocks in mode 1. Use 64-bit dimensions and offsets throughout instead of the current narrowing casts to `int`.

These views preserve OEQ's interleaved batch layout without explicitly transposing or copying the underlying buffers. PyTorch can handle BLAS-compatible strides directly, but some layouts can trigger its internal copy path; avoiding all copies is not an unconditional API guarantee. [BMM layout handling](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/Blas.cpp#L536).

## Device, stream, and autograd behavior

Guard the inputs' actual GPU device before making contiguous copies, allocating the output, constructing views, or invoking BMM. Restore the caller's device on return. The 2.10 stable implementation can use `aoti_torch_create_device_guard` / `aoti_torch_delete_device_guard`, whose implementation selects Torch's active accelerator; this avoids requiring a CUDA-specific guard implementation in the shared helper. The existing standalone prototype uses the older CUDA-named guard because it was tested on Torch 2.7. [Generic guard implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/inductor/aoti_torch/shim_common.cpp#L1468).

Use Torch's current stream for that device. Do not set a BLAS stream, create a stream, or synchronize inside the operator. Original tensors must remain valid through launch; asynchronous storage lifetime follows the usual PyTorch current-stream contract. No global or cached raw pointers or tensor views are needed.

Validate the raw-pointer helper's preconditions at the tensor boundary: same GPU device and dtype, supported float32/float64 types, contiguous CPU int64 counts, valid counts length and nonnegative values, compatible input shapes, and a valid mode. Check that counts describe the available rows before constructing views. Keep the existing custom operator schema, fake implementation, and backward formulas. BMM-out is an internal implementation detail; the custom autograd registration remains responsible for derivatives.

PyTorch's BMM implementation uses alpha 1 and beta 0, matching the existing GEMMs. It may choose a different BLAS backend or kernel and apply Torch's precision/determinism settings, so require numerical agreement rather than bitwise equivalence. [BMM implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/Blas.cpp#L680).

## Why this should work on ROCm

Source inspection of PyTorch **v2.10.0** supports using the same generated function on both GPU platforms:

1. PyTorch's build defines `GENERATED_CXX_TORCH_CUDA` to contain `c_shim_cuda.cpp`, then explicitly adds that source to `torch_hip` under `USE_ROCM`. The generated shim is therefore part of the ROCm library too. [Generated source definition](https://github.com/pytorch/pytorch/blob/v2.10.0/caffe2/CMakeLists.txt#L331), [ROCm library construction](https://github.com/pytorch/pytorch/blob/v2.10.0/caffe2/CMakeLists.txt#L941).
2. The shim generator names the API using the `CUDA` dispatch key and generates a call into the corresponding ATen backend. The exported name remains `aoti_torch_cuda_bmm_out`; do not invent an `aoti_torch_hip_bmm_out` symbol. [Shim generator](https://github.com/pytorch/pytorch/blob/v2.10.0/torchgen/gen_aoti_c_shim.py#L492).
3. BMM reaches Torch's GEMM/batched GEMM implementation. Its HIP conversion maps `cublasSgemmStridedBatched` and `cublasDgemmStridedBatched` to their `hipblas` equivalents. Torch also contains ROCm-specific backend selection, including a double-precision fallback when hipBLASLt cannot handle the operation. [HIP mappings](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/hipify/cuda_to_hip_mappings.py#L6826), [float/double backend selection](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CUDABlas.cpp#L779).

Torch's ROCm tensors use the `cuda` device interface. The helper should use Torch's CUDA device-type value for both these Torch builds, rather than confusing that convention with a distinct external HIP tensor type. [ROCm semantics](https://docs.pytorch.org/docs/main/notes/hip.html).

This is **source-level evidence**, not an AMD hardware test or a binary export audit of a ROCm wheel. Verify the exports in the supported wheel and run one AMD smoke case before marking ROCm complete.

## Build and packaging changes

- Remove direct cuBLAS/rocBLAS includes, calls, and handle management from the Torch grouped-GEMM path. On the PR branch, also remove the now-unused borrowed-handle adapters.
- Remove `CUDA::cublas`, the grouped-GEMM-only `find_package(rocblas)` / `${HIP_BLAS_LIB}`, and the JIT loader's explicit `-lcublas` link flag after confirming there are no remaining consumers. Torch still brings its own BLAS dependencies.
- Resolve the generated BMM symbol through the installed Torch GPU library: `libtorch_cuda` for CUDA and `libtorch_hip` for ROCm. Common tensor/guard C APIs are supplied by Torch's common library.
- OEQ's wheel build currently downloads CPU LibTorch headers/libraries and creates GPU link stubs from `extension/stubs/stream.cpp`. Extend those build-only stubs with the exact generated BMM declaration/definition, sharing the header so signature drift causes a build error. If retaining the prototype's GPU-specific guards, add their GPU exports too; the proposed 2.10 generic guard avoids that need. Never package or execute the fake implementations. Check that installed extensions resolve against the real Torch libraries.
- Apply changes to both the Python extension and the AOTI shared library targets, and to source/JIT builds. The existing HIP CMake target is named `torch_stable_hip` while import/install expectations use `oeq_stable_hip`; resolve that naming mismatch as part of making the HIP artifact load correctly.
- Keep the existing CUDA/HIP runtime and runtime-compiler dependencies needed by OEQ-generated kernels. Removing the BLAS dependency does not make the entire extension independent of the GPU platform.

The stable-wheel baseline remains 2.10. Confirm the supported source/JIT baseline before sharing every C helper with that build: the prototype's helpers were exercised on 2.7, whereas the proposed generic guard needs its own baseline check. If older JIT support must be preserved, a JIT-only adapter using that installed Torch's ordinary BMM-out/device guard is acceptable; it still eliminates direct vendor BLAS calls and does not enter the stable wheel.

## JAX scope

Neither the inspected PR snapshot nor the current JAX extension calls `group_gemm_blas`; its CMake target links the runtime, driver, and NVRTC, without BLAS. This replacement therefore remains entirely within Torch.

JAX's public FFI has buffers and a GPU stream, but no equivalent BMM shim or public BLAS-handle getter. If a future JAX operation needs this GEMM, express it in the compiled JAX graph where shapes permit, or provide a separate FFI implementation whose BLAS handles and calls come from the same library. [JAX GPU FFI](https://docs.jax.dev/en/latest/ffi.html#ffi-calls-on-a-gpu), [public XLA FFI API](https://github.com/openxla/xla/blob/main/xla/ffi/api/c_api.h#L757).

## Existing prototype and validation

A standalone C++ prototype already implements both layout branches using only Torch C APIs. Its single float32 forward smoke case passed on an H100 PCIe with Torch 2.7.0 / CUDA 12.8: counts `[2,0,5,1]`, batch 3, m 4, k 5, a nondefault stream, and maximum absolute error `4.37e-7` against an independent CPU float64 reference.

Earlier ctypes calls to the same C shim passed 40 cases spanning both dtypes and modes. Four selected profiles showed one GEMM kernel with no observed copy or GPU allocation inside the BMM call. These measurements do not cover the CPU cost of temporary tensor metadata or establish end-to-end performance.

The integrated extension, its autograd registration, the final wheel link setup, and ROCm execution remain to be validated. No additional GPU experiments were run for this design.

## Acceptance criteria

- Both grouped-GEMM modes use Torch BMM, with no direct vendor BLAS symbols or handles in OEQ's Torch extension.
- Existing layouts, empty groups, float32/float64 behavior, operator schema, fake implementation, and gradients are preserved.
- Device guarding and a nondefault current stream work; the caller's current device is restored.
- Stable CUDA and HIP artifacts resolve the official C symbols in the real installed Torch GPU library; link stubs do not ship. Audit direct dynamic dependencies to confirm BLAS is now Torch's responsibility.
- Run focused correctness checks for both modes/dtypes and one relevant autograd case during integration, plus one ROCm smoke case when AMD hardware is available. Keep this bounded; no exhaustive vendor-version matrix is needed to establish the prototype.
- Check representative overhead before claiming performance parity. Graph capture and broader multi-device coverage should be checked where required by the supported operator contract.

## Prototype code

The following is the existing standalone CUDA prototype, not the final production registration or generic-device-guard adaptation. The caller supplies validated buffers and zero-initializes outputs where empty groups need to remain zero.

```cpp
// Prototype for the PyTorch CUDA backend. Uses only PyTorch's stable C ABI.
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace {

void check(AOTITorchError status) {
    if (status != AOTI_TORCH_SUCCESS)
        throw std::runtime_error("PyTorch C shim failed");
}

using Tensor = std::unique_ptr<
    std::remove_pointer_t<AtenTensorHandle>,
    decltype(&aoti_torch_delete_tensor_object)>;
using DeviceGuard = std::unique_ptr<
    std::remove_pointer_t<CUDAGuardHandle>,
    decltype(&aoti_torch_delete_cuda_guard)>;

// Only tensor metadata is created; the caller retains ownership of the buffer.
Tensor view(void* data, std::array<int64_t, 3> sizes,
            std::array<int64_t, 3> strides, int32_t dtype, int32_t device) {
    AtenTensorHandle tensor = nullptr;
    check(aoti_torch_create_tensor_from_blob_v2(
        data, 3, sizes.data(), strides.data(), 0, dtype,
        aoti_torch_device_type_cuda(), device, &tensor,
        aoti_torch_layout_strided(), nullptr, 0));
    return Tensor(tensor, aoti_torch_delete_tensor_object);
}

} // namespace

template <typename T>
void group_gemm_cshim(
    T* A, T* B, T* C, const int64_t* ragged_counts, int num_groups,
    int64_t batch, int64_t m, int64_t k, int ragged_inner, int32_t device) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    const int32_t dtype = std::is_same_v<T, float>
        ? aoti_torch_dtype_float32() : aoti_torch_dtype_float64();

    CUDAGuardHandle raw_guard = nullptr;
    check(aoti_torch_create_cuda_guard(device, &raw_guard));
    DeviceGuard guard(raw_guard, aoti_torch_delete_cuda_guard);

    int64_t offset = 0;
    for (int i = 0; i < num_groups; ++i) {
        const int64_t n = ragged_counts[i];
        if (n == 0) continue; // Preserve the original empty-group behavior.

        if (ragged_inner == 0) {
            // [batch, n, k] @ [batch, k, m] -> [batch, n, m]
            auto input = view(B + offset * batch * k,
                              {batch, n, k}, {k, batch * k, 1}, dtype, device);
            auto weight = view(A + i * batch * m * k,
                               {batch, k, m}, {m * k, 1, k}, dtype, device);
            auto output = view(C + offset * batch * m,
                               {batch, n, m}, {m, batch * m, 1}, dtype, device);
            check(aoti_torch_cuda_bmm_out(output.get(), input.get(), weight.get()));
        } else {
            // [batch, m, n] @ [batch, n, k] -> [batch, m, k]
            auto left = view(A + offset * batch * m,
                             {batch, m, n}, {m, 1, batch * m}, dtype, device);
            auto right = view(B + offset * batch * k,
                              {batch, n, k}, {k, batch * k, 1}, dtype, device);
            auto output = view(C + i * batch * m * k,
                               {batch, m, k}, {m * k, k, 1}, dtype, device);
            check(aoti_torch_cuda_bmm_out(output.get(), left.get(), right.get()));
        }
        offset += n;
    }
}

// Small ctypes entry point for the smoke test, not production registration code.
extern "C" int oeq_group_gemm_cshim(
    int dtype, void* A, void* B, void* C, const int64_t* counts,
    int groups, int64_t batch, int64_t m, int64_t k, int inner, int32_t device) {
    try {
        if (dtype == 0) {
            group_gemm_cshim(static_cast<float*>(A), static_cast<float*>(B),
                             static_cast<float*>(C), counts, groups,
                             batch, m, k, inner, device);
        } else if (dtype == 1) {
            group_gemm_cshim(static_cast<double*>(A), static_cast<double*>(B),
                             static_cast<double*>(C), counts, groups,
                             batch, m, k, inner, device);
        } else {
            throw std::runtime_error("Expected float32 or float64");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
```

<details>
<summary>Reproduce the single CUDA smoke case</summary>

Save the C++ above as `group_gemm.cpp` and this script beside it as `smoke_test.py`, then run `python3 smoke_test.py` with a CUDA-enabled Torch installation and a C++ compiler. The script builds only the standalone prototype and executes one configuration.

```python
"""Compile the C++ prototype and run one float32 CUDA correctness case."""
import ctypes
from pathlib import Path
import subprocess

import torch


def main():
    root = Path(__file__).resolve().parent
    torch_root = Path(torch.__file__).resolve().parent
    library = root / "group_gemm.so"
    subprocess.run(
        [
            "g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-DUSE_CUDA",
            f"-I{torch_root / 'include'}", str(root / "group_gemm.cpp"),
            f"-L{torch_root / 'lib'}", f"-Wl,-rpath,{torch_root / 'lib'}",
            "-Wl,--no-undefined", "-ltorch_cuda", "-ltorch_cpu",
            "-o", str(library),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(library))
    call = lib.oeq_group_gemm_cshim
    i64 = ctypes.c_int64
    call.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(i64), ctypes.c_int, i64, i64, i64,
        ctypes.c_int, ctypes.c_int32,
    ]
    call.restype = ctypes.c_int

    # One case includes differently sized groups, an empty group, interleaved
    # batches, and a nondefault stream. The reference runs on the CPU.
    counts = [2, 0, 5, 1]
    batch, m, k = 3, 4, 5
    torch.manual_seed(123)
    weights_cpu = torch.randn(len(counts), batch, m, k)
    input_cpu = torch.randn(sum(counts), batch, k)
    expected = torch.empty(sum(counts), batch, m, dtype=torch.float64)
    offset = 0
    for i, n in enumerate(counts):
        expected[offset:offset + n] = torch.einsum(
            "bmk,nbk->nbm", weights_cpu[i].double(),
            input_cpu[offset:offset + n].double(),
        )
        offset += n

    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        weights = weights_cpu.cuda()
        inputs = input_cpu.cuda()
        output = torch.full(expected.shape, float("nan"), device="cuda")
        status = call(
            0, weights.data_ptr(), inputs.data_ptr(), output.data_ptr(),
            (i64 * len(counts))(*counts), len(counts), batch, m, k, 0, 0,
        )
        assert status == 0, f"C shim prototype returned {status}"
    stream.synchronize()
    actual = output.cpu().double()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    print(
        f"PASS: one float32 case, counts={counts}, batch={batch}, m={m}, k={k}; "
        f"max absolute error={(actual - expected).abs().max().item():.3g}; "
        f"GPU={torch.cuda.get_device_name(0)}, torch={torch.__version__}"
    )


if __name__ == "__main__":
    main()
```

</details>
