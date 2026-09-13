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
