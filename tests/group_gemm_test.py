import importlib
import subprocess
import sys

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required"
)


@pytest.fixture(scope="module")
def group_gemm():
    importlib.import_module("openequivariance._torch.symmetric_contraction")
    return torch.ops.libtorch_tp_jit.group_gemm


def reference(A, B, counts, inner):
    pieces = []
    offset = 0
    for i, n in enumerate(counts):
        if inner == 0:
            pieces.append(torch.einsum("bmk,nbk->nbm", A[i], B[offset : offset + n]))
        else:
            pieces.append(
                torch.einsum(
                    "nbm,nbk->bmk", A[offset : offset + n], B[offset : offset + n]
                )
            )
        offset += n
    if inner == 0:
        return torch.cat(pieces, dim=0)
    return torch.stack(pieces, dim=0)


def make_input(shape, dtype, layout="contiguous", device="cuda"):
    values = torch.arange(1, 1 + torch.Size(shape).numel(), dtype=dtype, device="cpu")
    values = (values.remainder(17) - 8).reshape(shape) / 8
    if layout == "offset":
        storage = torch.empty(values.numel() + 7, dtype=dtype, device=device)
        result = storage[3 : 3 + values.numel()].view(shape)
        result.copy_(values)
        return result
    if layout == "noncontiguous":
        storage = torch.empty((*shape, 2), dtype=dtype, device=device)
        result = storage[..., 0]
        result.copy_(values)
        return result
    return values.to(device)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("inner", [0, 1])
@pytest.mark.parametrize("layout", ["contiguous", "offset", "noncontiguous"])
def test_group_gemm_matches_reference(group_gemm, dtype, inner, layout):
    counts = [2, 0, 3]
    batch, m, k = 2, 3, 4
    A_shape = (len(counts), batch, m, k) if inner == 0 else (sum(counts), batch, m)
    A = make_input(A_shape, dtype, layout)
    B = make_input((sum(counts), batch, k), dtype, layout)
    expected = reference(A.cpu().double(), B.cpu().double(), counts, inner)
    counts_tensor = torch.tensor([n for n in counts for _ in range(2)], device="cpu")[
        ::2
    ]

    actual = group_gemm(A, B, counts_tensor, len(counts), batch, m, k, inner)

    torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-5, atol=1e-5)
    assert actual.dtype == dtype
    assert actual.device == A.device


@pytest.mark.parametrize("inner", [0, 1])
@pytest.mark.parametrize(
    "counts,batch,m,k",
    [
        ([0, 0], 2, 3, 4),
        ([], 2, 3, 4),
        ([2, 0, 3], 0, 3, 4),
        ([2, 0, 3], 2, 0, 4),
        ([2, 0, 3], 2, 3, 0),
    ],
)
def test_group_gemm_empty_dimensions(group_gemm, inner, counts, batch, m, k):
    A_shape = (len(counts), batch, m, k) if inner == 0 else (sum(counts), batch, m)
    A = torch.empty(A_shape, device="cuda", dtype=torch.float64)
    B = torch.empty((sum(counts), batch, k), device="cuda", dtype=torch.float64)
    actual = group_gemm(
        A,
        B,
        torch.tensor(counts, dtype=torch.int64, device="cpu"),
        len(counts),
        batch,
        m,
        k,
        inner,
    )
    expected_shape = (
        (sum(counts), batch, m) if inner == 0 else (len(counts), batch, m, k)
    )
    assert actual.shape == expected_shape
    torch.testing.assert_close(actual, torch.zeros_like(actual))


@pytest.mark.parametrize("inner", [0, 1])
def test_group_gemm_backward(group_gemm, inner):
    counts = [2, 0, 3]
    batch, m, k = 2, 3, 4
    A_shape = (len(counts), batch, m, k) if inner == 0 else (sum(counts), batch, m)
    A = make_input(A_shape, torch.float64, "noncontiguous").requires_grad_()
    B = make_input((sum(counts), batch, k), torch.float64, "offset").requires_grad_()
    A_ref = A.detach().cpu().requires_grad_()
    B_ref = B.detach().cpu().requires_grad_()
    expected = reference(A_ref, B_ref, counts, inner)
    actual = group_gemm(
        A, B, torch.tensor(counts, device="cpu"), len(counts), batch, m, k, inner
    )
    grad = make_input(actual.shape, torch.float64, "noncontiguous")

    actual_grads = torch.autograd.grad(actual, (A, B), grad)
    expected_grads = torch.autograd.grad(expected, (A_ref, B_ref), grad.cpu())

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad.cpu(), expected_grad, rtol=1e-10, atol=1e-10
        )


def test_group_gemm_current_stream(group_gemm):
    counts = torch.tensor([2, 0, 3], device="cpu")
    A = torch.zeros((3, 2, 3, 4), device="cuda")
    B = torch.zeros((5, 2, 4), device="cuda")
    group_gemm(A, B, counts, 3, 2, 3, 4, 0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if not torch.version.hip:
            torch.cuda._sleep(1_000_000)
        A.fill_(2)
        B.fill_(3)
        actual = group_gemm(A, B, counts, 3, 2, 3, 4, 0).clone()
    stream.synchronize()
    torch.testing.assert_close(actual, torch.full_like(actual, 24))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two GPUs are required")
def test_group_gemm_device_guard(group_gemm):
    with torch.cuda.device(0):
        A = torch.full((1, 2, 3, 4), 2.0, device="cuda:1")
        B = torch.full((2, 2, 4), 3.0, device="cuda:1")
        actual = group_gemm(A, B, torch.tensor([2], device="cpu"), 1, 2, 3, 4, 0)
        assert torch.cuda.current_device() == 0
        assert actual.device == torch.device("cuda:1")
        torch.testing.assert_close(actual, torch.full_like(actual, 24))
        with pytest.raises(RuntimeError, match="same device"):
            group_gemm(
                A, B.to("cuda:0"), torch.tensor([2], device="cpu"), 1, 2, 3, 4, 0
            )


@pytest.mark.parametrize("counts", [[-1, 0, 6], [2, 0, 2], [2, 0, 4]])
def test_group_gemm_rejects_invalid_counts(group_gemm, counts):
    A = torch.empty((3, 2, 3, 4), device="cuda")
    B = torch.empty((5, 2, 4), device="cuda")
    with pytest.raises(RuntimeError, match="ragged_counts"):
        group_gemm(A, B, torch.tensor(counts, device="cpu"), 3, 2, 3, 4, 0)


def test_group_gemm_requires_cpu_counts(group_gemm):
    A = torch.empty((3, 2, 3, 4), device="cuda")
    B = torch.empty((5, 2, 4), device="cuda")
    with pytest.raises(RuntimeError, match="ragged_counts must be on the CPU"):
        group_gemm(A, B, torch.tensor([2, 0, 3], device="cuda"), 3, 2, 3, 4, 0)


@pytest.mark.parametrize("inner", [0, 1])
@pytest.mark.parametrize(
    "counts,batch,m,k",
    [
        ([1, 0, 5], 1, 1, 7),
        ([1, 0, 5], 3, 7, 1),
        ([1, 0, 5], 3, 1, 1),
        ([1, 7, 0, 13], 4, 17, 29),
    ],
)
def test_group_gemm_varied_shapes(group_gemm, inner, counts, batch, m, k):
    A_shape = (len(counts), batch, m, k) if inner == 0 else (sum(counts), batch, m)
    A = make_input(A_shape, torch.float64, "offset")
    B = make_input((sum(counts), batch, k), torch.float64, "offset")
    actual = group_gemm(
        A, B, torch.tensor(counts, device="cpu"), len(counts), batch, m, k, inner
    )
    expected = reference(A.cpu(), B.cpu(), counts, inner)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("inner", [0, 1])
def test_group_gemm_double_backward(group_gemm, inner):
    counts = torch.tensor([1, 0, 2], device="cpu")
    A_shape = (3, 2, 2, 3) if inner == 0 else (3, 2, 2)
    A = make_input(A_shape, torch.float64).requires_grad_()
    B = make_input((3, 2, 3), torch.float64).requires_grad_()

    def operation(A, B):
        return group_gemm(A, B, counts, 3, 2, 2, 3, inner)

    assert torch.autograd.gradgradcheck(operation, (A, B), fast_mode=True)


@pytest.mark.parametrize("inner", [0, 1])
def test_group_gemm_graph_replay(group_gemm, inner):
    counts = [2, 0, 3]
    counts_tensor = torch.tensor(counts, device="cpu")
    A_shape = (3, 2, 3, 4) if inner == 0 else (5, 2, 3)
    A = make_input(A_shape, torch.float64)
    B = make_input((5, 2, 4), torch.float64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            group_gemm(A, B, counts_tensor, 3, 2, 3, 4, inner)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = group_gemm(A, B, counts_tensor, 3, 2, 3, 4, inner)

    for scale in (2, 3):
        A.mul_(scale)
        B.add_(0.25)
        graph.replay()
        expected = reference(A.cpu(), B.cpu(), counts, inner)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("inner", [0, 1])
def test_group_gemm_compile(group_gemm, inner):
    counts = [2, 0, 3]
    counts_tensor = torch.tensor(counts, device="cpu")
    A_shape = (3, 2, 3, 4) if inner == 0 else (5, 2, 3)
    A = make_input(A_shape, torch.float64).requires_grad_()
    B = make_input((5, 2, 4), torch.float64).requires_grad_()

    def operation(A, B, counts):
        return group_gemm(A, B, counts, 3, 2, 3, 4, inner)

    compiled = torch.compile(operation, fullgraph=True)
    actual = compiled(A, B, counts_tensor)
    A_ref = A.detach().cpu().requires_grad_()
    B_ref = B.detach().cpu().requires_grad_()
    expected = reference(A_ref, B_ref, counts, inner)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-10, atol=1e-10)

    actual_grads = torch.autograd.grad(actual.sum(), (A, B))
    expected_grads = torch.autograd.grad(expected.sum(), (A_ref, B_ref))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad.cpu(), expected_grad, rtol=1e-10, atol=1e-10
        )

    next_counts = [0, 3, 2]
    next_actual = compiled(A, B, torch.tensor(next_counts, device="cpu"))
    next_expected = reference(A_ref, B_ref, next_counts, inner)
    torch.testing.assert_close(next_actual.cpu(), next_expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("inner", [0, 1])
def test_group_gemm_aoti(group_gemm, inner, tmp_path):
    import openequivariance

    class Model(torch.nn.Module):
        def forward(self, A, B, counts):
            return group_gemm(A, B, counts, 3, 2, 3, 4, inner)

    counts = [2, 0, 3]
    counts_tensor = torch.tensor(counts, device="cpu")
    A_shape = (3, 2, 3, 4) if inner == 0 else (5, 2, 3)
    A = make_input(A_shape, torch.float64)
    B = make_input((5, 2, 4), torch.float64)
    exported = torch.export.export(Model(), (A, B, counts_tensor), strict=False)
    package_path = torch._inductor.aoti_compile_and_package(
        exported, package_path=str(tmp_path / "group_gemm.pt2")
    )
    inputs_path = tmp_path / "inputs.pt"
    torch.save(
        (A.cpu(), B.cpu(), counts_tensor, reference(A.cpu(), B.cpu(), counts, inner)),
        inputs_path,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
import torch._inductor.codecache

torch.ops.load_library(sys.argv[1])
model = torch._inductor.aoti_load_package(sys.argv[2])
A, B, counts, expected = torch.load(sys.argv[3], weights_only=True)
actual = model(A.cuda(), B.cuda(), counts)
torch.testing.assert_close(actual.cpu(), expected, rtol=1e-10, atol=1e-10)
""",
            openequivariance.torch_ext_so_path(),
            package_path,
            str(inputs_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
