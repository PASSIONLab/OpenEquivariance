import pytest
import torch

import openequivariance as oeq
from openequivariance.benchmark.problems import mace_problems


def _sorted_graph(node_count, nnz, gen):
    rows = torch.randint(0, node_count, (nnz,), device="cuda", generator=gen)
    cols = torch.randint(0, node_count, (nnz,), device="cuda", generator=gen)
    order = torch.argsort(rows * node_count + cols)
    rows, cols = rows[order], cols[order]
    sender_perm = torch.argsort(cols * node_count + rows)
    return rows, cols, sender_perm


@pytest.fixture(scope="module")
def problem():
    return mace_problems()[0]


@pytest.fixture(params=[False, True], ids=["atomic", "deterministic"])
def conv_and_inputs(request, problem):
    deterministic = request.param
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)

    node_count, nnz = 2000, 40000
    conv = oeq.TensorProductConv(problem, deterministic=deterministic)
    X = torch.randn(node_count, problem.irreps_in1.dim, device="cuda", generator=gen)
    Y = torch.randn(nnz, problem.irreps_in2.dim, device="cuda", generator=gen)
    W = torch.randn(nnz, problem.weight_numel, device="cuda", generator=gen)
    rows, cols, sender_perm = _sorted_graph(node_count, nnz, gen)
    if not deterministic:
        sender_perm = None
    G = torch.randn(node_count, problem.irreps_out.dim, device="cuda", generator=gen)
    return conv, deterministic, (X, Y, W, rows, cols, sender_perm, G)


def _fwd_bwd(conv, X, Y, W, rows, cols, sender_perm, G):
    out = conv(X, Y, W, rows, cols, sender_perm)
    gX, gY, gW = torch.autograd.grad((out * G).sum(), (X, Y, W))
    return out, gX, gY, gW


def _assert_close(actual, expected, deterministic):
    for a, e in zip(actual, expected):
        if deterministic:
            assert torch.equal(a, e)
        else:
            assert torch.allclose(a, e, atol=1e-4, rtol=1e-4)


def test_eager_cuda_graph_capture(conv_and_inputs):
    conv, deterministic, (X, Y, W, rows, cols, sender_perm, G) = conv_and_inputs
    X, Y, W = (t.clone().requires_grad_(True) for t in (X, Y, W))

    reference = [
        t.detach().clone() for t in _fwd_bwd(conv, X, Y, W, rows, cols, sender_perm, G)
    ]

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _fwd_bwd(conv, X, Y, W, rows, cols, sender_perm, G)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_outputs = _fwd_bwd(conv, X, Y, W, rows, cols, sender_perm, G)

    for _ in range(3):
        for t in static_outputs:
            t.zero_()
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(static_outputs, reference, deterministic)


def test_compile_reduce_overhead(conv_and_inputs):
    conv, deterministic, (X, Y, W, rows, cols, sender_perm, G) = conv_and_inputs
    X, Y, W = (t.clone().requires_grad_(True) for t in (X, Y, W))

    reference = [
        t.detach().clone() for t in _fwd_bwd(conv, X, Y, W, rows, cols, sender_perm, G)
    ]

    compiled = torch.compile(conv, mode="reduce-overhead")

    def step():
        return _fwd_bwd(compiled, X, Y, W, rows, cols, sender_perm, G)

    for _ in range(4):
        outputs = [t.detach().clone() for t in step()]
        torch.cuda.synchronize()
        _assert_close(outputs, reference, deterministic)


def test_concurrent_streams_share_no_state(conv_and_inputs):
    conv, deterministic, (X, Y, W, rows, cols, sender_perm, G) = conv_and_inputs
    X2, Y2, W2, G2 = (t.flip(0).contiguous() for t in (X, Y, W, G))

    def run(X, Y, W, G):
        X, Y, W = (t.clone().requires_grad_(True) for t in (X, Y, W))
        return [
            t.detach().clone()
            for t in _fwd_bwd(conv, X, Y, W, rows, cols, sender_perm, G)
        ]

    ref1 = run(X, Y, W, G)
    ref2 = run(X2, Y2, W2, G2)
    torch.cuda.synchronize()

    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    for _ in range(5):
        with torch.cuda.stream(s1):
            out1 = run(X, Y, W, G)
        with torch.cuda.stream(s2):
            out2 = run(X2, Y2, W2, G2)
        torch.cuda.synchronize()
        _assert_close(out1, ref1, deterministic)
        _assert_close(out2, ref2, deterministic)
