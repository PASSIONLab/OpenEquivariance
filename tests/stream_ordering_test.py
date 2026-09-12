# ruff: noqa: E731
import pytest
import torch

from e3nn import o3

from openequivariance import TensorProduct, TensorProductConv, TPProblem
from openequivariance._torch import extlib  # noqa: F401  loads the extension

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)

# ~50-100ms spin on modern GPUs: much longer than any launch latency here.
SLEEP_CYCLES = int(2e8)

NUM_W, NUM_FEATURES, M, K, N_PER_W = 4, 8, 16, 32, 256
N_BATCH = 1000


def _tpp():
    return TPProblem(
        o3.Irreps("1x2e"),
        o3.Irreps("1x3e"),
        o3.Irreps("1x2e"),
        [(0, 0, 0, "uvu", True)],
        shared_weights=False,
        internal_weights=False,
    )


class Case:
    """Callable op plus its input tensors; tensors[0] is the delayed input."""

    def __init__(self, fn, tensors):
        self.fn = fn
        self.tensors = tensors

    def __call__(self):
        return self.fn(*self.tensors)


def _build_group_gemm(device, gen):
    counts = torch.full((NUM_W,), N_PER_W, dtype=torch.int64)  # CPU tensor
    A = torch.randn(NUM_W, NUM_FEATURES, M, K, device=device, generator=gen)
    B = torch.randn(NUM_W * N_PER_W, NUM_FEATURES, K, device=device, generator=gen)
    fn = lambda A, B: torch.ops.libtorch_tp_jit.group_gemm(
        A, B, counts, NUM_W, NUM_FEATURES, M, K, 0
    )
    return Case(fn, [A, B])


def _build_tp(device, gen):
    tpp = _tpp()
    tp = TensorProduct(tpp)
    X = torch.rand(N_BATCH, tpp.irreps_in1.dim, device=device, generator=gen)
    Y = torch.rand(N_BATCH, tpp.irreps_in2.dim, device=device, generator=gen)
    W = torch.rand(N_BATCH, tpp.weight_numel, device=device, generator=gen)
    return Case(tp, [X, Y, W])


def _build_conv_atomic(device, gen):
    tpp = _tpp()
    conv = TensorProductConv(tpp, torch_op=True, deterministic=False).to(device)
    receivers = torch.tensor([0, 1, 1, 2], device=device, dtype=torch.long)
    senders = torch.tensor([1, 0, 2, 1], device=device, dtype=torch.long)
    X = torch.rand(3, tpp.irreps_in1.dim, device=device, generator=gen)
    Y = torch.rand(4, tpp.irreps_in2.dim, device=device, generator=gen)
    W = torch.rand(4, tpp.weight_numel, device=device, generator=gen)
    return Case(conv, [X, Y, W, receivers, senders])


BUILDERS = {
    "group_gemm": _build_group_gemm,
    "tp": _build_tp,
    "conv_atomic": _build_conv_atomic,
}


@pytest.fixture(params=list(BUILDERS))
def case_name(request):
    return request.param


def _build(case_name, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    return BUILDERS[case_name](device, gen)


def _delayed_run(case, tensor_device, stream):
    delayed = case.tensors[0]
    src = delayed.clone()
    torch.cuda.synchronize(tensor_device)
    delayed.zero_()
    torch.cuda.synchronize(tensor_device)

    with torch.cuda.device(tensor_device), torch.cuda.stream(stream):
        torch.cuda._sleep(SLEEP_CYCLES)
        delayed.copy_(src)

    with torch.cuda.stream(stream):
        out = case()
    torch.cuda.synchronize(tensor_device)
    return out


def test_ordering_current_device(case_name):
    case = _build(case_name, "cuda")
    torch.cuda.synchronize()
    ref = case()
    torch.cuda.synchronize()

    out = _delayed_run(case, 0, torch.cuda.Stream())
    torch.testing.assert_close(out, ref)


def test_cuda_graph_capture(case_name):
    case = _build(case_name, "cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            case()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = case()
    g.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(out, case())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2+ GPUs")
def test_ordering_nondefault_device(case_name):
    assert torch.cuda.current_device() == 0

    case = _build(case_name, "cuda:1")
    torch.cuda.synchronize(1)
    ref = case()  # already the report's mismatch scenario
    torch.cuda.synchronize(1)
    assert ref.device == case.tensors[0].device

    out = _delayed_run(case, 1, torch.cuda.Stream(device=1))
    torch.testing.assert_close(out, ref)
