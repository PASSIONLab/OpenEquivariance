import json
from dataclasses import dataclass
from typing import Callable, Tuple, Any
import logging

import pytest
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from e3nn import o3
from torch_geometric import EdgeIndex

from openequivariance import TensorProduct, TensorProductConv, TPProblem


@dataclass
class Executable:
    func: Callable[..., Any]
    buffers: Tuple[torch.Tensor, ...]
    label: str

    def __call__(self) -> Any:
        return self.func(*self.buffers)


cuda = torch.device("cuda")


@pytest.fixture
def N():
    return 1000


@pytest.fixture
def X_ir():
    return o3.Irreps("1x2e")


@pytest.fixture
def Y_ir():
    return o3.Irreps("1x3e")


@pytest.fixture
def Z_ir():
    return o3.Irreps("1x2e")


@pytest.fixture
def instructions():
    return [(0, 0, 0, "uvu", True)]


@pytest.fixture
def tpp(X_ir, Y_ir, Z_ir, instructions):
    return TPProblem(
        X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False
    )


@pytest.fixture
def gen():
    return torch.Generator(device="cuda")


@pytest.fixture
def X(N, X_ir, gen):
    return torch.rand(N, X_ir.dim, device="cuda", generator=gen)


@pytest.fixture
def Y(N, Y_ir, gen):
    return torch.rand(N, Y_ir.dim, device="cuda", generator=gen)


@pytest.fixture
def W(N, e3nn_tp, gen):
    return torch.rand(N, e3nn_tp.weight_numel, device="cuda", generator=gen)


@pytest.fixture
def torch_matmul(N):
    A = torch.empty((N, N), device=cuda).normal_(0.0, 1.0).to(device=cuda)
    B = torch.empty((N, N), device=cuda).normal_(0.0, 1.0).to(device=cuda)
    return Executable(torch.matmul, (A, B), "torch_matmul")


@pytest.fixture
def e3nn_tp(X_ir, Y_ir, Z_ir, instructions):
    return o3.TensorProduct(
        X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False
    ).to("cuda")


@pytest.fixture
def e3nn_tp_exec(e3nn_tp, X, Y, W):
    return Executable(e3nn_tp, (X, Y, W), "e3nn_tensor_product")


@pytest.fixture
def oeq_tp_exec(tpp, X, Y, W):
    tp_oeq = TensorProduct(tpp)
    return Executable(tp_oeq, (X, Y, W), "oeq_tensor_product")


@pytest.fixture
def oeq_conv_exec(X_ir, Y_ir, tpp, gen):
    node_ct, nonzero_ct = 3, 4

    # Receiver, sender indices for message passing GNN
    edge_index = EdgeIndex(
        [
            [0, 1, 1, 2],  # Receiver
            [1, 0, 2, 1],
        ],  # Sender
        device="cuda",
        dtype=torch.long,
    )

    X = torch.rand(node_ct, X_ir.dim, device="cuda", generator=gen)
    Y = torch.rand(nonzero_ct, Y_ir.dim, device="cuda", generator=gen)
    W = torch.rand(nonzero_ct, tpp.weight_numel, device="cuda", generator=gen)

    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    return Executable(tp_conv, (X, Y, W, edge_index[0], edge_index[1]), "oeq_conv")


@pytest.fixture(
    params=["torch_matmul", "e3nn_tp_exec", "oeq_tp_exec", "oeq_conv_exec"],
)
def executable(request):
    yield request.getfixturevalue(request.param)


def test_separate_streams(tmp_path, executable: Executable):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        streams = [1, 2]
        for priority in streams:
            s = torch.cuda.Stream(
                device=cuda, priority=priority
            )  # Create a new stream.
            with torch.cuda.stream(s):
                with record_function(f"executable_{priority}"):
                    for _ in range(5):
                        executable()

    prof.export_chrome_trace(str(tmp_path))

    trace = None
    with open(tmp_path, "r") as f:
        trace = json.load(f)

    relevant_events = []
    for event in trace["traceEvents"]:
        # print(event)
        if "gpu_user_annotation" == event.get("cat") and "executable_" in event.get(
            "name", ""
        ):
            relevant_events.append(event)

    keys = [e["name"] for e in relevant_events]
    values = [e["tid"] for e in relevant_events]

    logger = logging.getLogger()
    logger.debug(msg=keys)
    logger.debug(msg=values)

    assert len(keys) == len(streams)
    assert len(values) == len(streams)

    assert len(set(keys)) == len(set(values)), "The CUDA streams are not unique"
