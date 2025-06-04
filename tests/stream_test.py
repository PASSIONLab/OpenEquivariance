import json
from dataclasses import dataclass
from typing import Callable, Tuple, Any
import logging

import pytest
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from e3nn import o3

from openequivariance import TensorProduct, TPProblem


@dataclass
class Executable:
    func: Callable[..., Any]
    buffers: Tuple[torch.Tensor, ...]
    label: str

    def __call__(self) -> Any:
        return self.func(*self.buffers)


cuda = torch.device("cuda")
N = 10000

A = torch.empty((N, N), device=cuda).normal_(0.0, 1.0).to(device=cuda)
B = torch.empty((N, N), device=cuda).normal_(0.0, 1.0).to(device=cuda)
torch_matmul = Executable(torch.matmul, (A, B), "torch_matmul")

gen = torch.Generator(device="cuda")
X_ir, Y_ir, Z_ir = o3.Irreps("1x2e"), o3.Irreps("1x3e"), o3.Irreps("1x2e")
X = torch.rand(N, X_ir.dim, device="cuda", generator=gen)
Y = torch.rand(N, Y_ir.dim, device="cuda", generator=gen)

instructions = [(0, 0, 0, "uvu", True)]

tp_e3nn = o3.TensorProduct(
    X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False
).to("cuda")

W = torch.rand(N, tp_e3nn.weight_numel, device="cuda", generator=gen)

e3nn_tensor_product = Executable(tp_e3nn, (X, Y, W), "e3nn_tensor_product")

tpp_oeq = TPProblem(
    X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False
)
tp_oeq = TensorProduct(tpp_oeq)

oeq_tensor_product = Executable(tp_oeq, (X, Y, W), "oeq_tensor_product")


@pytest.fixture(
    params=[torch_matmul, e3nn_tensor_product, oeq_tensor_product],
    ids=lambda x: x.label,
)
def executable(request):
    return request.param


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
