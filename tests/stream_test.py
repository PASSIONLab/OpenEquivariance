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

    def __call__(self) -> Any:
        return self.func(*self.buffers)


cuda = torch.device("cuda")


@pytest.fixture
def N():
    return 1000


@pytest.fixture
def tpp(X_ir, Y_ir, Z_ir, instructions):
    X_ir = o3.Irreps("1x2e")
    Y_ir = o3.Irreps("1x3e")
    Z_ir = o3.Irreps("1x2e")
    instructions = [(0, 0, 0, "uvu", True)]
    return TPProblem(
        X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False
    )


@pytest.fixture
def gen():
    return torch.Generator(device="cuda")


@pytest.fixture
def X(N, tpp, gen):
    return torch.rand(N, tpp.irreps_in1.dim, device="cuda", generator=gen)


@pytest.fixture
def Y(N, tpp, gen):
    return torch.rand(N, tpp.irreps_in2.dim, device="cuda", generator=gen)


@pytest.fixture
def oeq_tp_fwd(tpp, N, X, Y, gen):
    tp_oeq = TensorProduct(tpp)
    W = torch.rand(N, tpp.weight_numel, device="cuda", generator=gen)
    return Executable(tp_oeq, (X, Y, W))


@pytest.fixture
def oeq_tp_bwd(tpp, N, X, Y, gen):
    tp_oeq = TensorProduct(tpp)
    W = torch.rand(N, tpp.weight_numel, device="cuda", generator=gen)

    # Set up backward-executing callable
    def backward_fn(X, Y, W):
        X.requires_grad_(True)
        Y.requires_grad_(True)
        W.requires_grad_(True)
        output = tp_oeq(X, Y, W).sum()
        output.backward()
        return output

    return Executable(backward_fn, (X, Y, W))


@pytest.fixture
def oeq_tp_double_bwd(tpp, N, X, Y, gen):
    tp_oeq = TensorProduct(tpp)
    W = torch.rand(N, tpp.weight_numel, device="cuda", generator=gen)

    def double_backward_fn(X, Y, W):
        # Forward pass
        X.requires_grad_(True)
        Y.requires_grad_(True)
        W.requires_grad_(True)

        # First forward
        out = tp_oeq(X, Y, W)
        out_grad = out.clone().detach().requires_grad_(True)

        # First backward (compute gradients w.r.t inputs)
        in1_grad, in2_grad, w_grad = torch.autograd.grad(
            outputs=out,
            inputs=(X, Y, W),
            grad_outputs=out_grad,
            create_graph=True,
        )

        # Dummy loss to propagate second backward
        dummy = torch.norm(in1_grad) + torch.norm(in2_grad) + torch.norm(w_grad)

        # Second backward
        dummy_grad = torch.tensor(1.0, device="cuda")
        dummy.backward(
            dummy_grad,
            retain_graph=True,
            inputs=(out_grad, X, Y, W),
        )

        return dummy

    return Executable(double_backward_fn, (X, Y, W))


@pytest.fixture
def oeq_conv_fwd(X_ir, Y_ir, tpp, gen):
    node_ct, nonzero_ct = 3, 4

    # Receiver, sender indices for message passing GNN
    edge_index = EdgeIndex(
        [
            [0, 1, 1, 2],  # Receiver
            [1, 0, 2, 1],  # Sender
        ],
        device="cuda",
        dtype=torch.long,
    )

    X = torch.rand(node_ct, X_ir.dim, device="cuda", generator=gen)
    Y = torch.rand(nonzero_ct, Y_ir.dim, device="cuda", generator=gen)
    W = torch.rand(nonzero_ct, tpp.weight_numel, device="cuda", generator=gen)

    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    return Executable(tp_conv, (X, Y, W, edge_index[0], edge_index[1]))


@pytest.fixture
def oeq_conv_bwd(X_ir, Y_ir, tpp, gen):
    node_ct, nonzero_ct = 3, 4

    # Receiver, sender indices for message passing GNN
    edge_index = EdgeIndex(
        [
            [0, 1, 1, 2],  # Receiver
            [1, 0, 2, 1],  # Sender
        ],
        device="cuda",
        dtype=torch.long,
    )

    X = torch.rand(node_ct, X_ir.dim, device="cuda", generator=gen)
    Y = torch.rand(nonzero_ct, Y_ir.dim, device="cuda", generator=gen)
    W = torch.rand(nonzero_ct, tpp.weight_numel, device="cuda", generator=gen)

    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    # Set up backward-executing callable
    def backward_fn(X, Y, W, receivers, senders):
        X.requires_grad_(True)
        Y.requires_grad_(True)
        W.requires_grad_(True)
        output = tp_conv(
            X, Y, W, receivers, senders
        ).sum()  # Scalar output for backward
        output.backward()
        return output

    return Executable(backward_fn, (X, Y, W, edge_index[0], edge_index[1]))


@pytest.fixture
def oeq_conv_double_bwd(X_ir, Y_ir, tpp, gen):
    node_ct, nonzero_ct = 3, 4

    # Receiver, sender indices for message passing GNN
    edge_index = EdgeIndex(
        [
            [0, 1, 1, 2],  # Receiver
            [1, 0, 2, 1],
        ],
        device="cuda",
        dtype=torch.long,
    )

    X = torch.rand(node_ct, X_ir.dim, device="cuda", generator=gen)
    Y = torch.rand(nonzero_ct, Y_ir.dim, device="cuda", generator=gen)
    W = torch.rand(nonzero_ct, tpp.weight_numel, device="cuda", generator=gen)

    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    def double_backward_fn(X, Y, W, receivers, senders):
        # First forward pass
        X.requires_grad_(True)
        Y.requires_grad_(True)
        W.requires_grad_(True)

        out = tp_conv(X, Y, W, receivers, senders)
        out_grad = out.clone().detach().requires_grad_(True)

        # First backward (gradients w.r.t inputs)
        in1_grad, in2_grad, w_grad = torch.autograd.grad(
            outputs=out,
            inputs=(X, Y, W),
            grad_outputs=out_grad,
            create_graph=True,
        )

        # Dummy loss for second backward
        dummy = torch.norm(in1_grad) + torch.norm(in2_grad) + torch.norm(w_grad)

        # Second backward
        dummy_grad = torch.tensor(1.0, device="cuda")
        dummy.backward(
            dummy_grad,
            retain_graph=True,
            inputs=(out_grad, X, Y, W),
        )

        return dummy

    return Executable(double_backward_fn, (X, Y, W, edge_index[0], edge_index[1]))


@pytest.fixture(
    params=[
        "oeq_tp_fwd",
        "oeq_tp_bwd",
        "oeq_tp_double_bwd",
        "oeq_conv_fwd",
        "oeq_conv_bwd",
        "oeq_conv_double_bwd",
    ],
)
def executable(request):
    yield request.getfixturevalue(request.param)


def test_separate_streams(tmp_path, executable: Executable):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        streams = [1, 2]
        for priority in streams:
            s = torch.cuda.Stream(device=cuda, priority=priority)
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
