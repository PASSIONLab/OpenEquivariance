# ruff: noqa: E731
import json
from dataclasses import dataclass
from typing import Callable, Tuple, Any, NamedTuple
import logging

import pytest
from pytest_check import check

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from e3nn import o3
from torch_geometric import EdgeIndex

from openequivariance import TensorProduct, TensorProductConv, TPProblem


class KernelExpectation(NamedTuple):
    kernel_name: str
    expected_appearances: int


KE = KernelExpectation


@dataclass
class Executable:
    func: Callable[..., Any]
    buffers: Tuple[torch.Tensor, ...]
    kernel_expectations: list[KernelExpectation]

    def __call__(self) -> Any:
        return self.func(*self.buffers)


cuda = torch.device("cuda")


@pytest.fixture
def gen():
    return torch.Generator(device="cuda")


@pytest.fixture
def N():
    return 1000


@pytest.fixture
def edge_index():
    return EdgeIndex(
        data=[
            [0, 1, 1, 2],  # Receiver
            [1, 0, 2, 1],  # Sender
        ],
        sparse_size=(3, 4),
        device="cuda",
        dtype=torch.long,
    )


@pytest.fixture
def tpp():
    X_ir = o3.Irreps("1x2e")
    Y_ir = o3.Irreps("1x3e")
    Z_ir = o3.Irreps("1x2e")
    instructions = [(0, 0, 0, "uvu", True)]
    return TPProblem(
        X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False
    )


@pytest.fixture
def tp_buffers(N, tpp, gen):
    X = torch.rand(N, tpp.irreps_in1.dim, device="cuda", generator=gen)
    Y = torch.rand(N, tpp.irreps_in2.dim, device="cuda", generator=gen)
    W = torch.rand(N, tpp.weight_numel, device="cuda", generator=gen)
    return (X, Y, W)


@pytest.fixture
def conv_buffers(edge_index, tpp, gen):
    X = torch.rand(
        edge_index.num_rows, tpp.irreps_in1.dim, device="cuda", generator=gen
    )
    Y = torch.rand(
        edge_index.num_cols, tpp.irreps_in2.dim, device="cuda", generator=gen
    )
    W = torch.rand(edge_index.num_cols, tpp.weight_numel, device="cuda", generator=gen)
    return (X, Y, W, edge_index[0], edge_index[1])


@pytest.fixture
def oeq_tp_fwd(tpp, tp_buffers):
    tp_oeq = TensorProduct(tpp)
    return Executable(tp_oeq, tp_buffers, [KE("forward", 1)])


@pytest.fixture
def oeq_tp_bwd(tpp, tp_buffers):
    tp_oeq = TensorProduct(tpp)

    # Set up backward-executing callable
    def backward_fn(X, Y, W):
        X.requires_grad_(True)
        Y.requires_grad_(True)
        W.requires_grad_(True)
        output = tp_oeq(X, Y, W).sum()
        output.backward()
        return output

    return Executable(backward_fn, tp_buffers, [KE("forward", 1), KE("backward", 1)])


@pytest.fixture
def oeq_tp_double_bwd(tpp, tp_buffers):
    tp_oeq = TensorProduct(tpp)

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

    return Executable(
        double_backward_fn,
        tp_buffers,
        [
            KE("forward", 1),
            KE("backward", 1),
            KE("double_backward_A", 1),
            KE("double_backward_B", 1),
        ],
    )


@pytest.fixture
def oeq_conv_atomic_fwd(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    return Executable(tp_conv, conv_buffers, [KE("forward", 1)])


@pytest.fixture
def oeq_conv_atomic_bwd(tpp, conv_buffers):
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

    return Executable(
        backward_fn,
        conv_buffers,
        [
            KE("forward", 1),
            KE("backward", 1),
        ],
    )


@pytest.fixture
def oeq_conv_atomic_double_bwd(tpp, conv_buffers):
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

    return Executable(
        double_backward_fn,
        conv_buffers,
        [
            KE("forward", 1),
            KE("backward", 1),
            KE("double_backward_A", 1),
            KE("double_backward_B", 1),
        ],
    )


@pytest.fixture
def oeq_conv_det_fwd(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    return Executable(tp_conv, conv_buffers, [KE("forward", 1), KE("fixup_forward", 1)])


@pytest.fixture
def oeq_conv_det_bwd(tpp, conv_buffers):
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

    return Executable(
        backward_fn,
        conv_buffers,
        [
            KE("forward", 1),
            KE("fixup_forward", 1),
            KE("backward", 1),
            KE("fixup_backward", 1),
        ],
    )


@pytest.fixture
def oeq_conv_det_double_bwd(tpp, conv_buffers):
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

    return Executable(
        double_backward_fn,
        conv_buffers,
        [
            KE("forward", 1),
            KE("fixup_forward", 2),
            KE("backward", 1),
            KE("fixup_backward", 1),
            KE("double_backward_A", 1),
            KE("double_backward_B", 1),
            KE("fixup_double_backwardB", 1),
        ],
    )


@pytest.fixture(
    params=[
        "oeq_tp_fwd",
        "oeq_tp_bwd",
        "oeq_tp_double_bwd",
        "oeq_conv_atomic_fwd",
        "oeq_conv_atomic_bwd",
        "oeq_conv_atomic_double_bwd",
        "oeq_conv_det_fwd",
        "oeq_conv_det_bwd",
        "oeq_conv_det_double_bwd",
    ],
)
def executable(request):
    yield request.getfixturevalue(request.param)


def test_separate_streams(request, tmp_path, executable: Executable):
    COUNT = 5
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        streams = [-1, -2]
        for priority in streams:
            s = torch.cuda.Stream(device=cuda, priority=priority)
            with torch.cuda.stream(s):
                with record_function(f"executable_{priority}"):
                    for _ in range(COUNT):
                        executable()

    prof.export_chrome_trace(str(tmp_path / "trace"))

    trace = None
    with open(tmp_path / "trace", "r") as f:
        trace = json.load(f)

    gpu_annotations = []
    for event in trace["traceEvents"]:
        if "gpu_user_annotation" == event.get("cat") and "executable_" in event.get(
            "name", ""
        ):
            gpu_annotations.append(event)

    names = [x["name"] for x in gpu_annotations]
    tids = [x["tid"] for x in gpu_annotations]

    logger = logging.getLogger()
    logger.debug(msg=names)
    logger.debug(msg=tids)

    with check:
        assert len(names) == len(streams)
        assert len(tids) == len(streams)
        assert len(set(tids)) == len(set(names)), "The CUDA streams are not unique"

    for tid in set(tids):
        for kernel_expectation in executable.kernel_expectations:
            criteria = (
                lambda event: (event.get("cat") == "kernel")
                and (event.get("name", "").startswith(kernel_expectation.kernel_name))
                and (event.get("tid") == tid)
            )
            matching = list(filter(criteria, trace["traceEvents"]))
            num_matching = len(matching)
            with check:
                assert (
                    num_matching == COUNT * kernel_expectation.expected_appearances
                ), f"{tid}_{kernel_expectation.kernel_name}"
