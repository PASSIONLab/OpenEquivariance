from dataclasses import dataclass
from typing import Callable, Any, Literal, get_args

import pytest
import torch
from e3nn import o3
from torch_geometric import EdgeIndex

from openequivariance import TPProblem
from openequivariance.implementations.dtype_enum import (
    enum_to_torch_dtype_mapping,
    dtype_to_enum_mapping,
)

from openequivariance.benchmark.TestBenchmarkSuite import Direction

Operation = Literal["TP", "TP_Conv"]
BufferName = Literal[
    "x",
    "x_grad",
    "x_dgrad",
    "y",
    "y_grad",
    "y_dgrad",
    "w",
    "w_grad",
    "w_dgrad",
    "z",
    "z_grad",
    "z_dgrad",
    "rows",
    "cols",
    "transpose_perm",
]


@dataclass
class Executable:
    func: Callable
    buffers: dict[BufferName, torch.Tensor]

    def __call__(self) -> Any:
        return self.func(**self.buffers)


@dataclass
class OperationSignature:
    inputs: set[BufferName]
    outputs: set[BufferName]


_OP = OperationSignature


@pytest.fixture(params=get_args(Direction))
def direction(request) -> Direction:
    return request.param


@pytest.fixture(params=get_args(Operation))
def operation(request) -> Operation:
    return request.param


def sig_tp_fwd():
    return _OP({"x", "y", "w"}, {"z"})


def sig_tp_bwd():
    return _OP({"x", "y", "w", "z_grad"}, {"w_grad", "x_grad", "y_grad"})


def sig_tp_dbl_bwd():
    return _OP(
        {"x", "x_dgrad", "y", "y_dgrad", "w", "w_dgrad", "z_grad"},
        {"x_grad", "y_grad", "w_grad", "z_dgrad"},
    )


def sig_tp(direction: Direction):
    match direction:
        case "forward":
            return sig_tp_fwd()
        case "backward":
            return sig_tp_bwd()
        case "double_backward":
            return sig_tp_dbl_bwd()
        case _:
            raise KeyError("Not a valid direction")


def sig_conv_fwd():
    return _OP({"x", "y", "w", "rows", "cols", "transpose_perm"}, {"z"})


def sig_conv_bwd():
    return _OP(
        {"x", "y", "w", "z_grad", "rows", "cols", "transpose_perm"},
        {"x_grad", "y_grad", "w_grad"},
    )


def sig_conv_dbl_bwd():
    return _OP(
        {
            "x",
            "x_dgrad",
            "y",
            "y_dgrad",
            "w",
            "w_dgrad",
            "z_grad",
            "rows",
            "cols",
            "transpose_perm",
        },
        {"x_grad", "y_grad", "w_grad", "z_dgrad"},
    )


def sig_conv(direction: Direction):
    match direction:
        case "forward":
            return sig_conv_fwd()
        case "backward":
            return sig_conv_bwd()
        case "double_backward":
            return sig_conv_bwd()
        case _:
            raise KeyError("Not a valid direction")


@pytest.fixture
def cuda():
    """
    Returns the default torch device
    """
    return torch.device("cuda")


@pytest.fixture
def gen(cuda):
    return torch.Generator(device=cuda)


@pytest.fixture
def num_batch() -> int:
    return 1000


@pytest.fixture
def edge_index() -> EdgeIndex:
    """
    A simple Edge_Index
    """
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
def tpp() -> TPProblem:
    """
    A simple Tensor Product Problem
    """
    X_ir = o3.Irreps("1x2e")
    Y_ir = o3.Irreps("1x3e")
    Z_ir = o3.Irreps("1x2e")
    instructions = [(0, 0, 0, "uvu", True)]
    return TPProblem(
        X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False
    )


def tp_shapes(num_batch: int, tpp: TPProblem) -> dict[BufferName, tuple[int, ...]]:
    return {
        "x": (num_batch, tpp.irreps_in1.dim),
        "x_grad": (num_batch, tpp.irreps_in1.dim),
        "x_dgrad": (num_batch, tpp.irreps_in1.dim),
        "y": (num_batch, tpp.irreps_in2.dim),
        "y_grad": (num_batch, tpp.irreps_in2.dim),
        "y_dgrad": (num_batch, tpp.irreps_in2.dim),
        "w": (tpp.weight_numel,)
        if tpp.shared_weights
        else (num_batch, tpp.weight_numel),
        "w_grad": (tpp.weight_numel,)
        if tpp.shared_weights
        else (num_batch, tpp.weight_numel),
        "z": (num_batch, tpp.irreps_out.dim),
        "z_grad": (num_batch, tpp.irreps_out.dim),
        "z_dgrad": (num_batch, tpp.irreps_out.dim),
    }


def tp_dtypes(tpp: TPProblem) -> dict[BufferName, torch.dtype]:
    irrep_dtype = enum_to_torch_dtype_mapping[dtype_to_enum_mapping[tpp.irrep_dtype]]
    weight_dtype = enum_to_torch_dtype_mapping[dtype_to_enum_mapping[tpp.weight_dtype]]
    return {
        "x": irrep_dtype,
        "x_grad": irrep_dtype,
        "x_dgrad": irrep_dtype,
        "y": irrep_dtype,
        "y_grad": irrep_dtype,
        "y_dgrad": irrep_dtype,
        "w": weight_dtype,
        "w_grad": weight_dtype,
        "w_dgrad": weight_dtype,
        "z": irrep_dtype,
        "z_grad": irrep_dtype,
        "z_dgrad": irrep_dtype,
    }


def conv_shapes(
    edge_index: EdgeIndex, tpp: TPProblem
) -> dict[BufferName, tuple[int, ...]]:
    node_count = edge_index.sparse_size(0)
    nnz_count = len(edge_index[0])
    return {
        "x": (node_count, tpp.irreps_in1.dim),
        "x_grad": (node_count, tpp.irreps_in1.dim),
        "x_dgrad": (node_count, tpp.irreps_in1.dim),
        "y": (nnz_count, tpp.irreps_in2.dim),
        "y_grad": (nnz_count, tpp.irreps_in2.dim),
        "y_dgrad": (nnz_count, tpp.irreps_in2.dim),
        "w": (tpp.weight_numel,)
        if tpp.shared_weights
        else (nnz_count, tpp.weight_numel),
        "w_grad": (tpp.weight_numel,)
        if tpp.shared_weights
        else (nnz_count, tpp.weight_numel),
        "w_dgrad": (tpp.weight_numel,)
        if tpp.shared_weights
        else (nnz_count, tpp.weight_numel),
    }


@pytest.fixture
def tp_buffers(num_batch, tpp: TPProblem, cuda, gen):
    """
    Creates appropriate buffers for a tpp, on provided cuda device, using the provided generator
    """
    X = torch.rand(
        num_batch,
        tpp.irreps_in1.dim,
        device=cuda,
        generator=gen,
        dtype=enum_to_torch_dtype_mapping[dtype_to_enum_mapping[tpp.irrep_dtype]],
    )
    Y = torch.rand(
        num_batch,
        tpp.irreps_in2.dim,
        device=cuda,
        generator=gen,
        dtype=enum_to_torch_dtype_mapping[dtype_to_enum_mapping[tpp.irrep_dtype]],
    )
    # TODO: Need to support shared weights
    W = torch.rand(
        num_batch,
        tpp.weight_numel,
        device=cuda,
        generator=gen,
        dtype=enum_to_torch_dtype_mapping[dtype_to_enum_mapping[tpp.weight_dtype]],
    )
    return (X, Y, W)


@pytest.fixture
def conv_buffers(edge_index, tpp: TPProblem, cuda, gen):
    """
    Creates appropriate buffers for a conv problem, on provided cuda device, using the provided generator
    """
    X = torch.rand(edge_index.num_rows, tpp.irreps_in1.dim, device=cuda, generator=gen)
    Y = torch.rand(edge_index.num_cols, tpp.irreps_in2.dim, device=cuda, generator=gen)
    # TODO: Need to support shared weights
    W = torch.rand(edge_index.num_cols, tpp.weight_numel, device=cuda, generator=gen)
    # TODO: Need to support deterministic aka transpose permutation
    return None
    return (X, Y, W, edge_index[0], edge_index[1])
