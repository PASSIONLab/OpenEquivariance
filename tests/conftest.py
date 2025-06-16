from dataclasses import dataclass
from typing import Callable, Tuple, Any

import pytest
import torch
from e3nn import o3
from torch_geometric import EdgeIndex

from openequivariance import TPProblem
from openequivariance.implementations.dtype_enum import (
    enum_to_torch_dtype_mapping,
    dtype_to_enum_mapping,
)


@dataclass
class Executable:
    func: Callable[..., Any]
    buffers: Tuple[torch.Tensor, ...]

    def __call__(self) -> Any:
        return self.func(*self.buffers)


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
def num_batch():
    return 1000


@pytest.fixture
def edge_index():
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
def tpp():
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
    return (X, Y, W, edge_index[0], edge_index[1])
