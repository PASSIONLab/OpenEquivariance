import pytest
import torch

from openequivariance import TPProblem, TensorProductConv

from e3nn import o3
from torch_geometric import EdgeIndex


@pytest.fixture
def gen():
    return torch.Generator(device="cuda")


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
def tp_conv(tpp):
    return TensorProductConv(tpp, deterministic=False)


def test_no_response(tp_conv, conv_buffers):
    torch.use_deterministic_algorithms(False)
    tp_conv(*conv_buffers)


def test_warning(tp_conv, conv_buffers, capfd):
    torch.use_deterministic_algorithms(True, warn_only=True)
    tp_conv(*conv_buffers)

    captured = capfd.readouterr()
    assert "Warning" in captured.err
    assert "does not have a deterministic implementation" in captured.err


def test_error(tp_conv, conv_buffers):
    torch.use_deterministic_algorithms(True, warn_only=False)
    with pytest.raises(RuntimeError):
        tp_conv(*conv_buffers)
