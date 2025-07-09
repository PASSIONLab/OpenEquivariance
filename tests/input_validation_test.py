import torch
from e3nn import o3
from torch_geometric import EdgeIndex
import pytest

from openequivariance import TPProblem, TensorProduct, TensorProductConv

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
def edge_index():
    ei = EdgeIndex(
        data=[
            [0, 1, 1, 2],  # Receiver
            [1, 0, 2, 1],  # Sender
        ],
        sort_order="row",
        sparse_size=(3, 4),
        device="cuda",
        dtype=torch.long,
    )
    ei.fill_cache_()
    return ei

@pytest.fixture
def tp_buffers(tpp):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(42)
    N = 1000

    X = torch.rand(N, tpp.irreps_in1.dim, device="cuda", generator=gen)
    Y = torch.rand(N, tpp.irreps_in2.dim, device="cuda", generator=gen)
    W = torch.rand(N, tpp.weight_numel, device="cuda", generator=gen)
    return [X, Y, W]


@pytest.fixture
def conv_buffers(edge_index, tpp):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(42)

    X = torch.rand(
        edge_index.num_rows, tpp.irreps_in1.dim, device="cuda", generator=gen
    )
    Y = torch.rand(
        edge_index.num_cols, tpp.irreps_in2.dim, device="cuda", generator=gen
    )
    W = torch.rand(edge_index.num_cols, tpp.weight_numel, device="cuda", generator=gen)
    _, inv_perm = edge_index.get_csc()
    return [X, Y, W, edge_index[0], edge_index[1], inv_perm]


def new_copy(bufs: list[torch.Tensor]) -> list[torch.Tensor]:
    return [buf.clone().detach() for buf in bufs]

@pytest.fixture(
    params=[
        "TensorProduct",
        "TensorProductConvAtomic",
        "TensorProductConvDeterministic",
        "TensorProductConvKahan",
    ]
)
def executable_and_buffers(request, conv_buffers, tp_buffers, tpp):
    match request.param:
        case "TensorProduct":
            return (TensorProduct(tpp), tp_buffers)
        case "TensorProductConvAtomic":
            return (TensorProductConv(tpp, deterministic=False), conv_buffers[:-1])
        case "TensorProductConvDeterministic":
            return (TensorProductConv(tpp, deterministic=True), conv_buffers)
        case "TensorProductConvKahan":
            return (
                TensorProductConv(tpp, deterministic=True, kahan=True),
                conv_buffers,
            )


def test_cpp_checks_forward_positive(executable_and_buffers):
    executable, buffers = executable_and_buffers
    executable(*buffers)


def test_cpp_checks_forward_dimensions(executable_and_buffers, subtests):
    executable, fixture_buffers = executable_and_buffers
    for i in range(len(fixture_buffers)):
        buffers = new_copy(fixture_buffers)
        with subtests.test(msg="Dimension Checks", buffer_index=i):
            with pytest.raises(RuntimeError, match=r"Shape mismatch"):
                buffers[i] = buffers[i].unsqueeze(1)
                executable(*buffers)


def test_cpp_checks_forward_sizes(executable_and_buffers, subtests):
    executable, fixture_buffers = executable_and_buffers
    for i in range(len(fixture_buffers)):
        for j in range(fixture_buffers[i].dim()):
            buffers = new_copy(fixture_buffers)
            with subtests.test(
                msg="Size Checks in each dim", buffer_index=i, dimension=j
            ):
                if i == 0 and j == 0 and isinstance(executable, TensorProductConv):
                    pytest.skip(reason="There is no way to check if this is wrong")
                with pytest.raises(RuntimeError, match=r"Shape mismatch"):
                    shape = list(buffers[i].shape)
                    shape[j] += 1
                    buffers[i] = buffers[i].resize_(shape)
                    executable(*buffers)


def test_cpp_checks_forward_device(executable_and_buffers, subtests):
    executable, fixture_buffers = executable_and_buffers
    for i in range(len(fixture_buffers)):
        buffers = new_copy(fixture_buffers)
        with subtests.test(msg="Device Checks", buffer_index=i):
            with pytest.raises(RuntimeError, match=r"is not on the GPU"):
                buffers[i] = buffers[i].to(device="cpu")
                executable(*buffers)


def test_cpp_checks_forward_dtype(executable_and_buffers, subtests):
    executable, fixture_buffers = executable_and_buffers
    for i in range(len(fixture_buffers)):
        buffers = new_copy(fixture_buffers)
        with subtests.test(msg="Dtype Checks", buffer_index=i):
            with pytest.raises(RuntimeError, match=r"Dtype mismatch"):
                buffers[i] = buffers[i].to(dtype=torch.bfloat16)
                executable(*buffers)
