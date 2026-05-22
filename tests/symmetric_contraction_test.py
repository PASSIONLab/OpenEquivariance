from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from e3nn import o3

import openequivariance as oeq
from openequivariance._torch.symmetric_contraction import SymmetricContraction

mace_symmetric_contraction = pytest.importorskip("mace.modules.symmetric_contraction")
MaceSymmetricContraction = mace_symmetric_contraction.SymmetricContraction


IRREPS_IN = o3.Irreps("2x0e + 2x1o")
IRREPS_OUT = o3.Irreps("2x0e + 2x1o")
CORRELATION = 2
NUM_ELEMENTS = 4
LABEL_VALUES = [0, 2, 3, 2, 0, 0, 2, 3, 2, 2]


@pytest.fixture(params=[torch.float32, torch.float64], ids=["F32", "F64"])
def dtype(request):
    return request.param


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip(
            "SymmetricContraction requires a CUDA/HIP device exposed through torch.cuda"
        )
    if not oeq.BUILT_EXTENSION:
        pytest.skip(
            f"OpenEquivariance extension is not built: {oeq.BUILT_EXTENSION_ERROR}"
        )
    return torch.device("cuda")


@pytest.fixture
def labels(device):
    return torch.tensor(LABEL_VALUES, device=device, dtype=torch.long)


@pytest.fixture
def node_attrs(labels, dtype):
    return F.one_hot(labels, num_classes=NUM_ELEMENTS).to(dtype=dtype)


@pytest.fixture
def node_feats(device, dtype):
    gen = torch.Generator(device=device)
    gen.manual_seed(2468)
    return torch.randn(
        len(LABEL_VALUES),
        IRREPS_IN.count((0, 1)),
        IRREPS_IN.dim // IRREPS_IN.count((0, 1)),
        device=device,
        dtype=dtype,
        generator=gen,
        requires_grad=True,
    )


@pytest.fixture
def modules(device, dtype):
    torch.manual_seed(12345)
    oeq_module = SymmetricContraction(
        IRREPS_IN,
        IRREPS_OUT,
        correlation=CORRELATION,
        num_elements=NUM_ELEMENTS,
        dtype=dtype,
    ).to(device)

    # MACE's original e3nn implementation reads torch.get_default_dtype()
    # during construction, so patch that lookup instead of mutating global state.
    with patch(
        "mace.modules.symmetric_contraction.torch.get_default_dtype",
        return_value=dtype,
    ):
        mace_module = MaceSymmetricContraction(
            IRREPS_IN,
            IRREPS_OUT,
            correlation=CORRELATION,
            num_elements=NUM_ELEMENTS,
        ).to(device=device, dtype=dtype)

    copy_matching_state(oeq_module, mace_module)
    return oeq_module, mace_module


def tolerance(dtype):
    if dtype == torch.float64:
        return {"rtol": 1e-10, "atol": 1e-10}
    return {"rtol": 1e-4, "atol": 1e-4}


def copy_matching_state(source, target):
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name] = value.detach().clone().to(target_state[name])
    target.load_state_dict(target_state)


def matching_trainable_parameters(source, target):
    source_params = dict(source.named_parameters())
    target_params = dict(target.named_parameters())
    names = [
        name
        for name, param in source_params.items()
        if param.requires_grad
        and name in target_params
        and target_params[name].requires_grad
        and target_params[name].shape == param.shape
    ]
    assert names, "No matching trainable parameters found"
    return tuple(source_params[name] for name in names), tuple(
        target_params[name] for name in names
    )


def random_like(tensor, seed):
    gen = torch.Generator(device=tensor.device)
    gen.manual_seed(seed)
    return torch.randn(
        tensor.shape, device=tensor.device, dtype=tensor.dtype, generator=gen
    )


class TestSymmetricContraction:
    def test_matches_mace_forward_backward(
        self, modules, node_feats, node_attrs, dtype
    ):
        oeq_module, mace_module = modules
        mace_node_feats = node_feats.detach().clone().requires_grad_()

        oeq_output = oeq_module(node_feats, node_attrs)
        mace_output = mace_module(mace_node_feats, node_attrs)

        assert oeq_output.shape == (len(LABEL_VALUES), IRREPS_OUT.dim)
        torch.testing.assert_close(oeq_output, mace_output, **tolerance(dtype))

        output_grad = random_like(oeq_output, seed=4321)
        oeq_params, mace_params = matching_trainable_parameters(oeq_module, mace_module)

        oeq_grads = torch.autograd.grad(
            oeq_output, (node_feats, *oeq_params), grad_outputs=output_grad
        )
        mace_grads = torch.autograd.grad(
            mace_output, (mace_node_feats, *mace_params), grad_outputs=output_grad
        )

        for oeq_grad, mace_grad in zip(oeq_grads, mace_grads):
            torch.testing.assert_close(oeq_grad, mace_grad, **tolerance(dtype))

    def test_matches_mace_double_backward(self, modules, node_feats, node_attrs, dtype):
        oeq_module, mace_module = modules
        mace_node_feats = node_feats.detach().clone().requires_grad_()

        oeq_output = oeq_module(node_feats, node_attrs)
        mace_output = mace_module(mace_node_feats, node_attrs)
        oeq_output_grad = random_like(oeq_output, seed=9876).requires_grad_()
        mace_output_grad = oeq_output_grad.detach().clone().requires_grad_()

        oeq_params, mace_params = matching_trainable_parameters(oeq_module, mace_module)
        oeq_tensors = (node_feats, *oeq_params)
        mace_tensors = (mace_node_feats, *mace_params)

        oeq_first_grads = torch.autograd.grad(
            oeq_output,
            oeq_tensors,
            grad_outputs=oeq_output_grad,
            create_graph=True,
        )
        mace_first_grads = torch.autograd.grad(
            mace_output,
            mace_tensors,
            grad_outputs=mace_output_grad,
            create_graph=True,
        )

        for oeq_grad, mace_grad in zip(oeq_first_grads, mace_first_grads):
            torch.testing.assert_close(oeq_grad, mace_grad, **tolerance(dtype))

        probes = [
            random_like(grad, seed=1357 + index)
            for index, grad in enumerate(oeq_first_grads)
        ]
        oeq_target = sum(
            (grad * probe).sum() for grad, probe in zip(oeq_first_grads, probes)
        )
        mace_target = sum(
            (grad * probe).sum() for grad, probe in zip(mace_first_grads, probes)
        )

        oeq_second_grads = torch.autograd.grad(
            oeq_target, oeq_tensors + (oeq_output_grad,)
        )
        mace_second_grads = torch.autograd.grad(
            mace_target, mace_tensors + (mace_output_grad,)
        )

        for oeq_grad, mace_grad in zip(oeq_second_grads, mace_second_grads):
            torch.testing.assert_close(oeq_grad, mace_grad, **tolerance(dtype))
