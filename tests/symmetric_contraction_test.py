import collections
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from e3nn import o3

from openequivariance._torch.symmetric_contraction import SymmetricContraction

from conftest import device_type

DEVICE = device_type()

mace_symmetric_contraction = pytest.importorskip("mace.modules.symmetric_contraction")
MaceSymmetricContraction = mace_symmetric_contraction.SymmetricContraction


SCConfig = collections.namedtuple(
    "SCConfig",
    [
        "irreps_in",
        "irreps_out",
        "correlation",
        "num_elements",
        "label_values",
        "device",
    ],
)

DEVICE = torch.device(DEVICE)

SC_CONFIGS = [
    SCConfig(
        o3.Irreps("2x0e + 2x1o"),
        o3.Irreps("2x0e + 2x1o"),
        2,
        4,
        [0, 2, 3, 2, 0, 0, 2, 3, 2, 2],
        DEVICE,
    ),
    SCConfig(
        o3.Irreps("1x0e + 1x1o + 1x2e"),
        o3.Irreps("1x0e + 1x1o"),
        3,
        3,
        [0, 1, 2, 0, 1, 2, 0, 1],
        DEVICE,
    ),
    SCConfig(
        o3.Irreps("4x0e + 4x1o"),
        o3.Irreps("4x0e"),
        2,
        5,
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        DEVICE,
    ),
]


@pytest.fixture(
    params=SC_CONFIGS,
    ids=lambda cfg: f"{cfg.irreps_in}-corr{cfg.correlation}",
)
def symmetric_contraction_config(request):
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64], ids=["F32", "F64"])
def dtype(request):
    return request.param


@pytest.fixture
def labels(symmetric_contraction_config):
    cfg = symmetric_contraction_config
    return torch.tensor(cfg.label_values, device=cfg.device, dtype=torch.long)


@pytest.fixture
def node_attrs(labels, dtype, symmetric_contraction_config):
    return F.one_hot(labels, num_classes=symmetric_contraction_config.num_elements).to(
        dtype=dtype
    )


@pytest.fixture
def node_feats(dtype, symmetric_contraction_config):
    cfg = symmetric_contraction_config
    gen = torch.Generator(device=cfg.device)
    gen.manual_seed(2468)
    return torch.randn(
        len(cfg.label_values),
        cfg.irreps_in.count((0, 1)),
        cfg.irreps_in.dim // cfg.irreps_in.count((0, 1)),
        device=cfg.device,
        dtype=dtype,
        generator=gen,
        requires_grad=True,
    )


@pytest.fixture
def modules(dtype, symmetric_contraction_config):
    cfg = symmetric_contraction_config
    torch.manual_seed(12345)
    oeq_module = SymmetricContraction(
        cfg.irreps_in,
        cfg.irreps_out,
        correlation=cfg.correlation,
        num_elements=cfg.num_elements,
        dtype=dtype,
    ).to(cfg.device)

    with patch(
        "mace.modules.symmetric_contraction.torch.get_default_dtype",
        return_value=dtype,
    ):
        mace_module = MaceSymmetricContraction(
            cfg.irreps_in,
            cfg.irreps_out,
            correlation=cfg.correlation,
            num_elements=cfg.num_elements,
        ).to(device=cfg.device, dtype=dtype)

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
        self, modules, node_feats, node_attrs, dtype, symmetric_contraction_config
    ):
        cfg = symmetric_contraction_config
        oeq_module, mace_module = modules
        mace_node_feats = node_feats.detach().clone().requires_grad_()

        oeq_output = oeq_module(node_feats, node_attrs)
        mace_output = mace_module(mace_node_feats, node_attrs)

        assert oeq_output.shape == (len(cfg.label_values), cfg.irreps_out.dim)
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

    def test_matches_mace_double_backward(
        self, modules, node_feats, node_attrs, dtype, symmetric_contraction_config
    ):
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

    def test_compile(self, modules, node_feats, node_attrs):
        sc, _ = modules
        ref = sc(node_feats, node_attrs)
        assert torch.allclose(ref, torch.compile(sc)(node_feats, node_attrs), atol=1e-5)

    def test_export(self, modules, node_feats, node_attrs):
        sc, _ = modules
        ref = sc(node_feats, node_attrs)
        exported = torch.export.export(sc, args=(node_feats, node_attrs), strict=False)
        assert torch.allclose(ref, exported.module()(node_feats, node_attrs), atol=1e-5)
