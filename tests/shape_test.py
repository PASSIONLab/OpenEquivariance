import pytest

import torch
import e3nn.o3 as o3
import openequivariance as oeq


def test_wrong_L1_dims():
    gen = torch.Generator(device="cuda")

    batch_size = 1000
    X_ir, Y_ir, Z_ir = o3.Irreps("1x2e"), o3.Irreps("1x3e"), o3.Irreps("1x2e")
    X = torch.rand(batch_size, X_ir.dim, device="cuda", generator=gen)
    Y = torch.rand(batch_size, Y_ir.dim, device="cuda", generator=gen)

    instructions = [(0, 0, 0, "uvu", True)]

    problem = oeq.TPProblem(
        X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False
    )
    tp = oeq.TensorProduct(problem, torch_op=True)

    W = torch.rand(batch_size, problem.weight_numel, device="cuda", generator=gen)

    # Make it wrong
    X = X.unsqueeze(0)
    with pytest.raises(RuntimeError):
        tp(X, Y, W)
