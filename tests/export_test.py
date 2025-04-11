import torch
import pytest, tempfile
from pytest_check import check

import numpy as np
import openequivariance as oeq
from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance.benchmark.correctness_utils import correctness_forward, correctness_backward, correctness_double_backward

def test_jitscript_batch():
    X_ir, Y_ir, Z_ir = oeq.Irreps("32x5e"), oeq.Irreps("1x3e"), oeq.Irreps("32x5e")
    problem = oeq.TPProblem(X_ir, Y_ir, Z_ir,
                            [(0, 0, 0, "uvu", True)], 
                            shared_weights=False, internal_weights=False,
                            irrep_dtype=np.float32, weight_dtype=np.float32)

    tp = oeq.TensorProduct(problem)

    batch_size = 1000
    gen = torch.Generator(device='cuda')
    gen.manual_seed(0)
    X = torch.rand(batch_size, X_ir.dim, device='cuda', generator=gen)
    Y = torch.rand(batch_size, Y_ir.dim, device='cuda', generator=gen)
    W = torch.rand(batch_size, problem.weight_numel, device='cuda', generator=gen)

    uncompiled_result = tp.forward(X, Y, W)
    print(tp.forward)

    scripted_tp = torch.jit.script(tp)
    loaded_tp = None

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
        scripted_tp.save(tmp_file.name) 
        loaded_tp = torch.jit.load(tmp_file.name)
    
    compiled_result = loaded_tp.forward(X, Y, W)
    assert torch.allclose(uncompiled_result, compiled_result, atol=1e-5)