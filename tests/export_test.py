import torch
import pytest, tempfile

import numpy as np
import openequivariance as oeq
from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance.benchmark.correctness_utils import correctness_forward, correctness_backward, correctness_double_backward

@pytest.fixture
def problem_and_irreps():
    X_ir, Y_ir, Z_ir = oeq.Irreps("32x5e"), oeq.Irreps("1x3e"), oeq.Irreps("32x5e")
    problem = oeq.TPProblem(X_ir, Y_ir, Z_ir,
                            [(0, 0, 0, "uvu", True)], 
                            shared_weights=False, internal_weights=False,
                            irrep_dtype=np.float32, weight_dtype=np.float32)

    batch_size = 1000
    gen = torch.Generator(device='cuda')
    gen.manual_seed(0)
    X = torch.rand(batch_size, X_ir.dim, device='cuda', generator=gen)
    Y = torch.rand(batch_size, Y_ir.dim, device='cuda', generator=gen)
    W = torch.rand(batch_size, problem.weight_numel, device='cuda', generator=gen)

    return problem, X_ir, Y_ir, Z_ir, 


@pytest.fixture
def batch_inputs(problem_and_irreps):
    problem, X_ir, Y_ir, _ = problem_and_irreps
    batch_size = 1000
    gen = torch.Generator(device='cuda')
    gen.manual_seed(0)
    X = torch.rand(batch_size, X_ir.dim, device='cuda', generator=gen)
    Y = torch.rand(batch_size, Y_ir.dim, device='cuda', generator=gen)
    W = torch.rand(batch_size, problem.weight_numel, device='cuda', generator=gen)
    return X, Y, W

def test_jitscript_batch(problem_and_irreps, batch_inputs):
    problem, _, _, _ = problem_and_irreps
    tp = oeq.TensorProduct(problem)
    uncompiled_result = tp.forward(*batch_inputs)

    scripted_tp = torch.jit.script(tp)
    loaded_tp = None
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
        scripted_tp.save(tmp_file.name) 
        loaded_tp = torch.jit.load(tmp_file.name)
    
    compiled_result = loaded_tp(*batch_inputs)
    assert torch.allclose(uncompiled_result, compiled_result, atol=1e-5)


def test_export_batch(problem_and_irreps, batch_inputs):
    problem, _, _, _ = problem_and_irreps
    tp = oeq.TensorProduct(problem)
    uncompiled_result = tp.forward(*batch_inputs)

    exported_tp = torch.export.export(tp, args=batch_inputs, strict=False)
    exported_result = exported_tp.module()(*batch_inputs)
    assert torch.allclose(uncompiled_result, exported_result, atol=1e-5)


def test_aoti_batch(problem_and_irreps, batch_inputs):
    problem, _, _, _ = problem_and_irreps
    tp = oeq.TensorProduct(problem)

    uncompiled_result = tp.forward(*batch_inputs)

    exported_tp = torch.export.export(tp, args=batch_inputs, strict=False)
    aoti_model = None
    with tempfile.NamedTemporaryFile(suffix=".pt2") as tmp_file:
        try:
            output_path = torch._inductor.aoti_compile_and_package( 
                exported_tp,
                package_path=tmp_file.name)
        except Exception as e:
            err_msg = \
            "AOTI compile_and_package failed. NOTE: OpenEquivariance only supports AOTI for " + \
            "PyTorch version >= 2.8.0.dev20250410+cu126 due to incomplete TorchBind support " + \
            "in prior versions. " + \
            f"{e}"
            assert False, err_msg 
                           
        aoti_model = torch._inductor.aoti_load_package(output_path)

    aoti_result = aoti_model(*batch_inputs)
    assert torch.allclose(uncompiled_result, aoti_result, atol=1e-5)