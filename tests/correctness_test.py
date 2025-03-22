import pytest
from pytest_check import check

import openequivariance as oeq
import numpy as np 
from openequivariance.implementations.e3nn_lite import TPProblem
from openequivariance.implementations.TensorProductBase import TensorProductBase
from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance.benchmark.correctness_utils import correctness_forward, correctness_backward, correctness_double_backward
from itertools import chain, product

@pytest.fixture(params=[TensorProduct], ids=['oeq.TensorProduct'])
def implementation(request):
    return request.param

@pytest.fixture(params=[np.float32, np.float64], ids=['F32', 'F64'])
def dtype(request):
    return request.param

class TPCorrectness:
    def test_tp_fwd(self, problem, implementation): 
        result = correctness_forward(
            problem=problem,
            test_implementation=implementation,
            reference_implementation=None, 
            batch_size=1000,
            correctness_threshold=1e-6,
            prng_seed=12345)
        
        assert result["output"]["pass"]

    def test_tp_bwd(self, problem, implementation): 
        result = correctness_backward(
            problem=problem,
            test_implementation=implementation,
            reference_implementation=None, 
            batch_size=1000,
            correctness_threshold=1e-4,
            prng_seed=12345)
        
        with check: 
            assert result["weight_grad"]["pass"]
        with check:
            assert result["in1_grad"]["pass"]
        with check:
            assert result["in2_grad"]["pass"]

    @pytest.mark.skip(reason="Need to add weight reordering in double-backward")
    def test_tp_double_bwd(self, problem, implementation):
        result = correctness_double_backward(
            problem = problem,
            test_implementation = implementation,
            reference_implementation = None,
            batch_size = 1000,
            correctness_threshold = 1e-4,
            prng_seed = 12345)
        
        with check:
            assert result["output_grad"]["pass"]
        with check: 
            assert result["in1_grad"]["pass"] 
        with check:      
            assert result["in2_grad"]["pass"] 
        with check: 
            assert result["weights_grad"]["pass"]

class TestProductionModels(TPCorrectness):
    from openequivariance.benchmark.benchmark_configs import e3nn_torch_tetris_polynomial, diffdock_configs
    production_model_tpps = list(chain(
            e3nn_torch_tetris_polynomial, 
            diffdock_configs))

    @pytest.fixture(params=production_model_tpps, ids = lambda x : x.label)
    def problem(self, request): 
        return request.param

class TestUVUSingleIrrep(TPCorrectness):
    muls = [
        (1, 1, 1), (2, 1, 2), (4, 1, 4), (8, 1, 8), (16, 1, 16), 
        (32, 1, 32), (5, 1, 5), (13, 1, 13), (19, 1, 19),
        (33, 1, 33), (49, 1, 49), (50, 1, 50), (123, 1, 123),
        (128, 1, 128), (256, 1, 256), (512, 1, 512),
        (1, 2, 1), (1, 4, 1), (1, 8, 1), (1, 16, 1), (1, 32, 1),
        (16, 3, 16), (16, 4, 16), (16, 8, 16), (24, 24, 24), (32, 32, 32) 
    ]
    
    irs = [ 
        (0, 0, 0), (1, 1, 1), (1, 0, 1), (1, 2, 1),
        (2, 0, 2), (2, 2, 4), (2, 2, 2), (5, 3, 5), (7, 2, 5) 
    ]
    
    def id_func(m, i): 
        return f"({m[0]}x{i[0]}e) x ({m[1]}x{i[1]}e) -> ({m[2]}x{i[2]})"

    @pytest.fixture(params=product(muls, irs), ids = lambda x: TestUVUSingleIrrep.id_func(x[0], x[1])) 
    def problem(self, request, dtype):
        mul, ir = request.param[0], request.param[1]
        instructions=[(0, 0, 0, "uvu", True)]
        c =f"{mul[0]}x{ir[0]}e", f"{mul[0]}x{ir[0]}e", f"{mul[1]}x{ir[1]}e",
        return oeq.TPProblem(c[0], c[1], c[2], 
                             instructions, shared_weights=False, 
                             internal_weights=False,
                             irrep_dtype=dtype, weight_dtype=dtype)