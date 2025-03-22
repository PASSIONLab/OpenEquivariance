import pytest
from pytest_check import check

import openequivariance as oeq 
from openequivariance.implementations.e3nn_lite import TPProblem
from openequivariance.implementations.TensorProductBase import TensorProductBase
from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance.benchmark.correctness_utils import correctness_forward, correctness_backward, correctness_double_backward
from itertools import chain

@pytest.fixture(params=[TensorProduct])
def test_impl(request):
    return request.param

class TPCorrectness:
    def test_tp_forward(self, problem, test_impl): 
        result = correctness_forward(
            problem=problem,
            test_implementation=test_impl,
            reference_implementation=None, 
            batch_size=1000,
            correctness_threshold=1e-6,
            prng_seed=12345)
        
        assert result["output"]["pass"]

    def test_tp_backward(self, problem, test_impl): 
        result = correctness_backward(
            problem=problem,
            test_implementation=test_impl,
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
    def test_tp_double_backward(self, problem, test_impl):
        result = correctness_double_backward(
            problem = problem,
            test_implementation = test_impl,
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
    @pytest.fixture(params=[ 
        (1, 1, 1), (2, 1, 2), (4, 1, 4), (8, 1, 8), (16, 1, 16), 
        (32, 1, 32), (5, 1, 5), (13, 1, 13), (19, 1, 19),
        (33, 1, 33), (49, 1, 49), (50, 1, 50), (123, 1, 123),
        (128, 1, 128), (256, 1, 256), (512, 1, 512),
        (1, 2, 1), (1, 4, 1), (1, 8, 1), (1, 16, 1), (1, 32, 1),
        (16, 3, 16), (16, 4, 16), (16, 8, 16), (24, 24, 24), (32, 32, 32) 
    ]) 
    def mul(self, request): 
        return request.param
    
    @pytest.fixture(params=[ 
        (0, 0, 0), (1, 1, 1), (1, 0, 1), (1, 2, 1),
        (2, 0, 2), (2, 2, 4), (2, 2, 2), (5, 3, 5), (7, 2, 5) 
    ]) 
    def ir(self, request): 
        return request.param

    @pytest.fixture
    def problem(self, mul, ir):
        instructions=[(0, 0, 0, "uvu", True)]
        config = (f"{mul[0]}x{ir[0]}e", f"{mul[1]}x{ir[1]}e", f"{mul[2]}x{ir[2]}e")
        return oeq.TPProblem(config[0], config[1], config[2], instructions, 
                             shared_weights=False, internal_weights=False)
