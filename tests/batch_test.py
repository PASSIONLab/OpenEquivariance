import pytest
from pytest_check import check

import numpy as np 
import openequivariance as oeq
from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance.benchmark.correctness_utils import correctness_forward, correctness_backward, correctness_double_backward
from itertools import chain, product

class TPCorrectness:
    def check_result(self, result, fieldname):
        with check:
            error = result[fieldname]["diff_Linf_norm"]
            thresh = result["thresh"]
            assert result[fieldname]["pass"], f"{fieldname} observed error={error:.5f} >= {thresh}"

    @pytest.fixture(params=[np.float32, np.float64], ids=['F32', 'F64'], scope="class")
    def dtype(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def tp_and_problem(self, problem):
        tp = TensorProduct(problem)
        return tp, problem

    @pytest.mark.skip
    def test_tp_fwd(self, tp_and_problem): 
        tp, problem = tp_and_problem
        result = correctness_forward(
            problem=problem,
            test_implementation=tp,
            reference_implementation=None, 
            batch_size=1000,
            correctness_threshold=1e-5,
            prng_seed=12345)

        self.check_result(result, "output")

    @pytest.mark.skip
    def test_tp_bwd(self, tp_and_problem): 
        tp, problem = tp_and_problem
        result = correctness_backward(
            problem=problem,
            test_implementation=tp,
            reference_implementation=None, 
            batch_size=1000,
            correctness_threshold=3e-4,
            prng_seed=12345)

        self.check_result(result, "weight_grad")
        self.check_result(result, "in1_grad")
        self.check_result(result, "in2_grad")

    def test_tp_double_bwd(self, tp_and_problem):
        tp, problem = tp_and_problem
        result = correctness_double_backward(
            problem = problem,
            test_implementation=tp,
            reference_implementation = None,
            batch_size = 1000,
            correctness_threshold = 3e-4,
            prng_seed = 12345)

        self.check_result(result, "output_double_grad")
        self.check_result(result, "in1_grad")
        self.check_result(result, "in2_grad")
        self.check_result(result, "weights_grad")

class TestProductionModels(TPCorrectness):
    from openequivariance.benchmark.benchmark_configs \
            import e3nn_torch_tetris_polynomial, diffdock_configs, mace_nequip_problems
    production_model_tpps = list(chain(
            mace_nequip_problems,
            e3nn_torch_tetris_polynomial, 
            diffdock_configs))

    @pytest.fixture(params=production_model_tpps, ids = lambda x : x.label)
    def problem(self, request, dtype):
        request.param.irrep_dtype, request.param.weight_dtype = dtype, dtype
        return request.param

class TestUVUSingleIrrep(TPCorrectness):
    muls = [
        (1, 1, 1), (2, 1, 2), (4, 1, 4), (8, 1, 8), (16, 1, 16), 
        (32, 1, 32), (5, 1, 5), (13, 1, 13), (19, 1, 19),
        (33, 1, 33), (49, 1, 49), (50, 1, 50), (123, 1, 123),
        (128, 1, 128), (256, 1, 256), (512, 1, 512),
        (1, 2, 1), (1, 4, 1), (1, 16, 1), (1, 32, 1),
        (16, 3, 16), (16, 9, 16), (24, 24, 24), (32, 32, 32) 
    ]
    
    irs = [ (0, 0, 0), (1, 1, 1), (1, 0, 1), (1, 2, 1),
        (2, 0, 2), (2, 2, 4), (2, 2, 2), (5, 3, 5), (7, 2, 5) ]
    
    def id_func(m, i): 
        return f"{m[0]}x{i[0]}e__x__{m[1]}x{i[1]}e---{m[2]}x{i[2]}e"

    @pytest.fixture(params=product(muls, irs), 
                    ids = lambda x: TestUVUSingleIrrep.id_func(x[0], x[1]),
                    scope="class") 
    def problem(self, request, dtype):
        m, i = request.param[0], request.param[1]
        instructions=[(0, 0, 0, "uvu", True)]
        return oeq.TPProblem(f"{m[0]}x{i[0]}e", f"{m[1]}x{i[1]}e", f"{m[2]}x{i[2]}e",
                             instructions, shared_weights=False, 
                             internal_weights=False,
                             irrep_dtype=dtype, weight_dtype=dtype)
    

class TestUVWSingleIrrep(TPCorrectness):
    muls = [
        (1, 1, 1), (2, 1, 2), (4, 1, 4), (8, 1, 8), (16, 1, 16), 
        (32, 1, 32), (5, 1, 5), (13, 1, 13), (19, 1, 19),
        (33, 1, 33), (49, 1, 49), (50, 1, 50), (64, 1, 64), 
        (1, 2, 1), (1, 4, 1), (1, 16, 1), (1, 32, 1),
        (16, 3, 16), (16, 9, 16), (24, 24, 24), (32, 32, 32) 
    ]
    
    irs = [ (0, 0, 0), (1, 1, 1), (1, 0, 1), (1, 2, 1),
        (2, 0, 2), (2, 2, 4), (2, 2, 2), (5, 3, 5), (7, 2, 5) ]

    def id_func(m, i): 
        return f"{m[0]}x{i[0]}e__x__{m[1]}x{i[1]}e---{m[2]}x{i[2]}e"

    @pytest.fixture(params=product(muls, irs), 
                    ids = lambda x: TestUVWSingleIrrep.id_func(x[0], x[1]),
                    scope="class")
    def problem(self, request, dtype):
        m, i = request.param[0], request.param[1]
        instructions=[(0, 0, 0, "uvw", True)]
        return oeq.TPProblem(f"{m[0]}x{i[0]}e", f"{m[1]}x{i[1]}e", f"{m[2]}x{i[2]}e",
                             instructions, shared_weights=False, 
                             internal_weights=False,
                             irrep_dtype=dtype, weight_dtype=dtype)