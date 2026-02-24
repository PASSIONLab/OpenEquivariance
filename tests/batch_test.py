import pytest
from pytest_check import check

import numpy as np
import openequivariance as oeq
from openequivariance.benchmark.correctness_utils import (
    correctness_forward,
    correctness_backward,
    correctness_double_backward,
)

from openequivariance.benchmark.problems import (
    e3nn_torch_tetris_poly_problems,
    diffdock_problems,
    mace_problems,
    nequip_problems,
)
from itertools import product
import torch


@pytest.fixture(params=[np.float32, np.float64], ids=["F32", "F64"], scope="module")
def dtype(request):
    return request.param


class TPCorrectness:
    def thresh(self, direction):
        return {"fwd": 1e-5, "bwd": 3e-4, "double_bwd": 3e-4}[direction]

    def check_result(self, result, fieldname):
        with check:
            error = result[fieldname]["diff_Linf_norm"]
            thresh = result["thresh"]
            assert result[fieldname]["pass"], (
                f"{fieldname} observed error={error:.5f} >= {thresh}"
            )

    @pytest.fixture(scope="class")
    def extra_tp_constructor_args(self):
        return {}

    @pytest.fixture(scope="class")
    def tp_and_problem(self, problem, extra_tp_constructor_args, with_jax):
        cls = oeq.TensorProduct
        if with_jax:
            import openequivariance.jax.TensorProduct as jax_tp

            cls = jax_tp
        tp = cls(problem, **extra_tp_constructor_args)
        return tp, problem

    def test_tp_fwd(self, tp_and_problem):
        tp, problem = tp_and_problem
        result = correctness_forward(
            problem=problem,
            test_implementation=tp,
            reference_implementation=None,
            batch_size=1000,
            correctness_threshold=self.thresh("fwd"),
            prng_seed=12345,
        )

        self.check_result(result, "output")

    def test_tp_bwd(self, tp_and_problem):
        tp, problem = tp_and_problem
        result = correctness_backward(
            problem=problem,
            test_implementation=tp,
            reference_implementation=None,
            batch_size=1000,
            correctness_threshold=self.thresh("bwd"),
            prng_seed=12345,
        )

        self.check_result(result, "weight_grad")
        self.check_result(result, "in1_grad")
        self.check_result(result, "in2_grad")

    def test_tp_double_bwd(self, tp_and_problem):
        tp, problem = tp_and_problem
        result = correctness_double_backward(
            problem=problem,
            test_implementation=tp,
            reference_implementation=None,
            batch_size=200,
            correctness_threshold=self.thresh("double_bwd"),
            prng_seed=12345,
        )

        self.check_result(result, "output_double_grad")
        self.check_result(result, "in1_grad")
        self.check_result(result, "in2_grad")
        self.check_result(result, "weights_grad")


class TestProductionModels(TPCorrectness):
    production_model_tpps = (
        mace_problems()
        + nequip_problems()
        + e3nn_torch_tetris_poly_problems()
        + diffdock_problems()
    )

    @pytest.fixture(params=production_model_tpps, ids=lambda x: x.label, scope="class")
    def problem(self, request, dtype):
        request.param.irrep_dtype, request.param.weight_dtype = dtype, dtype
        return request.param


class TestUVUSingleIrrep(TPCorrectness):
    muls = [
        (1, 1, 1),
        (2, 1, 2),
        (4, 1, 4),
        (8, 1, 8),
        (16, 1, 16),
        (32, 1, 32),
        (5, 1, 5),
        (13, 1, 13),
        (19, 1, 19),
        (33, 1, 33),
        (49, 1, 49),
        (50, 1, 50),
        (123, 1, 123),
        (128, 1, 128),
        (256, 1, 256),
        (512, 1, 512),
        (1, 2, 1),
        (1, 4, 1),
        (1, 16, 1),
        (1, 32, 1),
        (16, 3, 16),
        (16, 9, 16),
        (24, 24, 24),
        (32, 32, 32),
    ]

    irs = [
        (0, 0, 0),
        (1, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (2, 0, 2),
        (2, 2, 4),
        (2, 2, 2),
        (5, 3, 5),
        (7, 2, 5),
    ]

    def id_func(m, i):
        return f"{m[0]}x{i[0]}e__x__{m[1]}x{i[1]}e---{m[2]}x{i[2]}e"

    @pytest.fixture(
        params=product(muls, irs),
        ids=lambda x: TestUVUSingleIrrep.id_func(x[0], x[1]),
        scope="class",
    )
    def problem(self, request, dtype):
        m, i = request.param[0], request.param[1]
        instructions = [(0, 0, 0, "uvu", True)]
        return oeq.TPProblem(
            f"{m[0]}x{i[0]}e",
            f"{m[1]}x{i[1]}e",
            f"{m[2]}x{i[2]}e",
            instructions,
            shared_weights=False,
            internal_weights=False,
            irrep_dtype=dtype,
            weight_dtype=dtype,
        )


class TestUVWSingleIrrep(TPCorrectness):
    muls = [
        (1, 1, 1),
        (2, 1, 2),
        (4, 1, 4),
        (8, 1, 8),
        (16, 1, 16),
        (32, 1, 32),
        (5, 1, 5),
        (13, 1, 13),
        (19, 1, 19),
        (33, 1, 33),
        (49, 1, 49),
        (50, 1, 50),
        (64, 1, 64),
        (1, 2, 1),
        (1, 4, 1),
        (1, 16, 1),
        (1, 32, 1),
        (16, 3, 16),
        (16, 9, 16),
        (24, 24, 24),
        (32, 32, 32),
    ]

    irs = [
        (0, 0, 0),
        (1, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (2, 0, 2),
        (2, 2, 4),
        (2, 2, 2),
        (5, 3, 5),
        (7, 2, 5),
    ]

    def id_func(m, i):
        return f"{m[0]}x{i[0]}e__x__{m[1]}x{i[1]}e---{m[2]}x{i[2]}e"

    @pytest.fixture(
        params=product(muls, irs),
        ids=lambda x: TestUVWSingleIrrep.id_func(x[0], x[1]),
        scope="class",
    )
    def problem(self, request, dtype):
        m, i = request.param[0], request.param[1]
        instructions = [(0, 0, 0, "uvw", True)]
        return oeq.TPProblem(
            f"{m[0]}x{i[0]}e",
            f"{m[1]}x{i[1]}e",
            f"{m[2]}x{i[2]}e",
            instructions,
            shared_weights=False,
            internal_weights=False,
            irrep_dtype=dtype,
            weight_dtype=dtype,
        )


class TestSharedWeights(TPCorrectness):
    problems = [mace_problems()[0], diffdock_problems()[0]]

    def thresh(self, direction):
        return {
            "fwd": 1e-5,
            "bwd": 5e-4,  # Expect higher errors for shared weights
            "double_bwd": 5e-4,
        }[direction]

    @pytest.fixture(params=problems, ids=lambda x: x.label, scope="class")
    def problem(self, request, dtype):
        problem = request.param
        problem.irrep_dtype, problem.weight_dtype = dtype, dtype
        problem.shared_weights = True
        return problem


class TestTorchTo(TPCorrectness):
    problems = [mace_problems()[0]]

    @pytest.fixture(params=problems, ids=lambda x: x.label, scope="class")
    def problem(self, request, dtype):
        problem = request.param
        problem.irrep_dtype, problem.weight_dtype = dtype, dtype
        return problem

    @pytest.fixture(scope="class")
    def tp_and_problem(self, problem, extra_tp_constructor_args, with_jax):
        if with_jax:
            pytest.skip("N/A for JAX")
        else:
            tp = oeq.TensorProduct(problem, **extra_tp_constructor_args)
            switch_map = {
                np.float32: torch.float64,
                np.float64: torch.float32,
            }
            tp.to(switch_map[problem.irrep_dtype])
            return tp, tp.config


class TestTorchToSubmodule:
    """Test that TensorProduct works correctly as a submodule when parent's .to() is called"""

    @pytest.fixture(scope="class")
    def parent_module_and_problem(self, dtype, with_jax):
        if with_jax:
            pytest.skip("N/A for JAX")

        problem = mace_problems()[0].clone()
        problem.irrep_dtype, problem.weight_dtype = dtype, dtype

        class ParentModule(torch.nn.Module):
            def __init__(self, problem):
                super().__init__()
                self.tp = oeq.TensorProduct(problem)

            def forward(self, x, y, w):
                return self.tp(x, y, w)

        parent = ParentModule(problem)
        return parent, problem

    def _problem_dtype(self, problem):
        return torch.float32 if problem.irrep_dtype == np.float32 else torch.float64

    def _make_inputs(self, problem, batch_size, rng, dtype, device):
        in1 = torch.tensor(
            rng.uniform(size=(batch_size, problem.irreps_in1.dim)),
            dtype=dtype,
            device=device,
        )
        in2 = torch.tensor(
            rng.uniform(size=(batch_size, problem.irreps_in2.dim)),
            dtype=dtype,
            device=device,
        )
        weights_size = (
            (problem.weight_numel,)
            if problem.shared_weights
            else (batch_size, problem.weight_numel)
        )
        weights = torch.tensor(
            rng.uniform(size=weights_size),
            dtype=dtype,
            device=device,
        )
        return in1, in2, weights

    def test_submodule_dtype_conversion(self, parent_module_and_problem):
        """Test that calling .to() on parent module properly converts TensorProduct submodule"""
        parent, problem = parent_module_and_problem

        # Generate test inputs with the original dtype
        batch_size = 10
        rng = np.random.default_rng(12345)
        device = "cuda"
        input_dtype = self._problem_dtype(problem)
        in1, in2, weights = self._make_inputs(
            problem, batch_size, rng, input_dtype, device
        )

        # Run forward pass with original dtype
        output1 = parent(in1, in2, weights)
        assert output1.dtype == in1.dtype, (
            f"Expected output dtype {in1.dtype}, got {output1.dtype}"
        )

        # Convert parent module to different dtype
        switch_map = {
            np.float32: torch.float64,
            np.float64: torch.float32,
        }
        target_dtype = switch_map[problem.irrep_dtype]
        parent.to(target_dtype)

        # Generate new test inputs with the target dtype
        in1_new, in2_new, weights_new = self._make_inputs(
            problem, batch_size, rng, target_dtype, device
        )

        # This should work but will fail without _apply implementation
        output2 = parent(in1_new, in2_new, weights_new)
        assert output2.dtype == target_dtype, (
            f"Expected output dtype {target_dtype}, got {output2.dtype}"
        )
