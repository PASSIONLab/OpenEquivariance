import pytest
import tempfile
import urllib
from pytest_check import check

import numpy as np
import openequivariance as oeq
from openequivariance.benchmark.ConvBenchmarkSuite import load_graph
from itertools import product
import torch

from openequivariance.benchmark.problems import (
    mace_problems,
    diffdock_problems,
    e3tools_problems,
)

@pytest.fixture(params=[np.float32, np.float64], ids=["F32", "F64"], scope="module")
def dtype(request):
    return request.param

@pytest.fixture(params=["1drf_radius3.5.pickle"], ids=["1drf"], scope="module")
def graph(request):
    download_prefix = (
        "https://portal.nersc.gov/project/m1982/equivariant_nn_graphs/"
    )
    filename = request.param

    graph = None
    with tempfile.NamedTemporaryFile() as temp_file:
        urllib.request.urlretrieve(download_prefix + filename, temp_file.name)
        graph = load_graph(temp_file.name)

    # graph = load_graph("data/1drf_radius3.5.pickle")
    return graph

@pytest.fixture(scope="module")
def with_jax(request):
    return request.config.getoption("--jax")

class ConvCorrectness:
    def thresh(self, direction):
        return {"fwd": 3e-4, "bwd": 3e-4, "double_bwd": 3e-4}[direction]

    def check_result(self, result, fieldname):
        with check:
            error = result[fieldname]["diff_Linf_norm"]
            thresh = result["thresh"]
            assert result[fieldname]["pass"], (
                f"{fieldname} observed error={error:.5f} >= {thresh}"
            )

    @pytest.fixture(scope="class")
    def extra_conv_constructor_args(self):
        return {}

    @pytest.fixture(params=["atomic", "deterministic", "kahan"], scope="class")
    def conv_object(self, request, problem, extra_conv_constructor_args, with_jax):
        cls = oeq.TensorProductConv
        if with_jax:
            from openequivariance.jax import TensorProductConv as jax_conv

            cls = jax_conv

        if request.param == "atomic":
            return cls(problem, deterministic=False, **extra_conv_constructor_args)
        elif request.param == "deterministic":
            if not problem.shared_weights:
                return cls(problem, deterministic=True, **extra_conv_constructor_args)
            else:
                pytest.skip("Shared weights not supported with deterministic")
        elif request.param == "kahan":
            if problem.irrep_dtype == np.float32:
                if not problem.shared_weights:
                    return cls(
                        problem,
                        deterministic=True,
                        kahan=True,
                        **extra_conv_constructor_args,
                    )
                else:
                    pytest.skip("Shared weights not supported with kahan")
            else:
                pytest.skip("Only Float32 supported with kahan")

    def test_tp_fwd(self, conv_object, graph):
        if conv_object is None:
            pytest.skip("'conv_object' fixture returned None, skipping")

        result = conv_object.test_correctness_forward(
            graph,
            thresh=self.thresh("fwd"),
            prng_seed=12345,
            reference_implementation=None,
        )

        self.check_result(result, "output")

    def test_tp_bwd(self, conv_object, graph):
        if conv_object is None:
            pytest.skip("'conv_object' fixture returned None, skipping")

        result = conv_object.test_correctness_backward(
            graph,
            thresh=self.thresh("bwd"),
            prng_seed=12345,
            reference_implementation=None,
        )

        self.check_result(result, "weight_grad")
        self.check_result(result, "in1_grad")
        self.check_result(result, "in2_grad")

    def test_tp_double_bwd(self, conv_object, graph):
        if conv_object is None:
            pytest.skip("'conv_object' fixture returned None, skipping")

        result = conv_object.test_correctness_double_backward(
            graph,
            thresh=self.thresh("double_bwd"),
            prng_seed=12345,
            reference_implementation=None,
        )

        self.check_result(result, "output_grad")
        self.check_result(result, "in1_grad")
        self.check_result(result, "in2_grad")
        self.check_result(result, "weights_grad")


class TestProductionModels(ConvCorrectness):
    production_model_tpps = (
        mace_problems() + diffdock_problems() + [e3tools_problems()[0]]
    )

    @pytest.fixture(params=production_model_tpps, ids=lambda x: x.label, scope="class")
    def problem(self, request, dtype):
        request.param.irrep_dtype, request.param.weight_dtype = dtype, dtype
        return request.param


class TestUVUSingleIrrep(ConvCorrectness):
    muls = [
        (1, 1, 1),
        (8, 1, 8),
        (16, 1, 16),
        (32, 1, 32),
        (5, 1, 5),
        (13, 1, 13),
        (19, 1, 19),
        (33, 1, 33),
        (49, 1, 49),
        (128, 1, 128),
        (1, 2, 1),
        (1, 16, 1),
        (1, 32, 1),
        (16, 3, 16),
    ]

    irs = [(0, 0, 0), (1, 1, 1), (1, 0, 1), (1, 2, 1), (2, 0, 2), (5, 3, 5), (7, 2, 5)]

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


class TestUVWSingleIrrep(ConvCorrectness):
    muls = [
        (1, 1, 1),
        (4, 1, 4),
        (8, 1, 8),
        (16, 1, 16),
        (32, 1, 32),
        (5, 1, 5),
        (13, 1, 13),
        (33, 1, 33),
        (49, 1, 49),
        (64, 1, 64),
        (1, 2, 1),
        (1, 4, 1),
        (1, 16, 1),
        (1, 32, 1),
        (16, 3, 16),
    ]

    irs = [(0, 0, 0), (1, 1, 1), (1, 0, 1), (1, 2, 1), (5, 3, 5), (7, 2, 5)]

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


class TestAtomicSharedWeights(ConvCorrectness):
    problems = [mace_problems()[0], diffdock_problems()[0], e3tools_problems()[1]]

    def thresh(self, direction):
        return {
            "fwd": 1e-5,
            "bwd": 7.5e-2,  # Expect higher errors for shared weights
            "double_bwd": 5e-1,
        }[direction]

    @pytest.fixture(params=problems, ids=lambda x: x.label, scope="class")
    def problem(self, request, dtype):
        problem = request.param
        problem.irrep_dtype, problem.weight_dtype = dtype, dtype
        problem.shared_weights = True
        return problem

    @pytest.fixture(scope="class")
    def conv_object(self, request, problem):
        return oeq.TensorProductConv(problem, deterministic=False)


class TestTorchbindDisable(TestProductionModels):
    @pytest.fixture(scope="class")
    def extra_conv_constructor_args(self, with_jax):
        if with_jax:
            pytest.skip("N/A for JAX")
        return {"use_opaque": True}


class TestTorchTo(ConvCorrectness):
    problems = [mace_problems()[0]]

    @pytest.fixture(params=problems, ids=lambda x: x.label, scope="class")
    def problem(self, request, dtype, with_jax):
        if with_jax:
            pytest.skip("N/A for JAX")

        problem = request.param
        problem.irrep_dtype, problem.weight_dtype = dtype, dtype
        return problem

    @pytest.fixture(params=["atomic", "deterministic"], scope="class")
    def conv_object(self, request, problem, extra_conv_constructor_args):
        switch_map = {
            np.float32: torch.float64,
            np.float64: torch.float32,
        }
        module = oeq.TensorProductConv(
            problem,
            deterministic=(request.param == "deterministic"),
            **extra_conv_constructor_args,
        )
        return module.to(switch_map[problem.irrep_dtype])

class TestTorchToSubmodule:
    """Test that TensorProductConv works as a submodule when parent's .to() is called"""

    @pytest.fixture(params=["atomic", "deterministic"], scope="class")
    def parent_module_and_problem(
        self,
        request,
        dtype,
        with_jax
    ):
        if with_jax:
            pytest.skip("N/A for JAX")

        problem = mace_problems()[0].clone()
        problem.irrep_dtype, problem.weight_dtype = dtype, dtype
        deterministic = request.param == "deterministic"

        class ParentModule(torch.nn.Module):
            def __init__(self, problem, deterministic):
                super().__init__()
                self.conv = oeq.TensorProductConv(
                    problem,
                    deterministic=deterministic
                )

            def forward(self, x, y, w, rows, cols, sender_perm=None):
                return self.conv(x, y, w, rows, cols, sender_perm)

        parent = ParentModule(problem, deterministic)
        return parent, problem

    def _problem_dtype(self, problem):
        return torch.float32 if problem.irrep_dtype == np.float32 else torch.float64

    def _make_inputs(self, problem, graph, rng, dtype, device, deterministic):
        node_count = graph.node_count
        nnz = graph.nnz

        in1 = torch.tensor(
            rng.uniform(size=(node_count, problem.irreps_in1.dim)),
            dtype=dtype,
            device=device,
        )
        in2 = torch.tensor(
            rng.uniform(size=(nnz, problem.irreps_in2.dim)),
            dtype=dtype,
            device=device,
        )
        weights_size = (
            (problem.weight_numel,)
            if problem.shared_weights
            else (nnz, problem.weight_numel)
        )
        weights = torch.tensor(
            rng.uniform(size=weights_size),
            dtype=dtype,
            device=device,
        )

        rows = torch.tensor(graph.rows, device=device)
        cols = torch.tensor(graph.cols, device=device)
        sender_perm = (
            torch.tensor(graph.transpose_perm, device=device)
            if deterministic
            else None
        )
        return in1, in2, weights, rows, cols, sender_perm

    def test_submodule_dtype_conversion(self, parent_module_and_problem, graph):
        parent, problem = parent_module_and_problem
        device = "cuda"

        rng = np.random.default_rng(12345)
        input_dtype = self._problem_dtype(problem)
        in1, in2, weights, rows, cols, sender_perm = self._make_inputs(
            problem,
            graph,
            rng,
            input_dtype,
            device,
            parent.conv.deterministic,
        )

        output1 = parent(in1, in2, weights, rows, cols, sender_perm)
        assert output1.dtype == input_dtype, (
            f"Expected output dtype {input_dtype}, got {output1.dtype}"
        )

        switch_map = {
            np.float32: torch.float64,
            np.float64: torch.float32,
        }
        target_dtype = switch_map[problem.irrep_dtype]
        parent.to(target_dtype)

        in1_new, in2_new, weights_new, rows, cols, sender_perm = self._make_inputs(
            problem,
            graph,
            rng,
            target_dtype,
            device,
            parent.conv.deterministic,
        )

        output2 = parent(in1_new, in2_new, weights_new, rows, cols, sender_perm)
        assert output2.dtype == target_dtype, (
            f"Expected output dtype {target_dtype}, got {output2.dtype}"
        )
