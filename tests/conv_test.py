import pytest, tempfile, urllib 
from pytest_check import check

import numpy as np 
import openequivariance as oeq
from openequivariance.benchmark.ConvBenchmarkSuite import load_graph 
from itertools import chain, product

class ConvCorrectness:
    def check_result(self, result, fieldname):
        with check:
            error = result[fieldname]["diff_Linf_norm"]
            thresh = result["thresh"]
            assert result[fieldname]["pass"], f"{fieldname} observed error={error:.2f} >= {thresh}"

    @pytest.fixture(params=[np.float32, 
                            #np.float64
                            ], ids=['F32', 
                                              #'F64'
                                              ], scope='class')
    def dtype(self, request):
        return request.param

    @pytest.fixture(params=["1drf_radius3.5.pickle"], ids=['1drf'], scope='class')
    def graph(self, request):
        download_prefix = "https://portal.nersc.gov/project/m1982/equivariant_nn_graphs/"
        filename = request.param

        #graph = None
        #with tempfile.NamedTemporaryFile() as temp_file:
        #    urllib.request.urlretrieve(download_prefix + filename, temp_file.name)
        #    graph = load_graph(temp_file.name)

        graph = load_graph("data/1drf_radius3.5.pickle")
        return graph

    @pytest.fixture(params=['atomic', 
                            #'deterministic'
                            ], scope='class')
    def conv_object(self, request, problem):
        if request.param == 'atomic':
            return oeq.TensorProductConv(problem, deterministic=False)
        elif request.param == 'deterministic':
            return oeq.TensorProductConv(problem, deterministic=True)

    @pytest.mark.skip
    def test_tp_fwd(self, conv_object, graph):
        result = conv_object.test_correctness_forward(graph, 
                thresh=3e-05,
                prng_seed=12345,
                reference_implementation=None)

        self.check_result(result, "output")

    @pytest.mark.skip
    def test_tp_bwd(self, conv_object, graph):
        result = conv_object.test_correctness_backward(graph, 
                thresh=3e-04,
                prng_seed=12345,
                reference_implementation=None)

        self.check_result(result, "weight_grad")
        self.check_result(result, "in1_grad")
        self.check_result(result, "in2_grad")

    def test_tp_double_bwd(self, conv_object, graph):
        result = conv_object.test_correctness_double_backward(graph, 
                thresh=3e-04,
                prng_seed=12345,
                reference_implementation=None)

        self.check_result(result, "output_grad")
        self.check_result(result, "in1_grad")
        self.check_result(result, "in2_grad")
        self.check_result(result, "weights_grad")

class TestProductionModels(ConvCorrectness):
    from openequivariance.benchmark.benchmark_configs import mace_problems, diffdock_configs 
    production_model_tpps = list(chain(
        mace_problems, 
        diffdock_configs
        ))

    @pytest.fixture(params=production_model_tpps, ids = lambda x : x.label, scope="class")
    def problem(self, request, dtype):
        request.param.irrep_dtype, request.param.weight_dtype = dtype, dtype
        return request.param

 
class TestUVWSingleIrrep(ConvCorrectness):
    muls = [
        (1, 1, 1), (4, 1, 4), (8, 1, 8), (16, 1, 16), 
        (32, 1, 32), (5, 1, 5), (13, 1, 13), (33, 1, 33), (49, 1, 49), (64, 1, 64), 
        (1, 2, 1), (1, 4, 1), (1, 16, 1), (1, 32, 1), (16, 3, 16) 
    ]
    
    irs = [(0, 0, 0), (1, 1, 1), (1, 0, 1), (1, 2, 1), (5, 3, 5), (7, 2, 5) ]

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