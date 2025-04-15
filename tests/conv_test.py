import pytest, tempfile, urllib 
from pytest_check import check

import numpy as np 
import openequivariance as oeq
from openequivariance.benchmark.ConvBenchmarkSuite import load_graph 
from itertools import chain, product

@pytest.fixture(params=[np.float32, np.float64], ids=['F32', 'F64'])
def dtype(request):
    return request.param

@pytest.fixture(params=["1drf_radius6.0.pickle"], ids=['1drf'])
def graph(request):
    download_prefix = "https://portal.nersc.gov/project/m1982/equivariant_nn_graphs/"
    filename = request.param

    graph = None
    with tempfile.NamedTemporaryFile() as temp_file:
        urllib.request.urlretrieve(download_prefix + filename, temp_file.name)
        graph = load_graph(temp_file.name)

    return graph

class ConvCorrectness:
    def check_result(self, result, fieldname):
        with check:
            error = result[fieldname]["diff_Linf_norm"]
            thresh = result["thresh"]
            assert result[fieldname]["pass"], f"{fieldname} observed error={error:.2f} >= {thresh}"

    @pytest.fixture(params=['atomic', 'deterministic'])
    def conv_object(self, request, problem):
        if request.param == 'atomic':
            return oeq.TensorProductConv(problem, deterministic=False)
        elif request.param == 'deterministic':
            return oeq.TensorProductConv(problem, deterministic=True)

    def test_tp_fwd(self, conv_object, graph):
        result = conv_object.test_correctness_forward(graph, 
                thresh=3e-05,
                prng_seed=12345,
                reference_implementation=None)

        self.check_result(result, "output")

    #def test_tp_bwd(self, problem, implementation):
    #    pass
        #result = correctness_backward(
        #    problem=problem,
        #    test_implementation=implementation,
        #    reference_implementation=None, 
        #    batch_size=1000,
        #    correctness_threshold=3e-4,
        #    prng_seed=12345)

        #self.check_result(result, "weight_grad")
        #self.check_result(result, "in1_grad")
        #self.check_result(result, "in2_grad")

    #@pytest.mark.skip(reason="Need to add weight reordering in double-backward")
    #def test_tp_double_bwd(self, problem, implementation):
    #    pass
        #result = correctness_double_backward(
        #    problem = problem,
        #    test_implementation = implementation,
        #    reference_implementation = None,
        #    batch_size = 1000,
        #    correctness_threshold = 3e-4,
        #    prng_seed = 12345)

        #self.check_result(result, "output_grad")
        #self.check_result(result, "in1_grad")
        #self.check_result(result, "in2_grad")
        #self.check_result(result, "weights_grad")

class TestProductionModels(ConvCorrectness):
    from openequivariance.benchmark.benchmark_configs \
            import e3nn_torch_tetris_polynomial, diffdock_configs, mace_nequip_problems
    production_model_tpps = list(chain(
            mace_nequip_problems,
            e3nn_torch_tetris_polynomial, 
            diffdock_configs))[:1]

    @pytest.fixture(params=production_model_tpps, ids = lambda x : x.label)
    def problem(self, request, dtype):
        request.param.irrep_dtype, request.param.weight_dtype = dtype, dtype
        return request.param