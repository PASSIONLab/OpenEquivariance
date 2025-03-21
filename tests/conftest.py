from itertools import chain

import pytest

from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance.benchmark.benchmark_configs import e3nn_torch_tetris_polynomial, diffdock_configs

production_model_tpps = list(chain(
        e3nn_torch_tetris_polynomial, 
        diffdock_configs,

    ))

@pytest.fixture(params=production_model_tpps, ids = lambda x : x.label)
def production_model_tpp(request): 
    return request.param

@pytest.fixture(params=[TensorProduct])
def test_impl(request):
    return request.param
