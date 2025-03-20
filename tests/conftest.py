from itertools import chain

import pytest

from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance.benchmark.benchmark_configs import e3nn_torch_tetris_polynomial, diffdock_configs

real_world_tpps = list(chain(
        e3nn_torch_tetris_polynomial, 
        diffdock_configs,

    ))

@pytest.fixture(params=real_world_tpps, ids = lambda x : x.label)
def real_world_tpp(request): 
    return request.param

@pytest.fixture(params=[TensorProduct])
def test_impl(request):
    return request.param
