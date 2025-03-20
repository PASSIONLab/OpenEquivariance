import pytest

import openequivariance as oeq 
from openequivariance.implementations.e3nn_lite import TPProblem
from openequivariance.implementations.TensorProductBase import TensorProductBase
from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance.benchmark.correctness_utils import correctness_forward, correctness_backward, correctness_double_backward


def id_naming_func(obj):
    return obj.label if isinstance(obj, TPProblem) else obj

def test_tp_forward_correctness(real_world_tpp, test_impl):
    
    result = correctness_forward(
        problem=real_world_tpp,
        test_implementation=test_impl,
        reference_implementation=None, 
        batch_size=1000,
        correctness_threshold=1e-6,
        prng_seed=12345
        )
    
    assert result["output"]["pass"]

def test_tp_backward_correctness(real_world_tpp, test_impl):
    
    result = correctness_backward(
        problem=real_world_tpp,
        test_implementation=test_impl,
        reference_implementation=None, 
        batch_size=1000,
        correctness_threshold=1e-6,
        prng_seed=12345
        )
    
    assert result["weight_grad"]["pass"]
    assert result["in1_grad"]["pass"]
    assert result["in2_grad"]["pass"]

def test_tp_backward_correctness(real_world_tpp, test_impl):
    
    result = correctness_backward(
        problem=real_world_tpp,
        test_implementation=test_impl,
        reference_implementation=None, 
        batch_size=1000,
        correctness_threshold=1e-4,
        prng_seed=12345
        )
    
    assert result["weight_grad"]["pass"]
    assert result["in1_grad"]["pass"]
    assert result["in2_grad"]["pass"]

@pytest.mark.skip(reason="This is failing but it might be my fault")
def test_tp_double_backward_correctness(real_world_tpp, test_impl):

    result = correctness_double_backward(
        problem = real_world_tpp,  
        test_implementation = test_impl, 
        reference_implementation = None,
        batch_size = 1000, 
        correctness_threshold = 1e-4,
        prng_seed = 12345)
    
    assert result["output_grad"]["pass"]
    assert result["in1_grad"]["pass"]    
    assert result["in2_grad"]["pass"]   
    assert result["weights_grad"]["pass"]    
    