import numpy as np

from src.implementations.e3nn_lite import TPProblem

def get_random_buffers_forward(tpp : TPProblem, batch_size : int,  prng_seed : int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return properly sized numpy arrays needed to execute a tensor product in the forward direction
    Supports shared vs non-shared weights
    """
    assert isinstance(tpp, TPProblem)
    rng = np.random.default_rng(prng_seed)

    in1 = np.array(rng.uniform(size=(batch_size, tpp.irreps_in1.dim)), dtype=tpp.irrep_dtype) 
    in2 = np.array(rng.uniform(size=(batch_size, tpp.irreps_in2.dim)), dtype=tpp.irrep_dtype)

    weights_size = tuple([tpp.weight_numel]) if tpp.shared_weights else tuple([batch_size, tpp.weight_numel]) 
    weights = np.array(rng.uniform(size=weights_size), dtype=tpp.weight_dtype)

    out = np.zeros(shape=(batch_size, tpp.irreps_out.dim), dtype=tpp.weight_dtype)

    return in1, in2, weights, out 

def get_random_buffers_backward(tpp : TPProblem, batch_size : int, prng_seed : int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return properly sized numpy arrays needed to execute a tensor product in the backward direction
    Supports shared vs non-shared weights
    """
    assert isinstance(tpp, TPProblem)
    rng = np.random.default_rng(prng_seed)
    
    in1 = np.array(rng.uniform(size=(batch_size, tpp.irreps_in1.dim)), dtype=tpp.irrep_dtype) 
    in2 = np.array(rng.uniform(size=(batch_size, tpp.irreps_in2.dim)), dtype=tpp.irrep_dtype)
    out_grad = np.array(rng.uniform(size=(batch_size, tpp.irreps_out.dim)), dtype=tpp.irrep_dtype)

    weights_size = tuple([tpp.weight_numel]) if tpp.shared_weights else tuple([batch_size, tpp.weight_numel])
    weights = np.array(rng.uniform(size=weights_size), dtype=tpp.irrep_dtype)

    weights_grad = np.zeros_like(weights)
    in1_grad = np.zeros_like(in1)
    in2_grad = np.zeros_like(in2) 

    return in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad