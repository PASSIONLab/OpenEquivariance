from typing import Optional

from openequivariance.implementations.TensorProductBase import TensorProductBase
from openequivariance.implementations.CUETensorProduct import CUETensorProduct 
from openequivariance.implementations.e3nn_lite import TPProblem
from openequivariance.benchmark.random_buffer_utils import get_random_buffers_forward, get_random_buffers_backward
from openequivariance.benchmark.logging_utils import getLogger, bcolors 
import numpy as np 
import numpy.linalg as la

logger = getLogger()

def check_similiarity(name : str,  to_check : np.ndarray,  ground_truth : np.ndarray, correctness_threshold : float):
    result = {}
    if to_check.shape != ground_truth.shape:
        result["shape_match"] = False
        result["diff_Linf_norm"] = np.inf
        result["pass"] = False
        logger.error(f"{bcolors.FAIL}Ground truth {name} shape does not match input! {to_check.shape=}, {ground_truth.shape=} {bcolors.ENDC}")
    else:
        result["shape_match"] = True 
        diff_Linf_norm = float(la.norm((ground_truth - to_check).flatten(), ord=np.inf))
        result["diff_Linf_norm"] = diff_Linf_norm 
        result["pass"] = bool(diff_Linf_norm < correctness_threshold) 

        if result["pass"]:
            logger.info(f" {bcolors.OKGREEN}{name} correctness check pass. {diff_Linf_norm=:.3e}, {correctness_threshold=} {bcolors.ENDC}")
        else:
            logger.error(f"{bcolors.FAIL}{name} correctness check fail! {diff_Linf_norm=:.3e}, {correctness_threshold=} {bcolors.ENDC}")

    return result

def correctness_forward(
        problem : TPProblem,  
        test_implementation : type[TensorProductBase], 
        reference_implementation : Optional[type[TensorProductBase]], 
        batch_size : int, 
        correctness_threshold : float,
        prng_seed : int,
        ) -> dict:
    
    if reference_implementation is None:
        from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProduct
        reference_implementation = E3NNTensorProduct

    result = {
        "thresh": correctness_threshold, 
        "batch_size": batch_size
    }
 
    in1, in2, weights, out = get_random_buffers_forward(problem, batch_size, prng_seed)

    # run reference
    ref_tp = reference_implementation(problem)

    ref_out = out.copy()
    ref_tp.forward_cpu(
        L1_in=in1.copy(), 
        L2_in=in2.copy(), 
        L3_out=ref_out, 
        weights=weights.copy()) 

    weights_copy = weights.copy()
    if problem.shared_weights and test_implementation == CUETensorProduct:
        weights_copy = weights[np.newaxis, :]

    # run test
    test_tp = test_implementation(problem)
    test_out = out.copy()
    test_tp.forward_cpu(
        L1_in=in1.copy(), 
        L2_in=in2.copy(), 
        L3_out=test_out, 
        weights=weights_copy)

    for name, to_check, ground_truth in [
        ("output", ref_out, test_out)
        ]:
        result[name] = check_similiarity(name, to_check, ground_truth, correctness_threshold)
    
    return result

def correctness_backward(
        problem : TPProblem,  
        test_implementation : type[TensorProductBase], 
        reference_implementation : Optional[type[TensorProductBase]], 
        batch_size : int, 
        correctness_threshold : float,
        prng_seed : int,
        ) -> dict:
    
    if reference_implementation is None:
        from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProduct
        reference_implementation = E3NNTensorProduct

    result = {
        "thresh": correctness_threshold,
        "batch_size": batch_size
    }
    
    # run reference
    in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = get_random_buffers_backward(
        problem, 
        batch_size, 
        prng_seed
    )

    ref_tp = reference_implementation(problem)

    ref_weights_grad = weights_grad.copy()
    ref_in1_grad = in1_grad.copy()
    ref_in2_grad = in2_grad.copy()

    ref_tp.backward_cpu(
        L1_in=in1.copy(),
        L1_grad=ref_in1_grad,
        L2_in=in2.copy(), 
        L2_grad=ref_in2_grad, 
        L3_grad=out_grad.copy(), 
        weights=weights.copy(), 
        weights_grad=ref_weights_grad
        )

    # run test version
    test_weights_grad = weights_grad.copy()
    test_in1_grad = in1_grad.copy()
    test_in2_grad = in2_grad.copy()

    weights_copy = weights.copy()
    weights_grad_copy = weights_grad.copy()
    if problem.shared_weights and test_implementation == CUETensorProduct:
        weights_copy = weights[np.newaxis, :]
        test_weights_grad = test_weights_grad[np.newaxis, :] 

    test_tp = test_implementation(problem)
    test_tp.backward_cpu(
        L1_in=in1.copy(),
        L1_grad=test_in1_grad,
        L2_in=in2.copy(), 
        L2_grad=test_in2_grad, 
        L3_grad=out_grad.copy(), 
        weights=weights_copy,
        weights_grad=test_weights_grad
        )

    weight_threshold = correctness_threshold * batch_size if problem.shared_weights else correctness_threshold

    if problem.shared_weights:
        test_weights_grad = test_weights_grad.squeeze()

    for name, to_check, ground_truth, threshold in [
        ("weight_grad", test_weights_grad, ref_weights_grad, weight_threshold),
        ("in1_grad", test_in1_grad, ref_in1_grad, correctness_threshold),
        ("in2_grad", test_in2_grad, ref_in2_grad, correctness_threshold),
        ]:
        result[name] = check_similiarity(name, to_check, ground_truth, threshold)

    return result

def correctness_double_backward(
        problem : TPProblem,  
        test_implementation : type[TensorProductBase], 
        reference_implementation : Optional[type[TensorProductBase]], 
        batch_size : int, 
        correctness_threshold : float,
        prng_seed : int):

    global torch
    import torch
    
    in1, in2, out_grad, weights, _, _, _ = get_random_buffers_backward(
        problem, 
        batch_size, 
        prng_seed
    )
    rng = np.random.default_rng(seed=prng_seed * 2)
    dummy_grad = rng.standard_normal(1) 
 
    if reference_implementation is None:
        from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProduct
        reference_implementation = E3NNTensorProduct

    result = {
        "thresh": correctness_threshold,
        "batch_size": batch_size
    }

    tensors = []
    for impl in [test_implementation, reference_implementation]:
        tp = impl(problem, torch_op=True)

        print(f"testing impl: {tp.name()}")
        if impl == CUETensorProduct and problem.shared_weights :
            weights = weights[np.newaxis, :] 
            
        in1_torch = torch.tensor(in1, device='cuda', requires_grad=True)
        in2_torch = torch.tensor(in2, device='cuda', requires_grad=True)
        weights_torch = torch.tensor(weights, device='cuda', requires_grad=True)

        out_torch = tp.forward(in1_torch, in2_torch, weights_torch)
        out_grad = out_torch.clone().detach().to(device='cuda').requires_grad_(True)

        out_torch.backward(out_grad, 
            create_graph=True,
            retain_graph=True,
            inputs=[in1_torch, in2_torch, weights_torch])

        dummy = torch.norm(in1_torch.grad) + torch.norm(in2_torch.grad) + torch.norm(weights_torch.grad)
        dummy_grad = torch.tensor(float(dummy_grad), device='cuda', requires_grad=True)
        dummy.backward(dummy_grad,
            retain_graph=True, 
            inputs=[out_grad, in1_torch, in2_torch, weights_torch])

        temp = weights_torch.grad.detach().cpu().numpy()

        # if impl == CUETensorProduct and problem.shared_weights :
        #     temp = temp.squeeze()

        tensors.append((
            out_grad.grad.detach().cpu().numpy(),
            in1_torch.grad.detach().cpu().numpy(),
            in2_torch.grad.detach().cpu().numpy(),
            temp
        ))

        

    for name, to_check, ground_truth in [
        ("output_double_grad", tensors[0][0], tensors[1][0]),
        ("in1_grad", tensors[0][1], tensors[1][1]),
        ("in2_grad", tensors[0][2], tensors[1][2]),
        ("weights_grad", tensors[0][3], tensors[1][3])
        ]:
        result[name] = check_similiarity(name, to_check, ground_truth, correctness_threshold)

    return result