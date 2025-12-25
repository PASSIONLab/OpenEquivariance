from typing import Optional, Union

from openequivariance.core.TensorProductBase import TensorProductBase
from openequivariance.core.e3nn_lite import TPProblem
from openequivariance._torch.CUETensorProduct import CUETensorProduct
from openequivariance.benchmark.random_buffer_utils import (
    get_random_buffers_forward,
    get_random_buffers_backward,
    get_random_buffers_double_backward,
)

from openequivariance.benchmark.logging_utils import getLogger, bcolors
import numpy as np
import numpy.linalg as la

logger = getLogger()


def check_similiarity(
    name: str,
    to_check: np.ndarray,
    ground_truth: np.ndarray,
    correctness_threshold: float,
):
    result = {}
    if to_check.shape != ground_truth.shape:
        result["shape_match"] = False
        result["diff_Linf_norm"] = np.inf
        result["pass"] = False
        logger.error(
            f"{bcolors.FAIL}Ground truth {name} shape does not match input! {to_check.shape=}, {ground_truth.shape=} {bcolors.ENDC}"
        )
    else:
        result["shape_match"] = True
        diff_Linf_norm = float(la.norm((ground_truth - to_check).flatten(), ord=np.inf))
        result["diff_Linf_norm"] = diff_Linf_norm
        result["pass"] = bool(diff_Linf_norm < correctness_threshold)
        if result["pass"]:
            logger.info(
                f" {bcolors.OKGREEN}{name} correctness check pass. {diff_Linf_norm=:.3e}, {correctness_threshold=} {bcolors.ENDC}"
            )
        else:
            logger.error(
                f"{bcolors.FAIL}{name} correctness check fail! {diff_Linf_norm=:.3e}, {correctness_threshold=} {bcolors.ENDC}"
            )

    return result


def instantiate_implementation(
    implementation: Union[type[TensorProductBase], TensorProductBase],
    problem: TPProblem,
):
    if isinstance(implementation, type):
        test_tp = implementation(problem)
    else:
        test_tp = implementation

    if not isinstance(test_tp, TensorProductBase):
        raise TypeError(
            f"test_implementation must be a TensorProductBase or a subclass, got {type(implementation)}"
        )

    return test_tp


def correctness_forward(
    problem: TPProblem,
    test_implementation: Union[type[TensorProductBase], TensorProductBase],
    reference_implementation: Optional[type[TensorProductBase]],
    batch_size: int,
    correctness_threshold: float,
    prng_seed: int,
) -> dict:
    if reference_implementation is None:
        from openequivariance._torch.E3NNTensorProduct import E3NNTensorProduct

        reference_implementation = E3NNTensorProduct

    result = {"thresh": correctness_threshold, "batch_size": batch_size}

    in1, in2, weights, out = get_random_buffers_forward(problem, batch_size, prng_seed)

    # run reference
    ref_tp = reference_implementation(problem)

    ref_out = out.copy()
    ref_tp.forward_cpu(
        L1_in=in1.copy(), L2_in=in2.copy(), L3_out=ref_out, weights=weights.copy()
    )

    weights_copy = weights.copy()
    if problem.shared_weights and test_implementation == CUETensorProduct:
        weights_copy = weights[np.newaxis, :]

    # run test
    test_tp = instantiate_implementation(test_implementation, problem)
    test_out = out.copy()
    test_tp.forward_cpu(
        L1_in=in1.copy(), L2_in=in2.copy(), L3_out=test_out, weights=weights_copy
    )

    for name, to_check, ground_truth in [("output", ref_out, test_out)]:
        result[name] = check_similiarity(
            name, to_check, ground_truth, correctness_threshold
        )

    return result


def correctness_backward(
    problem: TPProblem,
    test_implementation: Union[type[TensorProductBase], TensorProductBase],
    reference_implementation: Optional[type[TensorProductBase]],
    batch_size: int,
    correctness_threshold: float,
    prng_seed: int,
) -> dict:
    if reference_implementation is None:
        from openequivariance._torch.E3NNTensorProduct import E3NNTensorProduct

        reference_implementation = E3NNTensorProduct

    result = {"thresh": correctness_threshold, "batch_size": batch_size}

    # run reference
    in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = (
        get_random_buffers_backward(problem, batch_size, prng_seed)
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
        weights_grad=ref_weights_grad,
    )

    # run test version
    test_weights_grad = weights_grad.copy()
    test_in1_grad = in1_grad.copy()
    test_in2_grad = in2_grad.copy()

    weights_copy = weights.copy()

    if problem.shared_weights and test_implementation == CUETensorProduct:
        weights_copy = weights[np.newaxis, :]
        test_weights_grad = test_weights_grad[np.newaxis, :]

    test_tp = instantiate_implementation(test_implementation, problem)
    test_tp.backward_cpu(
        L1_in=in1.copy(),
        L1_grad=test_in1_grad,
        L2_in=in2.copy(),
        L2_grad=test_in2_grad,
        L3_grad=out_grad.copy(),
        weights=weights_copy,
        weights_grad=test_weights_grad,
    )

    weight_threshold = (
        correctness_threshold * batch_size
        if problem.shared_weights
        else correctness_threshold
    )

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
    problem: TPProblem,
    test_implementation: Union[type[TensorProductBase], TensorProductBase],
    reference_implementation: Optional[type[TensorProductBase]],
    batch_size: int,
    correctness_threshold: float,
    prng_seed: int,
):
    global torch
    import torch

    in1, in2, out_grad, weights, weights_dgrad, in1_dgrad, in2_dgrad, _ = (
        get_random_buffers_double_backward(
            problem, batch_size=batch_size, prng_seed=prng_seed
        )
    )

    if reference_implementation is None:
        from openequivariance._torch.E3NNTensorProduct import E3NNTensorProduct

        reference_implementation = E3NNTensorProduct

    result = {"thresh": correctness_threshold, "batch_size": batch_size}

    tensors = []
    for _, impl in enumerate([test_implementation, reference_implementation]):
        tp = instantiate_implementation(impl, problem)
        weights_reordered = tp.reorder_weights_from_e3nn(
            weights, has_batch_dim=not problem.shared_weights
        )
        weights_dgrad_reordered = tp.reorder_weights_from_e3nn(
            weights_dgrad, has_batch_dim=not problem.shared_weights
        )

        if impl == CUETensorProduct and problem.shared_weights:
            weights_reordered = weights_reordered[np.newaxis, :]

        in1_grad, in2_grad, weights_grad, out_dgrad = tp.double_backward_cpu(
            in1,
            in2,
            out_grad,
            weights_reordered,
            weights_dgrad_reordered,
            in1_dgrad,
            in2_dgrad,
        )
        tensors.append(
            (
                out_dgrad,
                in1_grad,
                in2_grad,
                tp.reorder_weights_to_e3nn(
                    weights_grad, has_batch_dim=not problem.shared_weights
                ),
            )
        )

    for name, to_check, ground_truth in [
        ("output_double_grad", tensors[0][0], tensors[1][0]),
        ("in1_grad", tensors[0][1], tensors[1][1]),
        ("in2_grad", tensors[0][2], tensors[1][2]),
        ("weights_grad", tensors[0][3], tensors[1][3]),
    ]:
        result[name] = check_similiarity(
            name, to_check, ground_truth, correctness_threshold
        )

    return result
