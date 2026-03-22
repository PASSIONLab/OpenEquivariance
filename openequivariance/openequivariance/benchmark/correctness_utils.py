from typing import Optional, Union

import numpy as np
import numpy.linalg as la

from openequivariance._torch.CUETensorProduct import CUETensorProduct
from openequivariance.benchmark.logging_utils import bcolors, getLogger
from openequivariance.benchmark.random_buffer_utils import (
    get_random_buffers_backward,
    get_random_buffers_double_backward,
    get_random_buffers_forward,
)
from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.TensorProductBase import TensorProductBase
from openequivariance.core.utils import IrrepLayoutUtils

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
    outputs = []

    for i, impl in enumerate([test_implementation, reference_implementation]):
        is_test_impl = (i == 0)
        tp = instantiate_implementation(impl, problem)
        uses_cue = impl == CUETensorProduct or isinstance(tp, CUETensorProduct)
        run_in1, run_in2, run_weights, run_out = [ buf.copy() for buf in (in1, in2, weights, out) ] 

        if problem.shared_weights and uses_cue:
            run_weights = run_weights[np.newaxis, :]

        # Transpose inputs, if necessary, for the test implementation 
        if is_test_impl:
            run_in1, run_in2 = [
                IrrepLayoutUtils.transpose_irrep_layout(
                    arr, irreps, "mul_ir", tp.config.layout 
                )                for arr, irreps in zip(
                    (run_in1, run_in2), 
                    (problem.irreps_in1, problem.irreps_in2)
                )
            ]

        tp.forward_cpu(L1_in=run_in1, L2_in=run_in2, L3_out=run_out, weights=run_weights)

        if is_test_impl:
            run_out = IrrepLayoutUtils.transpose_irrep_layout(
                run_out, problem.irreps_out, tp.config.layout, "mul_ir"
            )

        outputs.append(run_out)

    for name, to_check, ground_truth in [("output", outputs[0], outputs[1])]:
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

    in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = (
        get_random_buffers_backward(problem, batch_size, prng_seed)
    )

    grads = []
    for i, impl in enumerate([test_implementation, reference_implementation]):
        is_test_impl = i == 0
        tp = instantiate_implementation(impl, problem)

        run_in1, run_in2, run_L3_grad, run_weights, run_weights_grad, run_in1_grad, run_in2_grad = [
            buf.copy()
            for buf in (in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad)
        ]

        uses_cue = impl == CUETensorProduct or isinstance(tp, CUETensorProduct)
        if problem.shared_weights and uses_cue:
            run_weights = run_weights[np.newaxis, :]
            run_weights_grad = run_weights_grad[np.newaxis, :]

        if is_test_impl:
            run_in1, run_in2, run_L3_grad = [
                IrrepLayoutUtils.transpose_irrep_layout(
                    arr, irreps, "mul_ir", tp.config.layout
                )
                for arr, irreps in zip(
                    (run_in1, run_in2, run_L3_grad),
                    (problem.irreps_in1, problem.irreps_in2, problem.irreps_out),
                )
            ]

        tp.backward_cpu(
            L1_in=run_in1,
            L1_grad=run_in1_grad,
            L2_in=run_in2,
            L2_grad=run_in2_grad,
            L3_grad=run_L3_grad,
            weights=run_weights,
            weights_grad=run_weights_grad,
        )

        if is_test_impl:
            run_in1_grad, run_in2_grad = [
                IrrepLayoutUtils.transpose_irrep_layout(
                    arr, irreps, tp.config.layout, "mul_ir"
                )
                for arr, irreps in zip(
                    (run_in1_grad, run_in2_grad),
                    (problem.irreps_in1, problem.irreps_in2),
                )
            ]

        if problem.shared_weights:
            run_weights_grad = run_weights_grad.squeeze()

        grads.append((run_weights_grad, run_in1_grad, run_in2_grad))

    weight_threshold = (
        correctness_threshold * batch_size
        if problem.shared_weights
        else correctness_threshold
    )

    for name, to_check, ground_truth, threshold in [
        ("weight_grad", grads[0][0], grads[1][0], weight_threshold),
        ("in1_grad", grads[0][1], grads[1][1], correctness_threshold),
        ("in2_grad", grads[0][2], grads[1][2], correctness_threshold),
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
    for i, impl in enumerate([test_implementation, reference_implementation]):
        is_test_impl = i == 0
        tp = instantiate_implementation(impl, problem)
        weights_reordered = tp.reorder_weights_from_e3nn(
            weights, has_batch_dim=not problem.shared_weights
        )
        weights_dgrad_reordered = tp.reorder_weights_from_e3nn(
            weights_dgrad, has_batch_dim=not problem.shared_weights
        )

        if impl == CUETensorProduct and problem.shared_weights:
            weights_reordered = weights_reordered[np.newaxis, :]

        db_in1, db_in2, db_out_grad, db_in1_dgrad, db_in2_dgrad = [
            buf.copy() for buf in (in1, in2, out_grad, in1_dgrad, in2_dgrad)
        ]

        if is_test_impl:
            db_in1, db_in2, db_out_grad, db_in1_dgrad, db_in2_dgrad = [
                IrrepLayoutUtils.transpose_irrep_layout(
                    arr, irreps, "mul_ir", tp.config.layout
                )
                for arr, irreps in zip(
                    (db_in1, db_in2, db_out_grad, db_in1_dgrad, db_in2_dgrad),
                    (
                        problem.irreps_in1,
                        problem.irreps_in2,
                        problem.irreps_out,
                        problem.irreps_in1,
                        problem.irreps_in2,
                    ),
                )
            ]

        in1_grad, in2_grad, weights_grad, out_dgrad = tp.double_backward_cpu(
            db_in1,
            db_in2,
            db_out_grad,
            weights_reordered,
            weights_dgrad_reordered,
            db_in1_dgrad,
            db_in2_dgrad,
        )

        if is_test_impl:
            out_dgrad, in1_grad, in2_grad = [
                IrrepLayoutUtils.transpose_irrep_layout(
                    arr, irreps, tp.config.layout, "mul_ir"
                )
                for arr, irreps in zip(
                    (out_dgrad, in1_grad, in2_grad),
                    (problem.irreps_out, problem.irreps_in1, problem.irreps_in2),
                )
            ]

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
