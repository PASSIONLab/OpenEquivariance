import copy
from typing import Optional, Union

import numpy as np
import numpy.linalg as la

from openequivariance._torch.CUETensorProduct import CUETensorProduct
from openequivariance.benchmark.logging import bcolors, getLogger
from openequivariance.benchmark.random_buffer_utils import (
    get_random_buffers_backward_conv,
    get_random_buffers_backward,
    get_random_buffers_double_backward_conv,
    get_random_buffers_double_backward,
    get_random_buffers_forward_conv,
    get_random_buffers_forward,
)
from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.TensorProductBase import TensorProductBase
from openequivariance.core.utils import transpose_irrep_layout

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
        is_test_impl = i == 0
        tp = instantiate_implementation(impl, problem)
        uses_cue = impl == CUETensorProduct or isinstance(tp, CUETensorProduct)
        run_in1, run_in2, run_weights, run_out = [
            buf.copy() for buf in (in1, in2, weights, out)
        ]

        if problem.shared_weights and uses_cue:
            run_weights = run_weights[np.newaxis, :]

        # Transpose inputs, if necessary, for the test implementation
        if is_test_impl:
            run_in1, run_in2 = [
                transpose_irrep_layout(arr, irreps, "mul_ir", tp.config.layout)
                for arr, irreps in zip(
                    (run_in1, run_in2), (problem.irreps_in1, problem.irreps_in2)
                )
            ]

        tp.forward_cpu(
            L1_in=run_in1, L2_in=run_in2, L3_out=run_out, weights=run_weights
        )

        if is_test_impl:
            run_out = transpose_irrep_layout(
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

        (
            run_in1,
            run_in2,
            run_L3_grad,
            run_weights,
            run_weights_grad,
            run_in1_grad,
            run_in2_grad,
        ) = [
            buf.copy()
            for buf in (in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad)
        ]

        uses_cue = impl == CUETensorProduct or isinstance(tp, CUETensorProduct)
        if problem.shared_weights and uses_cue:
            run_weights = run_weights[np.newaxis, :]
            run_weights_grad = run_weights_grad[np.newaxis, :]

        if is_test_impl:
            run_in1, run_in2, run_L3_grad = [
                transpose_irrep_layout(arr, irreps, "mul_ir", tp.config.layout)
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
                transpose_irrep_layout(arr, irreps, tp.config.layout, "mul_ir")
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
                transpose_irrep_layout(arr, irreps, "mul_ir", tp.config.layout)
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
                transpose_irrep_layout(arr, irreps, tp.config.layout, "mul_ir")
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


def correctness_forward_conv(
    conv,
    graph,
    thresh,
    prng_seed,
    reference_implementation=None,
    check_reproducible=True,
    high_precision_ref=False,
):
    global torch
    import torch

    if reference_implementation is None:
        from openequivariance._torch.E3NNConv import E3NNConv

        reference_implementation = E3NNConv

    result = {"thresh": thresh}

    in1, in2, weights, out = get_random_buffers_forward_conv(
        conv.config, graph.node_count, graph.nnz, prng_seed
    )
    reference_config = conv.config
    if high_precision_ref:
        reference_config = copy.deepcopy(conv.config)
        reference_config.irrep_dtype = np.float64
        reference_config.weight_dtype = np.float64
    outputs = []

    for i, impl in enumerate([conv, reference_implementation]):
        is_test_impl = i == 0
        tp = impl if is_test_impl else impl(reference_config)

        run_in1, run_in2, run_weights, run_out = [
            buf.copy() for buf in (in1, in2, weights, out)
        ]

        if not is_test_impl and high_precision_ref:
            run_in1, run_in2, run_weights, run_out = [
                np.array(el, dtype=np.float64)
                for el in (run_in1, run_in2, run_weights, run_out)
            ]

        if is_test_impl:
            run_in1, run_in2 = [
                transpose_irrep_layout(arr, irreps, "mul_ir", conv.config.layout)
                for arr, irreps in zip(
                    (run_in1, run_in2),
                    (conv.config.irreps_in1, conv.config.irreps_in2),
                )
            ]
            conv.forward_cpu(
                L1_in=run_in1,
                L2_in=run_in2,
                weights=run_weights,
                L3_out=run_out,
                graph=graph,
            )

            run_out = transpose_irrep_layout(
                run_out, conv.config.irreps_out, conv.config.layout, "mul_ir"
            )
        else:
            args = {
                "L1_in": run_in1,
                "L2_in": run_in2,
                "weights": run_weights,
                "rows": graph.rows,
                "cols": graph.cols,
            }

            if tp.deterministic:
                args["transpose_perm"] = graph.transpose_perm

            for key in args:
                args[key] = torch.tensor(args[key], device="cuda")

            run_out[:] = tp.forward(**args).cpu().numpy()

        outputs.append(run_out)

    test_out, ref_out = outputs[0], outputs[1]

    for name, to_check, ground_truth in [("output", ref_out, test_out)]:
        result[name] = check_similiarity(name, to_check, ground_truth, thresh)

    if check_reproducible:
        num_trials = 5
        for name in ["output"]:
            result[name]["num_reproducibility_trials"] = num_trials
            result[name]["bitwise_reproducible"] = True

        for _ in range(num_trials):
            repeated_run = out.copy()
            rep_in1, rep_in2, rep_weights = [buf.copy() for buf in (in1, in2, weights)]
            rep_in1, rep_in2 = [
                transpose_irrep_layout(arr, irreps, "mul_ir", conv.config.layout)
                for arr, irreps in zip(
                    (rep_in1, rep_in2),
                    (conv.config.irreps_in1, conv.config.irreps_in2),
                )
            ]
            conv.forward_cpu(
                L1_in=rep_in1,
                L2_in=rep_in2,
                weights=rep_weights,
                L3_out=repeated_run,
                graph=graph,
            )

            repeated_run = transpose_irrep_layout(
                repeated_run, conv.config.irreps_out, conv.config.layout, "mul_ir"
            )

            result["output"]["bitwise_reproducible"] = bool(
                result["output"]["bitwise_reproducible"]
                and np.all(repeated_run == test_out)
            )

    return result


def correctness_backward_conv(
    conv,
    graph,
    thresh,
    prng_seed,
    reference_implementation=None,
    high_precision_ref=False,
):
    if reference_implementation is None:
        from openequivariance._torch.E3NNConv import E3NNConv

        reference_implementation = E3NNConv

    result = {"thresh": thresh}

    buffers = get_random_buffers_backward_conv(
        conv.config, graph.node_count, graph.nnz, prng_seed
    )
    reference_problem = conv.config

    if high_precision_ref:
        reference_problem = copy.deepcopy(conv.config)
        reference_problem.irrep_dtype = np.float64
        reference_problem.weight_dtype = np.float64
    grads = []
    for i, impl in enumerate([conv, reference_implementation]):
        is_test_impl = i == 0
        tp = impl if is_test_impl else impl(reference_problem)

        (
            run_in1,
            run_in2,
            run_out_grad,
            run_weights,
            run_weights_grad,
            run_in1_grad,
            run_in2_grad,
        ) = [buf.copy() for buf in buffers]

        if not is_test_impl and high_precision_ref:
            (
                run_in1,
                run_in2,
                run_out_grad,
                run_weights,
                run_weights_grad,
                run_in1_grad,
                run_in2_grad,
            ) = [np.array(el, dtype=np.float64) for el in buffers]

        if is_test_impl:
            run_in1, run_in2, run_out_grad = [
                transpose_irrep_layout(arr, irreps, "mul_ir", conv.config.layout)
                for arr, irreps in zip(
                    (run_in1, run_in2, run_out_grad),
                    (
                        conv.config.irreps_in1,
                        conv.config.irreps_in2,
                        conv.config.irreps_out,
                    ),
                )
            ]

        tp.backward_cpu(
            L1_in=run_in1,
            L1_grad=run_in1_grad,
            L2_in=run_in2,
            L2_grad=run_in2_grad,
            L3_grad=run_out_grad,
            weights=run_weights,
            weights_grad=run_weights_grad,
            graph=graph,
        )

        if is_test_impl:
            run_in1_grad, run_in2_grad = [
                transpose_irrep_layout(arr, irreps, conv.config.layout, "mul_ir")
                for arr, irreps in zip(
                    (run_in1_grad, run_in2_grad),
                    (conv.config.irreps_in1, conv.config.irreps_in2),
                )
            ]

        grads.append((run_weights_grad, run_in1_grad, run_in2_grad))

    for name, to_check, ground_truth, threshold in [
        ("weight_grad", grads[0][0], grads[1][0], thresh),
        ("in1_grad", grads[0][1], grads[1][1], thresh),
        ("in2_grad", grads[0][2], grads[1][2], thresh),
    ]:
        result[name] = check_similiarity(name, to_check, ground_truth, threshold)

    return result


def correctness_double_backward_conv(
    conv,
    graph,
    thresh,
    prng_seed,
    reference_implementation=None,
    high_precision_ref=False,
):
    buffers = get_random_buffers_double_backward_conv(
        conv.config, graph.node_count, graph.nnz, prng_seed
    )

    if reference_implementation is None:
        from openequivariance._torch.E3NNConv import E3NNConv

        reference_implementation = E3NNConv

    reference_problem = conv.config
    if high_precision_ref:
        reference_problem = copy.deepcopy(conv.config)
        reference_problem.irrep_dtype = np.float64
        reference_problem.weight_dtype = np.float64

    reference_tp = reference_implementation(reference_problem, torch_op=True)

    result = {"thresh": thresh}
    tensors = []
    for i, tp in enumerate([conv, reference_tp]):
        is_test_impl = i == 0
        buffers_copy = [buf.copy() for buf in buffers]

        if i == 1 and high_precision_ref:
            buffers_copy = [np.array(el, dtype=np.float64) for el in buffers]

        in1, in2, out_grad, weights, weights_dgrad, in1_dgrad, in2_dgrad, _ = (
            buffers_copy
        )

        weights_reordered = tp.reorder_weights_from_e3nn(
            weights, not tp.config.shared_weights
        )
        weights_dgrad_reordered = tp.reorder_weights_from_e3nn(
            weights_dgrad, not tp.config.shared_weights
        )

        db_in1, db_in2, db_out_grad, db_in1_dgrad, db_in2_dgrad = [
            buf.copy() for buf in (in1, in2, out_grad, in1_dgrad, in2_dgrad)
        ]
        if is_test_impl:
            db_in1, db_in2, db_out_grad, db_in1_dgrad, db_in2_dgrad = [
                transpose_irrep_layout(arr, irreps, "mul_ir", tp.config.layout)
                for arr, irreps in zip(
                    (db_in1, db_in2, db_out_grad, db_in1_dgrad, db_in2_dgrad),
                    (
                        tp.config.irreps_in1,
                        tp.config.irreps_in2,
                        tp.config.irreps_out,
                        tp.config.irreps_in1,
                        tp.config.irreps_in2,
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
            graph,
        )

        if is_test_impl:
            out_dgrad, in1_grad, in2_grad = [
                transpose_irrep_layout(arr, irreps, tp.config.layout, "mul_ir")
                for arr, irreps in zip(
                    (out_dgrad, in1_grad, in2_grad),
                    (tp.config.irreps_out, tp.config.irreps_in1, tp.config.irreps_in2),
                )
            ]

        tensors.append(
            (
                out_dgrad,
                in1_grad,
                in2_grad,
                tp.reorder_weights_to_e3nn(
                    weights_grad, has_batch_dim=not conv.config.shared_weights
                ),
            )
        )

    for name, to_check, ground_truth in [
        ("output_grad", tensors[0][0], tensors[1][0]),
        ("in1_grad", tensors[0][1], tensors[1][1]),
        ("in2_grad", tensors[0][2], tensors[1][2]),
        ("weights_grad", tensors[0][3], tensors[1][3]),
    ]:
        result[name] = check_similiarity(name, to_check, ground_truth, thresh)

    return result
