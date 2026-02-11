import copy
import numpy as np

from openequivariance.core.random_buffer_utils import (
    get_random_buffers_forward_conv,
    get_random_buffers_backward_conv,
    get_random_buffers_double_backward_conv,
)
from .correctness_utils import check_similiarity


def _require_reference_impl(reference_implementation):
    if reference_implementation is None:
        raise ValueError(
            "reference_implementation is required. "
            "Pass tests.benchmarks._torch.E3NNConv.E3NNConv or another conv reference."
        )


def conv_correctness_forward(
    conv,
    graph,
    thresh,
    prng_seed,
    reference_implementation=None,
    check_reproducible=True,
    high_precision_ref=False,
):
    _require_reference_impl(reference_implementation)

    result = {"thresh": thresh}

    in1, in2, weights, out = get_random_buffers_forward_conv(
        conv.config, graph.node_count, graph.nnz, prng_seed
    )
    ref_in1, ref_in2, ref_weights, ref_out = [
        buf.copy() for buf in [in1, in2, weights, out]
    ]

    reference_config = conv.config
    if high_precision_ref:
        reference_config = copy.deepcopy(conv.config)
        reference_config.irrep_dtype = np.float64
        reference_config.weight_dtype = np.float64
        ref_in1, ref_in2, ref_weights, ref_out = [
            np.array(el, dtype=np.float64) for el in [ref_in1, ref_in2, ref_weights, ref_out]
        ]

    args = {
        "L1_in": ref_in1,
        "L2_in": ref_in2,
        "weights": ref_weights,
        "rows": graph.rows,
        "cols": graph.cols,
    }

    ref_tp = reference_implementation(reference_config)
    if ref_tp.deterministic:
        args["transpose_perm"] = graph.transpose_perm

    import torch

    for key in args:
        args[key] = torch.tensor(args[key], device="cuda")

    ref_out[:] = ref_tp.forward(**args).cpu().numpy()

    test_out = out.copy()
    conv.forward_cpu(
        L1_in=in1.copy(),
        L2_in=in2.copy(),
        weights=weights.copy(),
        L3_out=test_out,
        graph=graph,
    )

    for name, to_check, ground_truth in [("output", ref_out, test_out)]:
        result[name] = check_similiarity(name, to_check, ground_truth, thresh)

    if check_reproducible:
        num_trials = 5
        for name in ["output"]:
            result[name]["num_reproducibility_trials"] = num_trials
            result[name]["bitwise_reproducible"] = True

        for _ in range(num_trials):
            repeated_run = out.copy()
            conv.forward_cpu(
                L1_in=in1.copy(),
                L2_in=in2.copy(),
                weights=weights.copy(),
                L3_out=repeated_run,
                graph=graph,
            )

            for name, to_check, ground_truth in [("output", repeated_run, test_out)]:
                result[name]["bitwise_reproducible"] = bool(
                    result[name]["bitwise_reproducible"]
                    and np.all(repeated_run == test_out)
                )

    return result


def conv_correctness_backward(
    conv,
    graph,
    thresh,
    prng_seed,
    reference_implementation=None,
    high_precision_ref=False,
):
    _require_reference_impl(reference_implementation)

    result = {"thresh": thresh}

    buffers = get_random_buffers_backward_conv(
        conv.config, graph.node_count, graph.nnz, prng_seed
    )
    reference_buffers = [buf.copy() for buf in buffers]
    reference_problem = conv.config

    if high_precision_ref:
        reference_problem = copy.deepcopy(conv.config)
        reference_problem.irrep_dtype = np.float64
        reference_problem.weight_dtype = np.float64
        reference_buffers = [np.array(el, dtype=np.float64) for el in reference_buffers]

    ref_tp = reference_implementation(reference_problem)
    in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = buffers
    (
        ref_in1,
        ref_in2,
        ref_out_grad,
        ref_weights,
        ref_weights_grad,
        ref_in1_grad,
        ref_in2_grad,
    ) = reference_buffers

    ref_tp.backward_cpu(
        L1_in=ref_in1,
        L1_grad=ref_in1_grad,
        L2_in=ref_in2,
        L2_grad=ref_in2_grad,
        L3_grad=ref_out_grad,
        weights=ref_weights,
        weights_grad=ref_weights_grad,
        graph=graph,
    )

    test_weights_grad = weights_grad.copy()
    test_in1_grad = in1_grad.copy()
    test_in2_grad = in2_grad.copy()

    conv.backward_cpu(
        L1_in=in1.copy(),
        L1_grad=test_in1_grad,
        L2_in=in2.copy(),
        L2_grad=test_in2_grad,
        L3_grad=out_grad.copy(),
        weights=weights.copy(),
        weights_grad=test_weights_grad,
        graph=graph,
    )

    for name, to_check, ground_truth, threshold in [
        ("weight_grad", test_weights_grad, ref_weights_grad, thresh),
        ("in1_grad", test_in1_grad, ref_in1_grad, thresh),
        ("in2_grad", test_in2_grad, ref_in2_grad, thresh),
    ]:
        result[name] = check_similiarity(name, to_check, ground_truth, threshold)

    return result


def conv_correctness_double_backward(
    conv,
    graph,
    thresh,
    prng_seed,
    reference_implementation=None,
    high_precision_ref=False,
):
    _require_reference_impl(reference_implementation)

    buffers = get_random_buffers_double_backward_conv(
        conv.config, graph.node_count, graph.nnz, prng_seed
    )

    reference_problem = conv.config
    if high_precision_ref:
        reference_problem = copy.deepcopy(conv.config)
        reference_problem.irrep_dtype = np.float64
        reference_problem.weight_dtype = np.float64

    reference_tp = reference_implementation(reference_problem, torch_op=True)

    result = {"thresh": thresh}
    tensors = []
    for i, tp in enumerate([conv, reference_tp]):
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

        in1_grad, in2_grad, weights_grad, out_dgrad = tp.double_backward_cpu(
            in1,
            in2,
            out_grad,
            weights_reordered,
            weights_dgrad_reordered,
            in1_dgrad,
            in2_dgrad,
            graph,
        )

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
