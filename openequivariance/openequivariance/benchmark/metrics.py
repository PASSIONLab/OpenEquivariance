from openequivariance.core.utils import (
    count_cg_non_zero,
)

from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.logging import getLogger
import numpy as np

logger = getLogger()


def memory_streamed_forward(tpp: TPProblem, batch_size: int) -> dict[str, int]:
    """
    This represents an absolute minimum amount of memory that could be streamed on an ideal machine
    It returns the number of bytes streamed total and from each source
    """
    data_size = {}
    irrep_word_size = np.dtype(tpp.irrep_dtype).itemsize
    weight_word_size = np.dtype(tpp.weight_dtype).itemsize

    data_size["input 1"] = tpp.irreps_in1.dim * batch_size * irrep_word_size
    data_size["input 2"] = tpp.irreps_in2.dim * batch_size * irrep_word_size
    data_size["output"] = tpp.irreps_out.dim * batch_size * irrep_word_size
    data_size["weights"] = tpp.weight_numel * batch_size * weight_word_size
    data_size["total"] = sum(data_size.values())
    return data_size


def memory_streamed_backward(tpp: TPProblem, batch_size: int) -> dict:
    """
    This represents an absolute minimum amount of memory that could be streamed on an ideal machine
    It returns the number of bytes streamed total and from each source
    """
    data_size = {}
    irrep_word_size = np.dtype(tpp.irrep_dtype).itemsize
    weight_word_size = np.dtype(tpp.weight_dtype).itemsize

    data_size["input 1"] = tpp.irreps_in1.dim * batch_size * irrep_word_size
    data_size["input 1 grad"] = tpp.irreps_in1.dim * batch_size * irrep_word_size
    data_size["input 2"] = tpp.irreps_in2.dim * batch_size * irrep_word_size
    data_size["input 2 grad"] = tpp.irreps_in2.dim * batch_size * irrep_word_size
    data_size["output grad"] = tpp.irreps_out.dim * batch_size * irrep_word_size
    data_size["weights"] = tpp.weight_numel * batch_size * weight_word_size
    data_size["weights grad"] = tpp.weight_numel * batch_size * weight_word_size
    data_size["total"] = sum(data_size.values())
    return data_size


def flops_forward(tpp: TPProblem, batch_size: int) -> dict:
    """
    Default FLOP estimate aligned with LoopUnrollTP's forward FLOP accounting.
    """
    flops_count = {"CG_decomposition": 0, "linear_combination": 0, "outer_products": 0}

    for ins in tpp.instructions:  # type : Instruction
        l1, l2, l3 = (
            tpp.irreps_in1[ins.i_in1].ir.l,
            tpp.irreps_in2[ins.i_in2].ir.l,
            tpp.irreps_out[ins.i_out].ir.l,
        )

        flops_count["CG_decomposition"] += count_cg_non_zero(l1, l2, l3) * (
            ins.path_shape[0] * ins.path_shape[1]
        )
        flops_count["linear_combination"] += (
            (2 * l3 + 1) * np.prod(ins.path_shape) if ins.has_weight else 0
        )

    flops_count["CG_decomposition"] *= 3 * batch_size
    flops_count["linear_combination"] *= batch_size  # Weights do not require FMA here

    flops_count["total"] = sum(flops_count.values())
    return flops_count


def flops_backward(tpp: TPProblem, batch_size: int) -> dict:
    """
    Default FLOP estimate aligned with LoopUnrollTP's backward FLOP accounting.
    """
    flops_count = {"backward": 0}

    for ins in tpp.instructions:  # type : Instruction
        l1, l2, l3 = (
            tpp.irreps_in1[ins.i_in1].ir.l,
            tpp.irreps_in2[ins.i_in2].ir.l,
            tpp.irreps_out[ins.i_out].ir.l,
        )
        flops_count["backward"] += count_cg_non_zero(l1, l2, l3) * (
            ins.path_shape[0] * ins.path_shape[1]
        )

    flops_count["backward"] *= 9 * batch_size
    flops_count["total"] = sum(flops_count.values())
    return flops_count
