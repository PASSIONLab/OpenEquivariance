__all__ = [
    "FlashTPConv",
]

import torch
import numpy as np
from openequivariance.core.ConvolutionBase import ConvolutionBase
from openequivariance.core.utils import oeq_to_torch_dtype


class FlashTPConv(ConvolutionBase):
    def __init__(self, config, *, idx_dtype=np.int64, torch_op=True):
        super().__init__(config, idx_dtype=idx_dtype, torch_op=torch_op)
        from flashTP_e3nn import uvu_TP

        instructions = [
            (
                inst.i_in1,
                inst.i_in2,
                inst.i_out,
                inst.connection_mode,
                inst.has_weight,
                inst.path_weight,
            )
            for inst in config.instructions
        ]

        self.internal = uvu_TP(
            config.irreps_in1,
            config.irreps_in2,
            config.irreps_out,
            instructions,
            device="cuda",
            dtype=oeq_to_torch_dtype(config.irrep_dtype),
        )

    def forward(self, L1_in, L2_in, weights, rows, cols, transpose_perm=None):
        return self.internal(
            L1_in, L2_in, weights, rows.to(torch.int), cols.to(torch.int)
        )

    @staticmethod
    def name():
        return "FlashTPConv"
