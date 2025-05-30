from typing import Optional
import types

import numpy as np
import torch

from openequivariance import extlib
from openequivariance.implementations.convolution.ConvolutionBase import ConvolutionBase
from openequivariance.implementations.convolution.LoopUnrollConv import LoopUnrollConv
from openequivariance.implementations.TensorProduct import TensorProduct


class TensorProductConv(torch.nn.Module, LoopUnrollConv):
    """
    PyTorch-specialized dispatcher class.
    """

    def __init__(
        self,
        config,
        idx_dtype=np.int64,
        torch_op=True,
        deterministic=False,
        kahan=False,
    ):
        torch.nn.Module.__init__(self)
        LoopUnrollConv.__init__(
            self,
            config,
            idx_dtype=np.int64,
            torch_op=torch_op,
            deterministic=deterministic,
            kahan=kahan,
        )

        self.dummy_transpose_perm = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.weight_numel = self.config.weight_numel

        if not extlib.TORCH_COMPILE:
            self.forward = types.MethodType(LoopUnrollConv.forward, self)

    def forward(
        self,
        L1_in: torch.Tensor,
        L2_in: torch.Tensor,
        W: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        sender_perm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if sender_perm is None:
            return torch.ops.torch_tp_jit.jit_conv_forward(
                self.internal,
                L1_in,
                L2_in,
                W,
                rows,
                cols,
                self.workspace_buffer,
                self.dummy_transpose_perm,
            )
        else:
            return torch.ops.torch_tp_jit.jit_conv_forward(
                self.internal,
                L1_in,
                L2_in,
                W,
                rows,
                cols,
                self.workspace_buffer,
                sender_perm,
            )

    @staticmethod
    def name():
        return LoopUnrollConv.name()


# ==================================================================
# Reference implementations for benchmarking


class TensorProductConvKahan(TensorProductConv):
    def __init__(self, config, idx_dtype=np.int64, torch_op=True):
        super().__init__(config, idx_dtype, torch_op, deterministic=True, kahan=True)

    @staticmethod
    def name():
        return "LoopUnrollConvKahan"


class TensorProductConvDeterministic(TensorProductConv):
    def __init__(self, config, idx_dtype=np.int64, torch_op=True):
        super().__init__(config, idx_dtype, torch_op, deterministic=True)

    @staticmethod
    def name():
        return "LoopUnrollConvDeterministic"


class TensorProductConvAtomic(TensorProductConv):
    def __init__(self, config, idx_dtype=np.int64, torch_op=True):
        super().__init__(config, idx_dtype, torch_op, deterministic=False)

    @staticmethod
    def name():
        return "LoopUnrollConvAtomic"


class TensorProductConvScatterSum(ConvolutionBase):
    def __init__(self, config, idx_dtype=np.int64, torch_op=True):
        assert torch_op
        global torch
        import torch

        super().__init__(config, idx_dtype, torch_op=torch_op, deterministic=False)

        self.reference_tp = TensorProduct(config, torch_op=torch_op)
        from openequivariance.implementations.convolution.scatter import scatter_sum

        self.scatter_sum = scatter_sum

    def forward(self, L1_in, L2_in, weights, rows, cols):
        tp_outputs = self.reference_tp(L1_in[cols], L2_in, weights)
        return self.scatter_sum(
            src=tp_outputs, index=rows, dim=0, dim_size=L1_in.shape[0]
        )

    def forward_cpu(self, L1_in, L2_in, weights, L3_out, graph):
        tp_outputs = np.zeros((graph.nnz, self.L3.dim), dtype=L3_out.dtype)
        self.reference_tp.forward_cpu(L1_in[graph.cols], L2_in, tp_outputs, weights)
        np.add.at(L3_out, graph.rows, tp_outputs)

    def backward_cpu(
        self,
        L1_in: np.ndarray,
        L1_grad: np.ndarray,
        L2_in: np.ndarray,
        L2_grad: np.ndarray,
        L3_grad: np.ndarray,
        weights: np.ndarray,
        weights_grad: np.ndarray,
        graph,
    ):
        L1_grad_bcast = np.zeros((graph.nnz, self.L1.dim), dtype=L1_grad.dtype)
        self.reference_tp.backward_cpu(
            L1_in[graph.cols],
            L1_grad_bcast,
            L2_in,
            L2_grad,
            L3_grad[graph.rows],
            weights,
            weights_grad,
        )
        np.add.at(L1_grad, graph.cols, L1_grad_bcast)

    @staticmethod
    def name():
        return "LoopUnrollConvScatterSum"
