from typing import Optional
import types

import numpy as np
import torch

from openequivariance import extlib
from openequivariance.implementations.convolution.ConvolutionBase import ConvolutionBase
from openequivariance.implementations.convolution.LoopUnrollConv import LoopUnrollConv
from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance import TPProblem


class TensorProductConv(torch.nn.Module, LoopUnrollConv):
    """
    Given a **symmetric, directed** graph :math:`G = (V, E)`, inputs :math:`x_1...x_{|V|}`,
    :math:`y_1...y_{|E|}`, and weights :math:`W_1...W_{|E|}`, computes

    .. math::

        z_i = \sum_{(i, j, e) \in \mathcal{N}(i)} W_e (x_j \otimes_{\\textrm{CG}} y_e)

    where :math:`(i, j, e) \in \mathcal{N}(i)` indicates that node :math:`i` is connected to node :math:`j`
    via the edge indexed :math:`e`.

    This class offers multiple options to perform the summation: an atomic algorithm and a deterministic algorithm
    that relies on a sorted adjacency matrix input. If you use the determinstic algorithm, you must also supply
    a permutation to transpose the adjacency matrix.

    :param problem: Specification of the tensor product.
    :param deterministic: if ``False``, uses atomics for the convolution. If ``True``, uses a deterministic
           fixup-based algorithm. `Default`: ``False``.
    :param kahan: if ``True``, uses Kahan summation to improve accuracy during aggregation. To use this option,
           the input tensors must be in float32 precision AND you must set ``deterministic=True``. *Default*: ``False``.

    """

    def __init__(
        self,
        problem: TPProblem,
        deterministic: bool = False,
        kahan: bool = False,
        torch_op=True,
    ):
        torch.nn.Module.__init__(self)
        LoopUnrollConv.__init__(
            self,
            problem,
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
        X: torch.Tensor,
        Y: torch.Tensor,
        W: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        sender_perm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the fused CG tensor product + convolution.

        :param X: Tensor of shape ``[|V|, problem.irreps_in1.dim()]``, datatype ``problem.irrep_dtype``.
        :param Y: Tensor of shape ``[|E|, problem.irreps_in1.dim()]``, datatype ``problem.irrep_dtype``.
        :param W: Tensor of datatype ``problem.weight_dtype`` and shape

            * ``[|E|, problem.weight_numel]`` if ``problem.shared_weights=False``
            * ``[problem.weight_numel]`` if ``problem.shared_weights=True``

        :param rows: Tensor of shape ``[|E|]`` with row indices for each nonzero in the adjacency matrix,
                datatype ``torch.int64``. Must be row-major sorted along with ``cols`` when ``deterministic=True``.
        :param cols: Tensor of shape ``[|E|]`` with column indices for each nonzero in the adjacency matrix,
                datatype ``torch.int64``.
        :param sender_perm: Tensor of shape ``[|E|]`` and ``torch.int64`` datatype containing a
                permutation that transposes the adjacency matrix nonzeros from row-major to column-major order.
                Must be provided when ``deterministic=True``.

        :return: Tensor of shape ``[|V|, problem.irreps_out.dim()]``, datatype ``problem.irrep_dtype``.
        """
        if sender_perm is None:
            return torch.ops.libtorch_tp_jit.jit_conv_forward(
                self.internal,
                X,
                Y,
                W,
                rows,
                cols,
                self.workspace_buffer,
                self.dummy_transpose_perm,
            )
        else:
            return torch.ops.libtorch_tp_jit.jit_conv_forward(
                self.internal,
                X,
                Y,
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
