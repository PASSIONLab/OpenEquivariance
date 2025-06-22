from typing import Optional, List

import numpy as np
import torch

from openequivariance import extlib
from openequivariance.implementations.convolution.ConvolutionBase import ConvolutionBase
from openequivariance.implementations.convolution.LoopUnrollConv import LoopUnrollConv
from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance import TPProblem
from openequivariance.implementations.utils import torch_to_oeq_dtype


class TensorProductConv(torch.nn.Module, LoopUnrollConv):
    r"""
    Given a **symmetric, directed** graph :math:`G = (V, E)`, inputs :math:`x_1...x_{|V|}`,
    :math:`y_1...y_{|E|}`, and weights :math:`W_1...W_{|E|}`, computes

    .. math::

        z_i = \sum_{(i, j, e) \in \mathcal{N}(i)} W_e (x_j \otimes_{\textrm{CG}} y_e)

    where :math:`(i, j, e) \in \mathcal{N}(i)` indicates that node :math:`i` is connected to node :math:`j`
    via the edge indexed :math:`e`.

    This class offers multiple options to perform the summation: an atomic algorithm and a deterministic algorithm
    that relies on a sorted adjacency matrix input. If you use the determinstic algorithm, you must also supply
    a permutation to transpose the adjacency matrix.

    :param problem: Specification of the tensor product.
    :param deterministic: if ``False``, uses atomics for the convolution. If ``True``, uses a deterministic
           fixup-based algorithm. `Default`: ``False``.
    :param kahan: If ``True``, uses Kahan summation to improve accuracy during aggregation. To use this option,
           the input tensors must be in float32 precision AND you must set ``deterministic=True``. *Default*: ``False``.
    :param use_opaque: If ``True, uses an opaque forward pass that cannot be symbolically traced. *Default*: ``False``.
    """

    def __init__(
        self,
        problem: TPProblem,
        deterministic: bool = False,
        kahan: bool = False,
        torch_op=True,
        use_opaque: bool = False,
    ):
        torch.nn.Module.__init__(self)
        self.input_args = {
            "problem": problem,
            "deterministic": deterministic,
            "kahan": kahan,
            "torch_op": torch_op,
            "use_opaque": use_opaque,
        }
        self._init_class()

    def _init_class(self):
        LoopUnrollConv.__init__(
            self,
            self.input_args["problem"],
            idx_dtype=np.int64,
            torch_op=self.input_args["torch_op"],
            deterministic=self.input_args["deterministic"],
            kahan=self.input_args["kahan"],
        )
        self.dummy_transpose_perm = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.weight_numel = self.config.weight_numel
        self._setup_notorchbind()

        if (not extlib.TORCH_COMPILE) or self.input_args["use_opaque"]:
            self.forward = self.forward_opaque

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        if dtype is not None:
            updated_problem = self.input_args["problem"].clone()
            updated_problem.irrep_dtype = torch_to_oeq_dtype(dtype)
            updated_problem.weight_dtype = torch_to_oeq_dtype(dtype)
            self.input_args["problem"] = updated_problem
            self._init_class()

        torch.nn.Module.to(self, *args, **kwargs)
        return self

    def __getstate__(self):
        return self.input_args

    def __setstate__(self, state):
        torch.nn.Module.__init__(self)
        self.input_args = state
        self._init_class()

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        W: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        sender_perm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
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

    def _setup_notorchbind(self):
        @torch.library.custom_op(
            f"openequivariance::conv_forward{self.conv_id}",
            mutates_args=(),
            device_types="cuda",
        )
        def forward(
            L1_in: torch.Tensor,
            L2_in: torch.Tensor,
            weights: torch.Tensor,
            rows: torch.Tensor,
            cols: torch.Tensor,
            transpose_perm: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            L1_in_c, L2_in_c, weights_c = (
                L1_in.contiguous(),
                L2_in.contiguous(),
                weights.contiguous(),
            )
            L3_out = torch.zeros(
                (L1_in_c.shape[0], self.L3.dim), dtype=L1_in.dtype, device="cuda"
            )

            self.internal.exec_conv_rawptrs(
                L1_in_c.data_ptr(),
                L2_in_c.data_ptr(),
                weights_c.data_ptr(),
                L3_out.data_ptr(),
                rows.contiguous().data_ptr(),
                cols.contiguous().data_ptr(),
                rows.shape[0],
                L1_in.shape[0],
                self.workspace_ptr,
            )

            return L3_out

        @forward.register_fake
        def _(L1_in, L2_in, weights, rows, cols, transpose_perm=None):
            return L1_in.new_empty(L1_in.shape[0], self.L3.dim)

        self.forward_opaque = forward

        @torch.library.custom_op(
            f"openequivariance::conv_backward{self.conv_id}",
            mutates_args=(),
            device_types="cuda",
        )
        def backward_helper(
            L1_in: torch.Tensor,
            L2_in: torch.Tensor,
            weights: torch.Tensor,
            L3_grad: torch.Tensor,
            rows: torch.Tensor,
            cols: torch.Tensor,
            transpose_perm: Optional[torch.Tensor] = None,
        ) -> List[torch.Tensor]:
            L1_grad = torch.zeros_like(L1_in)
            L2_grad = torch.zeros_like(L2_in)
            weights_grad = torch.empty_like(weights)

            if self.config.shared_weights:
                weights_grad[:] = 0.0

            transpose_perm_ptr = 0
            if transpose_perm is not None:
                transpose_perm_ptr = transpose_perm.data_ptr()

            self.internal.backward_rawptrs(
                L1_in.contiguous().data_ptr(),
                L1_grad.data_ptr(),
                L2_in.contiguous().data_ptr(),
                L2_grad.data_ptr(),
                weights.contiguous().data_ptr(),
                weights_grad.data_ptr(),
                L3_grad.contiguous().data_ptr(),
                rows.contiguous().data_ptr(),
                cols.contiguous().data_ptr(),
                rows.shape[0],
                L1_in.shape[0],
                self.workspace_ptr,
                transpose_perm_ptr,
            )

            return [L1_grad, L2_grad, weights_grad]

        @backward_helper.register_fake
        def _(L1_in, L2_in, weights, L3_grad, rows, cols, transpose_perm=None):
            return [
                L1_in.new_empty(*L1_in.shape),
                L2_in.new_empty(*L2_in.shape),
                weights.new_empty(*weights.shape),
            ]

        def setup_context(ctx, inputs, output):
            (
                ctx.L1_in,
                ctx.L2_in,
                ctx.weights,
                ctx.rows,
                ctx.cols,
                ctx.transpose_perm,
            ) = inputs

        def backward(ctx, grad_output):
            result = backward_helper(
                ctx.L1_in,
                ctx.L2_in,
                ctx.weights,
                grad_output,
                ctx.rows,
                ctx.cols,
                ctx.transpose_perm,
            )
            return result[0], result[1], result[2], None, None, None

        self.forward_opaque.register_autograd(backward, setup_context=setup_context)

        def setup_context_double_backward(ctx, inputs, output):
            (
                ctx.L1_in,
                ctx.L2_in,
                ctx.weights,
                ctx.L3_grad,
                ctx.rows,
                ctx.cols,
                ctx.transpose_perm,
            ) = inputs

        @torch.library.custom_op(
            f"openequivariance::conv_double_backward{self.conv_id}",
            mutates_args=(),
            device_types="cuda",
        )
        def double_backward_helper(
            L1_in: torch.Tensor,
            L2_in: torch.Tensor,
            W: torch.Tensor,
            L3_grad: torch.Tensor,
            L1_dgrad: torch.Tensor,
            L2_dgrad: torch.Tensor,
            w_dgrad: torch.Tensor,
            rows: torch.Tensor,
            cols: torch.Tensor,
            transpose_perm: Optional[torch.Tensor] = None,
        ) -> List[torch.Tensor]:
            L1_grad = torch.zeros_like(L1_in)
            L2_grad = torch.zeros_like(L2_in)
            W_grad = torch.empty_like(W)
            L3_dgrad = torch.zeros_like(L3_grad)

            if self.config.shared_weights:
                W_grad[:] = 0.0

            transpose_perm_ptr = 0
            if transpose_perm is not None:
                transpose_perm_ptr = transpose_perm.data_ptr()

            self.internal.double_backward_rawptrs(
                L1_in.contiguous().data_ptr(),
                L2_in.contiguous().data_ptr(),
                W.contiguous().data_ptr(),
                L3_grad.contiguous().data_ptr(),
                L1_dgrad.contiguous().data_ptr(),
                L2_dgrad.contiguous().data_ptr(),
                w_dgrad.contiguous().data_ptr(),
                L1_grad.data_ptr(),
                L2_grad.data_ptr(),
                W_grad.data_ptr(),
                L3_dgrad.data_ptr(),
                rows.contiguous().data_ptr(),
                cols.contiguous().data_ptr(),
                rows.shape[0],
                L1_in.shape[0],
                self.workspace_ptr,
                transpose_perm_ptr,
            )
            return [L1_grad, L2_grad, W_grad, L3_dgrad]

        @double_backward_helper.register_fake
        def _(
            L1_in,
            L2_in,
            W,
            L3_grad,
            L1_dgrad,
            L2_dgrad,
            w_dgrad,
            rows,
            cols,
            transpose_perm=None,
        ):
            return [
                L1_in.new_empty(*L1_in.shape),
                L2_in.new_empty(*L2_in.shape),
                W.new_empty(*W.shape),
                L3_grad.new_empty(*L3_grad.shape),
            ]

        def double_backward(ctx, grad_output):
            L1_dgrad, L2_dgrad, w_dgrad = grad_output[0], grad_output[1], grad_output[2]

            L1_grad, L2_grad, W_grad, L3_dgrad = double_backward_helper(
                ctx.L1_in,
                ctx.L2_in,
                ctx.weights,
                ctx.L3_grad,
                L1_dgrad,
                L2_dgrad,
                w_dgrad,
                ctx.rows,
                ctx.cols,
                ctx.transpose_perm,
            )

            if ctx.transpose_perm is None:
                return L1_grad, L2_grad, W_grad, L3_dgrad, None, None
            else:
                return L1_grad, L2_grad, W_grad, L3_dgrad, None, None, None

        backward_helper.register_autograd(
            double_backward, setup_context=setup_context_double_backward
        )


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
