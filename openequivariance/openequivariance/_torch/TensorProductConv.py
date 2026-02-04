from typing import Optional

import numpy as np
import torch

import openequivariance._torch.extlib as extlib
from openequivariance._torch.extlib import (
    postprocess_kernel,
    DeviceProp,
)

from openequivariance.core.ConvolutionBase import (
    ConvolutionBase,
    scatter_add_wrapper,
)
from openequivariance.core.LoopUnrollConv import LoopUnrollConv
from openequivariance._torch.TensorProduct import TensorProduct
from openequivariance import TPProblem
from openequivariance.core.utils import torch_to_oeq_dtype
from openequivariance._torch.utils import enum_to_torch_dtype, reorder_torch, string_to_tensor

from openequivariance.benchmark.logging_utils import getLogger
from openequivariance._torch.NPDoubleBackwardMixin import NumpyDoubleBackwardMixinConv

logger = getLogger()


class TensorProductConv(torch.nn.Module, LoopUnrollConv, NumpyDoubleBackwardMixinConv):
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
    :param use_opaque: This parameter is deprecated. 
    """

    def __init__(
        self,
        problem: TPProblem,
        *,
        deterministic: bool = False,
        kahan: bool = False,
        torch_op: bool = True,
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
        dp = DeviceProp(0)
        LoopUnrollConv.__init__(
            self,
            self.input_args["problem"],
            dp,
            postprocess_kernel,
            idx_dtype=np.int64,
            torch_op=self.input_args["torch_op"],
            deterministic=self.input_args["deterministic"],
            kahan=self.input_args["kahan"],
        )

        self.allocate_workspace(self.workspace_size)
        self.dummy_transpose_perm = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.weight_numel = self.config.weight_numel
        self.kernel= string_to_tensor(self.kernel_string)


    def to(self, *args, **kwargs):
        r"""
        See `torch.nn.Module.to() <https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to>`_.
        """
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
            sender_perm = self.dummy_transpose_perm

        return torch.ops.libtorch_tp_jit.jit_conv_forward(
            self.kernel,
            self.hash,
            X,
            Y,
            W,
            rows,
            cols,
            self.workspace_buffer,
            sender_perm,
        )

    def allocate_workspace(self, size_bytes):
        self.workspace_size = size_bytes
        self.workspace_buffer = torch.zeros(
            size_bytes, dtype=torch.uint8, device="cuda"
        )
        self.workspace_ptr = self.workspace_buffer.data_ptr()
        logger.info(f"Convolution requires {size_bytes // 1000000}MB of workspace.")

    def reorder_weights_from_e3nn(self, weights, has_batch_dim=True):
        return reorder_torch(
            self.forward_schedule, weights, "forward", not self.config.shared_weights
        )

    def reorder_weights_to_e3nn(self, weights, has_batch_dim=True):
        return reorder_torch(
            self.forward_schedule, weights, "backward", not self.config.shared_weights
        )

    @staticmethod
    def name():
        return "LoopUnrollConv"

    @classmethod
    def register_torch_fakes(cls):
        global torch
        import torch

        @torch.library.register_fake("libtorch_tp_jit::jit_conv_forward")
        def fake_forward(
            kernel, hash, L1_in, L2_in, W, rows, cols, workspace_buffer, sender_perm
        ):
            L3_dim, irrep_dtype = None, None
            if hasattr(jit, "wrapped_obj"):
                L3_dim = jit.wrapped_obj.kernel_dims["L3_dim"]
                irrep_dtype = jit.wrapped_obj.kernel_dims["irrep_dtype"]
            else:
                L3_dim = jit.L3_dim
                irrep_dtype = jit.irrep_dtype

            return torch.empty(
                L1_in.shape[0],
                L3_dim,
                device="cuda",
                dtype=enum_to_torch_dtype[irrep_dtype],
            )

        @torch.library.register_fake("libtorch_tp_jit::jit_conv_backward")
        def fake_backward(
            kernel, hash, L1_in, L2_in, W, L3_grad, rows, cols, workspace_buffer, sender_perm
        ):
            return torch.empty_like(L1_in), torch.empty_like(L2_in), torch.empty_like(W)

        @torch.library.register_fake("libtorch_tp_jit::jit_conv_double_backward")
        def fake_double_backward(
            kernel, hash,
            L1_in,
            L2_in,
            W,
            L3_grad,
            L1_dgrad,
            L2_dgrad,
            w_dgrad,
            rows,
            cols,
            workspace_buffer,
            transpose_perm=None,
        ):
            return [
                L1_in.new_empty(*L1_in.shape),
                L2_in.new_empty(*L2_in.shape),
                W.new_empty(*W.shape),
                L3_grad.new_empty(*L3_grad.shape),
            ]

    @classmethod
    def register_autograd(cls):
        backward_op = torch.ops.libtorch_tp_jit.jit_conv_backward
        double_backward_op = torch.ops.libtorch_tp_jit.jit_conv_double_backward

        def setup_context(ctx, inputs, output):
            (
                ctx.kernel,
                ctx.hash,
                ctx.L1_in,
                ctx.L2_in,
                ctx.W,
                ctx.rows,
                ctx.cols,
                ctx.workspace_buffer,
                ctx.sender_perm,
            ) = inputs

        def backward(ctx, grad_output):
            L1_grad, L2_grad, W_grad = backward_op(
                ctx.kernel,
                ctx.hash,
                ctx.L1_in,
                ctx.L2_in,
                ctx.W,
                grad_output,
                ctx.rows,
                ctx.cols,
                ctx.workspace_buffer,
                ctx.sender_perm,
            )
            return None, None, L1_grad, L2_grad, W_grad, None, None, None, None

        torch.library.register_autograd(
            "libtorch_tp_jit::jit_conv_forward", backward, setup_context=setup_context
        )

        def setup_context_double_backward(ctx, inputs, output):
            (
                ctx.kernel,
                ctx.hash,
                ctx.L1_in,
                ctx.L2_in,
                ctx.W,
                ctx.grad_output,
                ctx.rows,
                ctx.cols,
                ctx.workspace_buffer,
                ctx.sender_perm,
            ) = inputs
            ctx.inputs = inputs

        def double_backward(ctx, E, F, G):
            result = double_backward_op(
                ctx.kernel,
                ctx.hash,
                ctx.L1_in,
                ctx.L2_in,
                ctx.W,
                ctx.grad_output,
                E,
                F,
                G,
                ctx.rows,
                ctx.cols,
                ctx.workspace_buffer,
                ctx.sender_perm,
            )
            return (
                None,
                None,
                result[0],
                result[1],
                result[2],
                result[3],
                None,
                None,
                None,
                None,
            )

        torch.library.register_autograd(
            "libtorch_tp_jit::jit_conv_backward",
            double_backward,
            setup_context=setup_context_double_backward,
        )

    @classmethod
    def register_autocast(cls):
        torch.library.register_autocast(
            "libtorch_tp_jit::jit_conv_forward", "cuda", torch.float32
        )
        torch.library.register_autocast(
            "libtorch_tp_jit::jit_conv_backward", "cuda", torch.float32
        )
        torch.library.register_autocast(
            "libtorch_tp_jit::jit_conv_double_backward", "cuda", torch.float32
        )

    def forward_cpu(self, L1_in, L2_in, weights, L3_out, graph):
        assert graph.rows.dtype == self.idx_dtype
        assert graph.cols.dtype == self.idx_dtype

        weights_chunked = self.reorder_weights_from_e3nn(
            weights, not self.config.shared_weights
        )

        torch_L1_in = torch.tensor(L1_in, device="cuda")
        torch_L2_in = torch.tensor(L2_in, device="cuda")
        torch_weights = torch.tensor(weights_chunked, device="cuda")
        torch_rows = torch.tensor(graph.rows, device="cuda")
        torch_cols = torch.tensor(graph.cols, device="cuda")

        if self.deterministic:
            torch_sender_perm = torch.tensor(graph.transpose_perm, device="cuda")
        else:
            torch_sender_perm = None

        result = self.forward(
            torch_L1_in,
            torch_L2_in,
            torch_weights,
            torch_rows,
            torch_cols,
            torch_sender_perm,
        )
        L3_out[:] = result.numpy(force=True)

    def backward_cpu(
        self, L1_in, L1_grad, L2_in, L2_grad, weights, weights_grad, L3_grad, graph
    ):
        assert graph.rows.dtype == self.idx_dtype
        assert graph.cols.dtype == self.idx_dtype

        weights_chunked = self.reorder_weights_from_e3nn(
            weights, not self.config.shared_weights
        )

        torch_L1_in = torch.tensor(L1_in, requires_grad=True, device="cuda")
        torch_L2_in = torch.tensor(L2_in, requires_grad=True, device="cuda")
        torch_weights = torch.tensor(
            weights_chunked, requires_grad=True, device="cuda"
        )
        torch_L3_grad = torch.tensor(L3_grad, device="cuda")
        torch_rows = torch.tensor(graph.rows, device="cuda")
        torch_cols = torch.tensor(graph.cols, device="cuda")

        if self.deterministic:
            torch_sender_perm = torch.tensor(graph.transpose_perm, device="cuda")
        else:
            torch_sender_perm = None
        
        torch_out = self.forward(
            torch_L1_in,
            torch_L2_in,
            torch_weights,
            torch_rows,
            torch_cols,
            torch_sender_perm,
        )
        torch_out.backward(gradient=torch_L3_grad)
        L1_grad[:] = torch_L1_in.grad.numpy(force=True)
        L2_grad[:] = torch_L2_in.grad.numpy(force=True)
        weights_grad[:] = torch_weights.grad.numpy(force=True)

        weights_grad[:] = self.reorder_weights_to_e3nn(
            weights_grad, not self.config.shared_weights
        )

        return L1_grad, L2_grad, weights_grad


if extlib.TORCH_COMPILE:
    TensorProductConv.register_torch_fakes()
    TensorProductConv.register_autograd()
    TensorProductConv.register_autocast()


# ==================================================================
# Reference implementations for benchmarking


class TensorProductConvKahan(TensorProductConv):
    def __init__(self, config, *, torch_op=True):
        super().__init__(config, torch_op=torch_op, deterministic=True, kahan=True)

    @staticmethod
    def name():
        return "LoopUnrollConvKahan"


class TensorProductConvDeterministic(TensorProductConv):
    def __init__(self, config, *, torch_op=True):
        super().__init__(config, torch_op=torch_op, deterministic=True)

    @staticmethod
    def name():
        return "LoopUnrollConvDeterministic"


class TensorProductConvAtomic(TensorProductConv):
    def __init__(self, config, *, torch_op=True):
        super().__init__(config, torch_op=torch_op, deterministic=False)

    @staticmethod
    def name():
        return "LoopUnrollConvAtomic"


class TensorProductConvScatterSum(ConvolutionBase):
    def __init__(self, config, *, torch_op=True):
        assert torch_op

        super().__init__(config, torch_op=torch_op, deterministic=False)

        self.reference_tp = TensorProduct(config, torch_op=torch_op)
        self.reorder_weights_from_e3nn = self.reference_tp.reorder_weights_from_e3nn
        self.reorder_weights_to_e3nn = self.reference_tp.reorder_weights_to_e3nn

    def forward(self, L1_in, L2_in, weights, rows, cols, sender_perm=None):
        messages = self.reference_tp(L1_in[cols], L2_in, weights)
        return scatter_add_wrapper(messages, rows, L1_in.size(0))

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
