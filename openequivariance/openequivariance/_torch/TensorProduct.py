from openequivariance.core.LoopUnrollTP import LoopUnrollTP
from openequivariance import TPProblem
from openequivariance._torch import extlib
import torch
from openequivariance.core.utils import torch_to_oeq_dtype
from openequivariance.benchmark.logging_utils import getLogger
from openequivariance._torch.utils import reorder_torch, string_to_tensor
from openequivariance._torch.NPDoubleBackwardMixin import NumpyDoubleBackwardMixin

import numpy as np

logger = getLogger()


class TensorProduct(torch.nn.Module, LoopUnrollTP, NumpyDoubleBackwardMixin):
    r"""
    Drop-in replacement for ``o3.TensorProduct`` from e3nn. Supports forward,
    backward, and double-backward passes using JIT-compiled kernels. Initialization
    fails if:

    * There are no visible GPUs.
    * The provided tensor product specification is unsupported.

    :param problem: Specification of the tensor product.
    :param use_opaque: This parameter is deprecated. 
    """

    def __init__(self, problem: TPProblem, torch_op=True, use_opaque=False):
        torch.nn.Module.__init__(self)
        self.input_args = {
            "problem": problem,
            "torch_op": torch_op,
            "use_opaque": use_opaque,
        }
        self._init_class()

    def _init_class(self):
        dp = extlib.DeviceProp(0)
        LoopUnrollTP.__init__(
            self,
            self.input_args["problem"],
            dp,
            extlib.postprocess_kernel,
            self.input_args["torch_op"],
        )

        self.kernel= string_to_tensor(self.kernel_string)
        self.weight_numel = self.input_args["problem"].weight_numel


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

    def reorder_weights_from_e3nn(self, weights, has_batch_dim=True):
        return reorder_torch(
            self.forward_schedule, weights, "forward", not self.config.shared_weights
        )

    def reorder_weights_to_e3nn(self, weights, has_batch_dim=True):
        return reorder_torch(
            self.forward_schedule, weights, "backward", not self.config.shared_weights
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, W: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes :math:`W (x \otimes_{\textrm{CG}} y)`, identical to
        ``o3.TensorProduct.forward``.

        :param x: Tensor of shape ``[batch_size, problem.irreps_in1.dim()]``, datatype
                  ``problem.irrep_dtype``.
        :param y: Tensor of shape ``[batch_size, problem.irreps_in2.dim()]``, datatype
                  ``problem.irrep_dtype``.
        :param W: Tensor of datatype ``problem.weight_dtype`` and shape

            * ``[batch_size, problem.weight_numel]`` if ``problem.shared_weights=False``
            * ``[problem.weight_numel]`` if ``problem.shared_weights=True``

        :return: Tensor of shape ``[batch_size, problem.irreps_out.dim()]``, datatype ``problem.irrep_dtype``.
        """
        return torch.ops.libtorch_tp_jit.jit_tp_forward(self.kernel, self.hash, x, y, W)


    @classmethod
    def register_torch_fakes(cls):
        @torch.library.register_fake("libtorch_tp_jit::jit_tp_forward")
        def fake_forward(jit, L1_in, L2_in, W):
            L3_dim = None
            if hasattr(jit, "wrapped_obj"):
                L3_dim = jit.wrapped_obj.kernel_dims["L3_dim"]
            else:
                L3_dim = jit.L3_dim

            return L1_in.new_empty(L1_in.shape[0], L3_dim)

        @torch.library.register_fake("libtorch_tp_jit::jit_tp_backward")
        def fake_backward(jit, L1_in, L2_in, W, L3_grad):
            return torch.empty_like(L1_in), torch.empty_like(L2_in), torch.empty_like(W)

    @classmethod
    def register_autograd(cls):
        backward_op = torch.ops.libtorch_tp_jit.jit_tp_backward

        def setup_context(ctx, inputs, output):
            ctx.kernel, ctx.hash, ctx.L1_in, ctx.L2_in, ctx.weights = inputs

        def backward(ctx, grad_output):
            L1_grad, L2_grad, W_grad = backward_op(
                ctx.kernel, ctx.hash, ctx.L1_in, ctx.L2_in, ctx.weights, grad_output
            )
            return None, None, L1_grad, L2_grad, W_grad

        torch.library.register_autograd(
            "libtorch_tp_jit::jit_tp_forward", backward, setup_context=setup_context
        )

        def setup_context_double_backward(ctx, inputs, output):
            ctx.kernel, ctx.hash, ctx.L1_in, ctx.L2_in, ctx.weights, ctx.L3_grad = inputs

        def double_backward(ctx, E, F, G):
            result = torch.ops.libtorch_tp_jit.jit_tp_double_backward(
                ctx.kernel, ctx.hash, ctx.L1_in, ctx.L2_in, ctx.weights, ctx.L3_grad, E, F, G
            )
            return None, None, result[0], result[1], result[2], result[3]

        torch.library.register_autograd(
            "libtorch_tp_jit::jit_tp_backward",
            double_backward,
            setup_context=setup_context_double_backward,
        )

    @classmethod
    def register_autocast(cls):
        global torch
        import torch

        torch.library.register_autocast(
            "libtorch_tp_jit::jit_tp_forward", "cuda", torch.float32
        )
        torch.library.register_autocast(
            "libtorch_tp_jit::jit_tp_backward", "cuda", torch.float32
        )
        torch.library.register_autocast(
            "libtorch_tp_jit::jit_tp_double_backward", "cuda", torch.float32
        )

    @staticmethod
    def name():
        return "LoopUnrollTP"

    def forward_cpu(
        self,
        L1_in: np.ndarray,
        L2_in: np.ndarray,
        L3_out: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        weights_chunked = self.reorder_weights_from_e3nn(
            weights, not self.config.shared_weights
        )

        torch_L1_in = torch.tensor(L1_in, device="cuda")
        torch_L2_in = torch.tensor(L2_in, device="cuda")
        torch_weights = torch.tensor(weights_chunked, device="cuda")
        torch_L3_out = self.forward(torch_L1_in, torch_L2_in, torch_weights)

        L3_out[:] = torch_L3_out.numpy(force=True)

    def backward_cpu(
        self, L1_in, L1_grad, L2_in, L2_grad, L3_grad, weights, weights_grad
    ) -> None:
        weights_chunked = self.reorder_weights_from_e3nn(
            weights, not self.config.shared_weights
        )

        torch_L1_in = torch.tensor(L1_in, requires_grad=True, device="cuda")
        torch_L2_in = torch.tensor(L2_in, requires_grad=True, device="cuda")
        torch_weights = torch.tensor(weights_chunked, requires_grad=True, device="cuda")

        torch_out = self.forward(torch_L1_in, torch_L2_in, torch_weights)

        torch_L3_grad_in = torch.tensor(L3_grad, device="cuda")

        torch_out.backward(gradient=torch_L3_grad_in)

        L1_grad[:] = torch_L1_in.grad.numpy(force=True)
        L2_grad[:] = torch_L2_in.grad.numpy(force=True)
        weights_grad[:] = torch_weights.grad.numpy(force=True)

        weights_grad[:] = self.reorder_weights_to_e3nn(
            weights_grad, not self.config.shared_weights
        )


if extlib.TORCH_COMPILE:
    TensorProduct.register_torch_fakes()
    TensorProduct.register_autograd()
    TensorProduct.register_autocast()
