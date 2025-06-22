from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP
from openequivariance import TPProblem
from openequivariance import extlib
import torch
import typing
from openequivariance.implementations.utils import torch_to_oeq_dtype


class TensorProduct(torch.nn.Module, LoopUnrollTP):
    r"""
    Drop-in replacement for ``o3.TensorProduct`` from e3nn. Supports forward,
    backward, and double-backward passes using JIT-compiled kernels. Initialization
    fails if:

    * There are no visible GPUs.
    * The provided tensor product specification is unsupported.

    :param problem: Specification of the tensor product.
    :param use_opaque: If ``True, uses an opaque forward pass that cannot be symbolically traced. *Default*: ``False``.
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
        LoopUnrollTP.__init__(
            self, self.input_args["problem"], self.input_args["torch_op"]
        )
        self.weight_numel = self.input_args["problem"].weight_numel
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

    @staticmethod
    def name():
        return LoopUnrollTP.name()

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
        return torch.ops.libtorch_tp_jit.jit_tp_forward(self.internal, x, y, W)

    def _setup_notorchbind(self):
        """
        In case TorchBind is not available (e.g. for torch.compile below PT2.8, etc.),
        set up operations using custom ops.
        """

        @torch.library.custom_op(
            f"openequivariance::tp_forward{self.tp_id}",
            mutates_args=(),
            device_types="cuda",
        )
        def forward(
            L1_in: torch.Tensor, L2_in: torch.Tensor, weights: torch.Tensor
        ) -> torch.Tensor:
            L1_in_c, L2_in_c, weights_c = (
                L1_in.contiguous(),
                L2_in.contiguous(),
                weights.contiguous(),
            )
            L3_out = torch.empty(
                (L1_in_c.shape[0], self.L3.dim), dtype=L1_in.dtype, device=L1_in.device
            )
            self.forward_raw(
                L1_in_c.shape[0],
                L1_in_c.data_ptr(),
                L2_in_c.data_ptr(),
                L3_out.data_ptr(),
                weights_c.data_ptr(),
            )
            return L3_out

        @forward.register_fake
        def _(L1_in, L2_in, weights):
            return L1_in.new_empty(L1_in.shape[0], self.L3.dim)

        self.forward_opaque = forward

        # ---------------- Backward pass -----------------
        @torch.library.custom_op(
            f"openequivariance::tp_grad_helper{self.tp_id}",
            mutates_args=(),
            device_types="cuda",
        )
        def backward_helper(
            L1_in: torch.Tensor,
            L2_in: torch.Tensor,
            weights: torch.Tensor,
            L3_grad: torch.Tensor,
        ) -> typing.List[torch.Tensor]:
            L1_grad = torch.zeros_like(L1_in)
            L2_grad = torch.zeros_like(L2_in)
            weights_grad = torch.empty_like(weights)

            if self.config.shared_weights:
                weights_grad[:] = 0.0

            self.backward_raw(
                L1_in.shape[0],
                L1_in.contiguous().data_ptr(),
                L1_grad.data_ptr(),
                L2_in.contiguous().data_ptr(),
                L2_grad.data_ptr(),
                weights.contiguous().data_ptr(),
                weights_grad.data_ptr(),
                L3_grad.contiguous().data_ptr(),
            )

            return [L1_grad, L2_grad, weights_grad]

        @backward_helper.register_fake
        def _(L1_in, L2_in, weights, L3_grad):
            return [
                L1_in.new_empty(*L1_in.shape),
                L2_in.new_empty(*L2_in.shape),
                weights.new_empty(*weights.shape),
            ]

        def setup_context(ctx, inputs, output):
            ctx.L1_in, ctx.L2_in, ctx.weights = inputs

        def backward(ctx, grad_output):
            result = backward_helper(ctx.L1_in, ctx.L2_in, ctx.weights, grad_output)
            return result[0], result[1], result[2]

        self.forward_opaque.register_autograd(backward, setup_context=setup_context)

        def setup_context_double_backward(ctx, inputs, output):
            ctx.L1_in, ctx.L2_in, ctx.weights, ctx.L3_grad = inputs

        def double_backward(ctx, grad_output):
            A, B, C, D = ctx.L1_in, ctx.L2_in, ctx.L3_grad, ctx.weights
            E, F, G = grad_output[0], grad_output[1], grad_output[2]

            op1 = backward_helper(E, F, D, C)
            op2 = backward_helper(A, B, G, C)
            op3 = forward(E, B, D)
            op4 = backward_helper(E, B, D, C)
            op5 = backward_helper(A, F, D, C)
            op6 = forward(A, F, D)
            op7 = forward(A, B, G)

            return (
                op1[0] + op2[0],
                op1[1] + op2[1],
                (op4[2] + op5[2]),
                (op3 + op6 + op7),
            )

        backward_helper.register_autograd(
            double_backward, setup_context=setup_context_double_backward
        )
