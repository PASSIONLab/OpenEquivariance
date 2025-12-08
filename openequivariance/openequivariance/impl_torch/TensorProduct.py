from openequivariance.core.LoopUnrollTP import LoopUnrollTP
from openequivariance import TPProblem
from openequivariance.impl_torch import extlib
import torch
import typing
from openequivariance.core.utils import torch_to_oeq_dtype
from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.impl_torch.utils import reorder_torch
from openequivariance.impl_torch.NPDoubleBackwardMixin import NumpyDoubleBackwardMixin
from openequivariance.core.e3nn_lite import Irreps

logger = getLogger()


class TensorProduct(torch.nn.Module, LoopUnrollTP, NumpyDoubleBackwardMixin):
    r"""
    Drop-in replacement for ``o3.TensorProduct`` from e3nn. Supports forward,
    backward, and double-backward passes using JIT-compiled kernels. Initialization
    fails if:

    * There are no visible GPUs.
    * The provided tensor product specification is unsupported.

    :param problem: Specification of the tensor product.
    :param use_opaque: If ``True``, uses an opaque forward pass that cannot be symbolically traced. *Default*: ``False``.
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

        internal_cls = None
        if extlib.TORCH_COMPILE:
            internal_cls = torch.classes.libtorch_tp_jit.TorchJITProduct
        else:
            internal_cls = extlib.JITTPImpl

        logger.info("Starting kernel compiler...")
        self.internal = internal_cls(
            self.jit_kernel,
            vars(self.forward_schedule.launch_config),
            vars(self.backward_schedule.launch_config),
            vars(self.double_backward_schedule.launch_config),
            self.kernelProp,
        )
        logger.info("Kernel compiled!")
        logger.info(f"Kernel File Size: {len(self.jit_kernel) // 1024} KB")

        self.weight_numel = self.input_args["problem"].weight_numel
        self._setup_notorchbind()
        if (not extlib.TORCH_COMPILE) or self.input_args["use_opaque"]:
            self.forward = self.forward_opaque

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
        return reorder_torch(self.forward_schedule, weights, "forward", not self.config.shared_weights)

    def reorder_weights_to_e3nn(self, weights, has_batch_dim=True):
        return reorder_torch(self.forward_schedule, weights, "backward", not self.config.shared_weights)

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

    @classmethod
    def register_torch_fakes(cls):
        @torch._library.register_fake_class("libtorch_tp_jit::TorchJITProduct")
        class TorchJITProduct:
            def __init__(
                self,
                kernel_plaintext: str,
                fwd_config: dict[str, int],
                bwd_config: dict[str, int],
                dbl_bwd_config: dict[str, int],
                kernel_dims: dict[str, int],
            ) -> None:
                (
                    self.kernel_plaintext,
                    self.fwd_config,
                    self.bwd_config,
                    self.dbl_bwd_config,
                    self.kernel_dims,
                ) = (
                    kernel_plaintext,
                    fwd_config,
                    bwd_config,
                    dbl_bwd_config,
                    kernel_dims,
                )

            @classmethod
            def __obj_unflatten__(cls, flattened_product):
                return cls(**dict(flattened_product))

            def __len__(self):
                return 0

            def __setstate__(self, state):
                self.kernel_plaintext = state["kernel_plaintext"]
                self.fwd_config = state["fwd_config"]
                self.bwd_config = state["bwd_config"]
                self.dbl_bwd_config = state["dbl_bwd_config"]
                self.kernel_dims = state["kernel_dims"]

            def exec_tensor_product_rawptr(*args, **kwargs):
                pass

            def backward_rawptr(*args, **kwargs):
                pass

            def L3_dim_getter(self):
                return self.kernel_dims["L3_dim"]

            def irrep_dtype_getter(self):
                return self.kernel_dims["irrep_dtype"]

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
            ctx.jit, ctx.L1_in, ctx.L2_in, ctx.weights = inputs

        def backward(ctx, grad_output):
            L1_grad, L2_grad, W_grad = backward_op(
                ctx.jit, ctx.L1_in, ctx.L2_in, ctx.weights, grad_output
            )
            return None, L1_grad, L2_grad, W_grad

        torch.library.register_autograd(
            "libtorch_tp_jit::jit_tp_forward", backward, setup_context=setup_context
        )

        def setup_context_double_backward(ctx, inputs, output):
            ctx.jit, ctx.L1_in, ctx.L2_in, ctx.weights, ctx.L3_grad = inputs

        def double_backward(ctx, E, F, G):
            result = torch.ops.libtorch_tp_jit.jit_tp_double_backward(
                ctx.jit, ctx.L1_in, ctx.L2_in, ctx.weights, ctx.L3_grad, E, F, G
            )
            return None, result[0], result[1], result[2], result[3]

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


if extlib.TORCH_COMPILE and __name__ != "__main__":
    TensorProduct.register_torch_fakes()
    TensorProduct.register_autograd()
    TensorProduct.register_autocast()
