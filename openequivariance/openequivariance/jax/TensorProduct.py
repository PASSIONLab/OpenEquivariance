import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from openequivariance.jax import extlib
from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.LoopUnrollTP import LoopUnrollTP
from openequivariance.core.utils import hash_attributes
from openequivariance.jax.utils import reorder_jax


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def forward(X, Y, W, L3_dim, irrep_dtype, attrs):
    forward_call = jax.ffi.ffi_call(
        "tp_forward", jax.ShapeDtypeStruct((X.shape[0], L3_dim), irrep_dtype)
    )
    return forward_call(X, Y, W, **attrs)


def forward_fwd(X, Y, W, L3_dim, irrep_dtype, attrs):
    return forward(X, Y, W, L3_dim, irrep_dtype, attrs), (X, Y, W)


def forward_bwd(L3_dim, irrep_dtype, attrs, inputs, dZ):
    X, Y, W = inputs
    return backward(X, Y, W, dZ, irrep_dtype, attrs)


forward.defvjp(forward_fwd, forward_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def backward(X, Y, W, dZ, irrep_dtype, attrs):
    backward_call = jax.ffi.ffi_call(
        "tp_backward",
        (
            jax.ShapeDtypeStruct(X.shape, irrep_dtype),
            jax.ShapeDtypeStruct(Y.shape, irrep_dtype),
            jax.ShapeDtypeStruct(W.shape, irrep_dtype),
        ),
    )
    return backward_call(X, Y, W, dZ, **attrs)


def backward_fwd(X, Y, W, dZ, irrep_dtype, attrs):
    return backward(X, Y, W, dZ, irrep_dtype, attrs), (X, Y, W, dZ)


def backward_bwd(irrep_dtype, attrs, inputs, derivs):
    X, Y, W, dZ = inputs
    ddX, ddY, ddW = derivs
    return double_backward(X, Y, W, dZ, ddX, ddY, ddW, irrep_dtype, attrs)


backward.defvjp(backward_fwd, backward_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(7, 8))
def double_backward(X, Y, W, dZ, ddX, ddY, ddW, irrep_dtype, attrs):
    double_backward_call = jax.ffi.ffi_call(
        "tp_double_backward",
        (
            jax.ShapeDtypeStruct(X.shape, irrep_dtype),
            jax.ShapeDtypeStruct(Y.shape, irrep_dtype),
            jax.ShapeDtypeStruct(W.shape, irrep_dtype),
            jax.ShapeDtypeStruct(dZ.shape, irrep_dtype),
        ),
    )
    return double_backward_call(X, Y, W, dZ, ddX, ddY, ddW, **attrs)


def double_backward_fwd(X, Y, W, dZ, ddX, ddY, ddW, irrep_dtype, attrs):
    out = double_backward(X, Y, W, dZ, ddX, ddY, ddW, irrep_dtype, attrs)
    return out, (X, Y, W, dZ, ddX, ddY, ddW)


def zeros_like(x):
    return jnp.zeros_like(x)


def triple_backward(irrep_dtype, attrs, residuals, tangent_outputs):
    X, Y, W, dZ, ddX, ddY, ddW = residuals
    t_dX, t_dY, t_dW, t_ddZ = tangent_outputs

    op1_inputs = (ddX, ddY, W, dZ, t_dX, t_dY, zeros_like(W))
    g1_ddX, g1_ddY, g1_W, g1_dZ = double_backward(*op1_inputs, irrep_dtype, attrs)

    op2_inputs = (X, Y, ddW, dZ, t_dX, t_dY, zeros_like(ddW))
    g2_X, g2_Y, g2_ddW, g2_dZ = double_backward(*op2_inputs, irrep_dtype, attrs)

    op3_inputs = (ddX, Y, W, dZ, zeros_like(ddX), zeros_like(Y), t_dW)
    g3_ddX, g3_Y, g3_W, g3_dZ = double_backward(*op3_inputs, irrep_dtype, attrs)

    op4_inputs = (X, ddY, W, dZ, zeros_like(X), zeros_like(ddY), t_dW)
    g4_X, g4_ddY, g4_W, g4_dZ = double_backward(*op4_inputs, irrep_dtype, attrs)

    g5_ddX, g5_Y, g5_W = backward(ddX, Y, W, t_ddZ, irrep_dtype, attrs)
    g6_X, g6_ddY, g6_W = backward(X, ddY, W, t_ddZ, irrep_dtype, attrs)
    g7_X, g7_Y, g7_ddW = backward(X, Y, ddW, t_ddZ, irrep_dtype, attrs)

    grad_X = g2_X + g4_X + g6_X + g7_X
    grad_Y = g2_Y + g3_Y + g5_Y + g7_Y
    grad_W = g1_W + g3_W + g4_W + g5_W + g6_W
    grad_dZ = g1_dZ + g2_dZ + g3_dZ + g4_dZ

    grad_ddX = g1_ddX + g3_ddX + g5_ddX
    grad_ddY = g1_ddY + g4_ddY + g6_ddY
    grad_ddW = g2_ddW + g7_ddW

    return grad_X, grad_Y, grad_W, grad_dZ, grad_ddX, grad_ddY, grad_ddW


double_backward.defvjp(double_backward_fwd, triple_backward)


class TensorProduct(LoopUnrollTP):
    r"""
    Identical to ``oeq.torch.TensorProduct`` with functionality in JAX.

    :param problem: Specification of the tensor product.
    """

    def __init__(self, problem: TPProblem):
        dp = extlib.DeviceProp(0)
        super().__init__(problem, dp, extlib.postprocess_kernel, torch_op=False)

        self.attrs = {
            "kernel": self.jit_kernel,
            "forward_config": vars(self.forward_schedule.launch_config),
            "backward_config": vars(self.backward_schedule.launch_config),
            "double_backward_config": vars(self.double_backward_schedule.launch_config),
            "kernel_prop": self.kernelProp,
        }
        hash_attributes(self.attrs)

        self.weight_numel = problem.weight_numel
        self.L3_dim = self.config.irreps_out.dim

    def forward(
        self, X: jax.numpy.ndarray, Y: jax.numpy.ndarray, W: jax.numpy.ndarray
    ) -> jax.numpy.ndarray:
        return forward(X, Y, W, self.L3_dim, self.config.irrep_dtype, self.attrs)

    def __call__(
        self, X: jax.numpy.ndarray, Y: jax.numpy.ndarray, W: jax.numpy.ndarray
    ) -> jax.numpy.ndarray:
        return self.forward(X, Y, W)

    def reorder_weights_from_e3nn(self, weights, has_batch_dim=True):
        return reorder_jax(
            self.forward_schedule, weights, "forward", not self.config.shared_weights
        )

    def reorder_weights_to_e3nn(self, weights, has_batch_dim=True):
        return reorder_jax(
            self.forward_schedule, weights, "backward", not self.config.shared_weights
        )

    def forward_cpu(self, L1_in, L2_in, L3_out, weights) -> None:
        weights = self.reorder_weights_from_e3nn(
            weights, has_batch_dim=not self.config.shared_weights
        )
        result = self.forward(
            jax.numpy.asarray(L1_in),
            jax.numpy.asarray(L2_in),
            jax.numpy.asarray(weights),
        )
        L3_out[:] = np.asarray(result)

    def backward_cpu(
        self, L1_in, L1_grad, L2_in, L2_grad, L3_grad, weights, weights_grad
    ) -> None:
        weights = self.reorder_weights_from_e3nn(
            weights, has_batch_dim=not self.config.shared_weights
        )
        backward_fn = jax.vjp(
            lambda X, Y, W: self.forward(X, Y, W),
            jax.numpy.asarray(L1_in),
            jax.numpy.asarray(L2_in),
            jax.numpy.asarray(weights),
        )[1]
        L1_grad_jax, L2_grad_jax, weights_grad_jax = backward_fn(
            jax.numpy.asarray(L3_grad)
        )
        L1_grad[:] = np.asarray(L1_grad_jax)
        L2_grad[:] = np.asarray(L2_grad_jax)
        weights_grad[:] = np.asarray(weights_grad_jax)
        weights_grad[:] = self.reorder_weights_to_e3nn(
            weights_grad, has_batch_dim=not self.config.shared_weights
        )

    def double_backward_cpu(
        self, in1, in2, out_grad, weights, weights_dgrad, in1_dgrad, in2_dgrad
    ):
        in1_jax = jax.numpy.asarray(in1)
        in2_jax = jax.numpy.asarray(in2)
        weights_jax = jax.numpy.asarray(weights)
        out_grad_jax = jax.numpy.asarray(out_grad)
        in1_dgrad_jax = jax.numpy.asarray(in1_dgrad)
        in2_dgrad_jax = jax.numpy.asarray(in2_dgrad)
        weights_dgrad_jax = jax.numpy.asarray(weights_dgrad)

        in1_grad, in2_grad, weights_grad, out_dgrad = jax.vjp(
            lambda x, y, w, o: jax.vjp(lambda a, b, c: self.forward(a, b, c), x, y, w)[
                1
            ](o),
            in1_jax,
            in2_jax,
            weights_jax,
            out_grad_jax,
        )[1]((in1_dgrad_jax, in2_dgrad_jax, weights_dgrad_jax))

        return in1_grad, in2_grad, weights_grad, out_dgrad