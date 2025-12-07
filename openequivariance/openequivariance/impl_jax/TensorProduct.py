import jax
import numpy as np
from functools import partial
from openequivariance.impl_jax import extlib
from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.LoopUnrollTP import LoopUnrollTP
from openequivariance.core.utils import hash_attributes
from openequivariance.impl_jax.utils import reorder_jax

@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def forward(X, Y, W, L3_dim, irrep_dtype, attrs):
    forward_call = jax.ffi.ffi_call(
        "tp_forward", jax.ShapeDtypeStruct((X.shape[0], L3_dim), irrep_dtype)
    )
    return forward_call(X, Y, W, **attrs)


def forward_with_inputs(X, Y, W, L3_dim, irrep_dtype, attrs):
    return forward(X, Y, W, L3_dim, irrep_dtype, attrs), (X, Y, W)


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


def backward_with_inputs(X, Y, W, dZ, irrep_dtype, attrs):
    return backward(X, Y, W, dZ, irrep_dtype, attrs), (X, Y, W, dZ)


def double_backward(irrep_dtype, attrs, inputs, derivatives):
    double_backward_call = jax.ffi.ffi_call(
        "tp_double_backward",
        (
            jax.ShapeDtypeStruct(inputs[0].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[1].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[2].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[3].shape, irrep_dtype),
        ),
    )
    return double_backward_call(*inputs, *derivatives, **attrs)


def backward_autograd(L3_dim, irrep_dtype, attrs, inputs, dZ):
    return backward(inputs[0], inputs[1], inputs[2], dZ, irrep_dtype, attrs)


forward.defvjp(forward_with_inputs, backward_autograd)
backward.defvjp(backward_with_inputs, double_backward)


class TensorProduct(LoopUnrollTP):
    def __init__(self, config: TPProblem):
        dp = extlib.DeviceProp(0)
        super().__init__(config, dp, extlib.postprocess_kernel, torch_op=False)

        self.attrs = {
            "kernel": self.jit_kernel,
            "forward_config": vars(self.forward_schedule.launch_config),
            "backward_config": vars(self.backward_schedule.launch_config),
            "double_backward_config": vars(self.double_backward_schedule.launch_config),
            "kernel_prop": self.kernelProp,
        }
        hash_attributes(self.attrs)

        self.weight_numel = config.weight_numel
        self.L3_dim = self.config.irreps_out.dim

    def forward(self, X: jax.numpy.ndarray, Y: jax.numpy.ndarray, W: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return forward(X, Y, W, self.L3_dim, self.config.irrep_dtype, self.attrs)

    def __call__(
        self, X: jax.numpy.ndarray, Y: jax.numpy.ndarray, W: jax.numpy.ndarray
    ) -> jax.numpy.ndarray:
        return self.forward(X, Y, W)

    def reorder_weights_from_e3nn(self, weights, has_batch_dim=True):
        return reorder_jax(self.forward_schedule, weights, "forward", not self.config.shared_weights)

    def reorder_weights_to_e3nn(self, weights, has_batch_dim=True):
        return reorder_jax(self.forward_schedule, weights, "backward", not self.config.shared_weights)

    def forward_cpu(self, L1_in, L2_in, L3_out, weights) -> None:
        result = self.forward(
            jax.numpy.asarray(L1_in),
            jax.numpy.asarray(L2_in),
            jax.numpy.asarray(weights),
        )
        L3_out[:] = np.asarray(result)

    def backward_cpu(
        self, L1_in, L1_grad, L2_in, L2_grad, L3_grad, weights, weights_grad
    ) -> None:
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