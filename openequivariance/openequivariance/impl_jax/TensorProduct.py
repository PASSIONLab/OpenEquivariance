import jax
from functools import partial
from openequivariance.impl_jax import extlib
from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.LoopUnrollTP import LoopUnrollTP
from openequivariance.core.utils import hash_attributes


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

    def forward(self, X: jax.ndarray, Y: jax.ndarray, W: jax.ndarray) -> jax.ndarray:
        return forward(X, Y, W, self.L3_dim, self.config.irrep_dtype, self.attrs)

    def __call__(
        self, X: jax.numpy.ndarray, Y: jax.numpy.ndarray, W: jax.numpy.ndarray
    ) -> jax.numpy.ndarray:
        return self.forward(X, Y, W)
