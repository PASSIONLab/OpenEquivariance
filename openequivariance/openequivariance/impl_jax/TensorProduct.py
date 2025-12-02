import numpy as np

import jax

from functools import partial
from openequivariance.impl_jax import extlib
import hashlib
from openequivariance.core.e3nn_lite import TPProblem, Irreps
from openequivariance.core.LoopUnrollTP import LoopUnrollTP
from openequivariance.core.utils import hash_attributes
import jax.numpy as jnp

@partial(jax.custom_vjp, nondiff_argnums=(3,4,5))
def forward(X, Y, W, L3_dim, irrep_dtype, attrs):
    forward_call = jax.ffi.ffi_call("tp_forward", 
        jax.ShapeDtypeStruct((X.shape[0], L3_dim), irrep_dtype))
    return forward_call(X, Y, W, **attrs)

def forward_with_inputs(X, Y, W, L3_dim, irrep_dtype, attrs):
    return forward(X, Y, W, L3_dim, irrep_dtype, attrs), (X, Y, W)

@partial(jax.custom_vjp, nondiff_argnums=(4,5))
def backward(X, Y, W, dZ, irrep_dtype, attrs):
    backward_call = jax.ffi.ffi_call("tp_backward",
        (
            jax.ShapeDtypeStruct(X.shape, irrep_dtype),
            jax.ShapeDtypeStruct(Y.shape, irrep_dtype),
            jax.ShapeDtypeStruct(W.shape, irrep_dtype),
        ))

    return backward_call(X, Y, W, dZ, **attrs)

def backward_with_inputs(X, Y, W, dZ, irrep_dtype, attrs):
    return backward(X, Y, W, dZ, irrep_dtype, attrs), (X, Y, W, dZ)

def double_backward(irrep_dtype, attrs, inputs, derivatives):
    double_backward_call = jax.ffi.ffi_call("tp_double_backward",
        (
            jax.ShapeDtypeStruct(inputs[0].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[1].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[2].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[3].shape, irrep_dtype),
        ))
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
            "kernel_prop": self.kernelProp
        }
        hash_attributes(self.attrs)
 
        self.weight_numel = config.weight_numel
        self.L3_dim = self.config.irreps_out.dim

    def forward(self, X: jax.ndarray, Y: jax.ndarray, W: jax.ndarray) -> jax.ndarray:
        return forward(X, Y, W, self.L3_dim, self.config.irrep_dtype, self.attrs)


def jax_to_torch(x):
    import numpy as np
    import torch
    return torch.tensor(np.asarray(x), requires_grad=True)

if __name__ == "__main__":
    X_ir, Y_ir, Z_ir = Irreps("1x2e"), Irreps("1x3e"), Irreps("1x2e") 
    instructions=[(0, 0, 0, "uvu", True)]
    problem = TPProblem(X_ir, Y_ir, Z_ir, 
                        instructions, 
                        shared_weights=False, 
                        internal_weights=False)
    tensor_product = TensorProduct(problem)
    batch_size = 100

    X = jax.random.uniform(jax.random.PRNGKey(0), (batch_size, X_ir.dim), dtype=jax.numpy.float32)
    Y = jax.random.uniform(jax.random.PRNGKey(1), (batch_size, Y_ir.dim), dtype=jax.numpy.float32)
    W = jax.random.uniform(jax.random.PRNGKey(2), (batch_size, tensor_product.weight_numel), dtype=jax.numpy.float32)
    Z = tensor_product.forward(X, Y, W)

    # Test forward jax vjp 
    ctZ = jax.random.uniform(jax.random.PRNGKey(3), Z.shape, dtype=jax.numpy.float32)
    result = jax.vjp(lambda x, y, w: tensor_product.forward(x, y, w), X, Y, W)[1](ctZ)

    print("COMPLETED FORWARD PASS!")

    ddX = jax.random.uniform(jax.random.PRNGKey(4), X.shape, dtype=jax.numpy.float32)
    ddY = jax.random.uniform(jax.random.PRNGKey(5), Y.shape, dtype=jax.numpy.float32)
    ddW = jax.random.uniform(jax.random.PRNGKey(6), W.shape, dtype=jax.numpy.float32)

    result_double_backward = jax.vjp(
        lambda x, y, w: jax.vjp(lambda a, b, c: tensor_product.forward(a, b, c), x, y, w)[1](ctZ),
        X, Y, W
    )[1]((ddX, ddY, ddW))

    print("COMPLETED DOUBLE BACKWARD PASS!")

    from e3nn import o3 
    e3nn_tp = o3.TensorProduct(X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False)
    print(jax_to_torch(W).shape)

    X_t = jax_to_torch(X)
    Y_t = jax_to_torch(Y)
    W_t = jax_to_torch(W)
    Z_t = jax_to_torch(Z)
    Z_e3nn = e3nn_tp(X_t, Y_t, W_t)
    print("E3NN RESULT:", (Z_e3nn - Z_t).norm())

    Z_e3nn.backward(jax_to_torch(ctZ))
    #^^^ Print the norms of the differences in gradients instead
    print("E3NN GRADS NORM:", (jax_to_torch(result[0]) - X_t.grad).norm(), 
          (jax_to_torch(result[1]) - Y_t.grad).norm(), 
          (jax_to_torch(result[2]) - W_t.grad).norm())