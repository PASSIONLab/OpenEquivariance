import numpy as np

import jax

from functools import partial
from openequivariance.impl_jax import extlib
import hashlib
from openequivariance.core.e3nn_lite import TPProblem, Irreps
from openequivariance.core.LoopUnrollTP import LoopUnrollTP
import jax.numpy as jnp

def hash_attributes(attrs):
    m = hashlib.sha256()

    for key in sorted(attrs.keys()):
        m.update(attrs[key].__repr__().encode("utf-8"))

    hash = int(m.hexdigest()[:16], 16) >> 1
    attrs["hash"] = hash

@partial(jax.custom_vjp, nondiff_argnums=(3,4,5))
def forward(X, Y, W, L3_dim, irrep_dtype, attrs):
    forward_call = jax.ffi.ffi_call("tp_forward", 
        jax.ShapeDtypeStruct((X.shape[0], L3_dim), irrep_dtype))
    return forward_call(X, Y, W, **attrs)

def forward_with_inputs(X, Y, W, L3_dim, irrep_dtype, attrs):
    return forward(X, Y, W, L3_dim, irrep_dtype, attrs), (X, Y, W) 

def backward(L3_dim, irrep_dtype, attrs, inputs, dZ):
    backward_call = jax.ffi.ffi_call("tp_backward",
        (
            jax.ShapeDtypeStruct(inputs[0].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[1].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[2].shape, irrep_dtype),
        ))

    return backward_call(*inputs, dZ, **attrs) 

forward.defvjp(forward_with_inputs, backward)

class TensorProduct(LoopUnrollTP):
    def __init__(self, config):
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

    def forward(self, X, Y, W):
        return forward(X, Y, W, self.L3_dim, self.config.irrep_dtype, self.attrs)

if __name__ == "__main__":
    tp_problem = None
    X_ir, Y_ir, Z_ir = Irreps("1x2e"), Irreps("1x3e"), Irreps("1x2e") 
    instructions=[(0, 0, 0, "uvu", True)]
    problem = TPProblem(X_ir, Y_ir, Z_ir, 
                        instructions, 
                        shared_weights=False, 
                        internal_weights=False)
    tensor_product = TensorProduct(problem)
    batch_size = 1

    # Convert the above to JAX Arrays
    X = jax.random.uniform(jax.random.PRNGKey(0), (batch_size, X_ir.dim), dtype=jax.numpy.float32)
    Y = jax.random.uniform(jax.random.PRNGKey(1), (batch_size, Y_ir.dim), dtype=jax.numpy.float32)
    W = jax.random.uniform(jax.random.PRNGKey(2), (batch_size, tensor_product.weight_numel), dtype=jax.numpy.float32)

    Z = tensor_product.forward(X, Y, W)

    # Test via jax vjp

    ctZ = jnp.ones_like(Z)
    result = jax.vjp(lambda x, y, w: tensor_product.forward(x, y, w), X, Y, W)[1](ctZ)

    print(result)
    print("COMPLETE!")
