import numpy as np
from functools import partial
from typing import Optional
from openequivariance.impl_jax import extlib

from openequivariance.core.e3nn_lite import TPProblem, Irreps
from openequivariance.core.LoopUnrollConv import LoopUnrollConv 
from openequivariance.core.utils import hash_attributes

import jax
import jax.numpy as jnp

from openequivariance.benchmark.logging_utils import getLogger
logger = getLogger()

@partial(jax.custom_vjp, nondiff_argnums=(3,4,5,6,7,8,9))
def forward(X, Y, W, rows, cols, workspace, sender_perm, L3_dim, irrep_dtype, attrs):
    forward_call = jax.ffi.ffi_call("conv_forward", 
        jax.ShapeDtypeStruct((X.shape[0], L3_dim), irrep_dtype))
    return forward_call(X, Y, W, rows, cols, workspace, sender_perm, **attrs)

def forward_with_inputs(X, Y, W, rows, cols, workspace, sender_perm, L3_dim, irrep_dtype, attrs):
    return forward(X, Y, W, rows, cols, workspace, sender_perm, L3_dim, irrep_dtype, attrs), (X, Y, W, rows, cols, sender_perm, workspace)

@partial(jax.custom_vjp, nondiff_argnums=(4,5,6,7,8,9))
def backward(X, Y, W, dZ, rows, cols, workspace, sender_perm, irrep_dtype, attrs):
    backward_call = jax.ffi.ffi_call("conv_backward", 
        (jax.ShapeDtypeStruct(X.shape, irrep_dtype),
         jax.ShapeDtypeStruct(Y.shape, irrep_dtype),
         jax.ShapeDtypeStruct(W.shape, irrep_dtype)))
    return backward_call(X, Y, W, dZ, rows, cols, workspace, sender_perm, **attrs)

def backward_with_inputs(X, Y, W, dZ, rows, cols, workspace, sender_perm, irrep_dtype, attrs):
    return backward(X, Y, W, dZ, rows, cols, workspace, sender_perm, irrep_dtype, attrs), (X, Y, W, dZ) #rows, cols, sender_perm, workspace)

def double_backward(rows, cols, workspace, sender_perm, irrep_dtype, attrs, inputs, derivatives):
    double_backward_call = jax.ffi.ffi_call("conv_double_backward",
        (
            jax.ShapeDtypeStruct(inputs[0].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[1].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[2].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[3].shape, irrep_dtype),
        ))
    return double_backward_call(*inputs, *derivatives, rows, cols, workspace, sender_perm, **attrs)

def backward_autograd(rows, cols, workspace, sender_perm, L3_dim, irrep_dtype, attrs, inputs, dZ):
    return backward(inputs[0], inputs[1], inputs[2], dZ, rows, cols, workspace, sender_perm, irrep_dtype, attrs)

forward.defvjp(forward_with_inputs, backward_autograd)
backward.defvjp(backward_with_inputs, double_backward)

class TensorProductConv(LoopUnrollConv):
    def __init__(self, config: TPProblem, deterministic: bool = False, kahan: bool = False):
        dp = extlib.DeviceProp(0)
        super().__init__(
            config,
            dp, extlib.postprocess_kernel,
            idx_dtype=np.int32, # N.B. this is distinct from the PyTorch version 
            torch_op=False,
            deterministic=deterministic,
            kahan=kahan
        )

        self.attrs = {
            "kernel": self.jit_kernel,
            "forward_config": vars(self.forward_schedule.launch_config),
            "backward_config": vars(self.backward_schedule.launch_config),
            "double_backward_config": vars(self.double_backward_schedule.launch_config),
            "kernel_prop": self.kernel_prop
        }
        hash_attributes(self.attrs)
 
        self.weight_numel = config.weight_numel
        self.L3_dim = self.config.irreps_out.dim

        self.workspace = jnp.zeros((self.workspace_size,), dtype=jnp.uint8)
        logger.info(f"Convolution requires {self.workspace_size // (2 ** 20)}MB of workspace.")
        self.dummy_transpose_perm = jnp.zeros((1,), dtype=jnp.int32)

    def forward(
            self,
            X: jax.numpy.ndarray, 
            Y: jax.numpy.ndarray, 
            W: jax.numpy.ndarray, 
            rows: jax.numpy.ndarray, 
            cols: jax.numpy.ndarray, 
            sender_perm: Optional[jax.numpy.ndarray] = None) -> jax.numpy.ndarray:
        
        if not self.deterministic:
            sender_perm = self.dummy_transpose_perm
        else:
            assert sender_perm is not None, "Must provide sender_perm for deterministic convolutions." 

        return forward(
            X, Y, W, 
            rows, cols, 
            self.workspace,
            sender_perm,
            self.L3_dim, 
            self.config.irrep_dtype, 
            self.attrs)
    
    def __call__(self,
            X: jax.numpy.ndarray, 
            Y: jax.numpy.ndarray, 
            W: jax.numpy.ndarray, 
            rows: jax.numpy.ndarray, 
            cols: jax.numpy.ndarray, 
            sender_perm: Optional[jax.numpy.ndarray] = None) -> jax.numpy.ndarray:
        return self.forward(X, Y, W, rows, cols, sender_perm)

