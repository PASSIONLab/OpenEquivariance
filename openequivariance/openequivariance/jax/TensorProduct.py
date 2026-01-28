import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from openequivariance.jax import extlib
from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.LoopUnrollTP import LoopUnrollTP
from openequivariance.jax.utils import reorder_jax, trace

import json
import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import mlir, ad

# ==============================================================================
# 1. Forward Primitive
# ==============================================================================

tp_fwd_p = core.Primitive("tp_fwd")

def tp_fwd_impl(X, Y, W, *, L3_dim, kernel, hash):
    irrep_dtype = X.dtype
    out_shape = jax.ShapeDtypeStruct((X.shape[0], L3_dim), irrep_dtype)
    call = jax.ffi.ffi_call("tp_forward", out_shape)
    return call(X, Y, W, L3_dim=L3_dim, kernel=kernel, hash=hash)

def tp_fwd_abstract_eval(X, Y, W, *, L3_dim, kernel, hash):
    return jax.core.ShapedArray((X.shape[0], L3_dim), X.dtype)

tp_fwd_p.def_impl(tp_fwd_impl)
tp_fwd_p.def_abstract_eval(tp_fwd_abstract_eval)
mlir.register_lowering(tp_fwd_p, mlir.lower_fun(tp_fwd_impl, multiple_results=False), platform="cuda")
mlir.register_lowering(tp_fwd_p, mlir.lower_fun(tp_fwd_impl, multiple_results=False), platform="rocm")


# ==============================================================================
# 2. Backward Primitive
# ==============================================================================

tp_bwd_p = core.Primitive("tp_bwd")

def tp_bwd_impl(X, Y, W, dZ, *, kernel, hash):
    print("hihi, Am here")
    irrep_dtype = X.dtype
    out_shapes = (
        jax.ShapeDtypeStruct(X.shape, irrep_dtype),
        jax.ShapeDtypeStruct(Y.shape, irrep_dtype),
        jax.ShapeDtypeStruct(W.shape, irrep_dtype),
    )
    call = jax.ffi.ffi_call("tp_backward", out_shapes)

    print("MADE IT HERE!")
    print(X.shape, Y.shape, W.shape, dZ.shape)
    print(type(kernel))
    print(hash)
    print(type(X))
    print(type(Y))
    print(type(W))
    print(type(dZ))

    result = call(X, Y, W, dZ, kernel=kernel, hash=hash)
    print("Made call successfully")
    print(type(result))
    return result

def tp_bwd_abstract_eval(X, Y, W, dZ, *, kernel, hash):
    irrep_dtype = X.dtype
    return (
        core.ShapedArray(X.shape, irrep_dtype),
        core.ShapedArray(Y.shape, irrep_dtype),
        core.ShapedArray(W.shape, irrep_dtype),
    )

tp_bwd_p.def_impl(tp_bwd_impl)
tp_bwd_p.def_abstract_eval(tp_bwd_abstract_eval)
mlir.register_lowering(tp_bwd_p, mlir.lower_fun(tp_bwd_impl, multiple_results=True), platform="cuda")
mlir.register_lowering(tp_bwd_p, mlir.lower_fun(tp_bwd_impl, multiple_results=True), platform="rocm")


# ==============================================================================
# 3. Double Backward Primitive
# ==============================================================================

tp_dbwd_p = core.Primitive("tp_dbwd")

def tp_dbwd_impl(X, Y, W, dZ, ddX, ddY, ddW, *, kernel, hash):
    irrep_dtype = X.dtype
    out_shapes = (
        jax.ShapeDtypeStruct(X.shape, irrep_dtype),
        jax.ShapeDtypeStruct(Y.shape, irrep_dtype),
        jax.ShapeDtypeStruct(W.shape, irrep_dtype),
        jax.ShapeDtypeStruct(dZ.shape, irrep_dtype),
    )
    call = jax.ffi.ffi_call("tp_double_backward", out_shapes)
    return call(X, Y, W, dZ, ddX, ddY, ddW, kernel=kernel, hash=hash)

def tp_dbwd_abstract_eval(X, Y, W, dZ, ddX, ddY, ddW, *, kernel, hash):
    irrep_dtype = X.dtype
    return (
        jax.core.ShapedArray(X.shape, irrep_dtype),
        jax.core.ShapedArray(Y.shape, irrep_dtype),
        jax.core.ShapedArray(W.shape, irrep_dtype),
        jax.core.ShapedArray(dZ.shape, irrep_dtype),
    )

tp_dbwd_p.def_impl(tp_dbwd_impl)
tp_dbwd_p.def_abstract_eval(tp_dbwd_abstract_eval)
mlir.register_lowering(tp_dbwd_p, mlir.lower_fun(tp_dbwd_impl, multiple_results=True), platform="cuda")
mlir.register_lowering(tp_dbwd_p, mlir.lower_fun(tp_dbwd_impl, multiple_results=True), platform="rocm")

# ==============================================================================
# 4. Forward JVP Primitive Definition
# ==============================================================================

tp_fwd_jvp_p = core.Primitive("tp_fwd_jvp")

def tp_fwd_jvp_impl(X, Y, W, dX, dY, dW, *, L3_dim, kernel, hash):
    term1 = tp_fwd_p.bind(dX, Y, W, L3_dim=L3_dim, kernel=kernel, hash=hash)
    term2 = tp_fwd_p.bind(X, dY, W, L3_dim=L3_dim, kernel=kernel, hash=hash)
    term3 = tp_fwd_p.bind(X, Y, dW, L3_dim=L3_dim, kernel=kernel, hash=hash)
    return term1 + term2 + term3

def tp_fwd_jvp_abstract_eval(X, Y, W, dX, dY, dW, *, L3_dim, kernel, hash):
    return jax.core.ShapedArray((X.shape[0], L3_dim), X.dtype)

tp_fwd_jvp_p.def_impl(tp_fwd_jvp_impl)
tp_fwd_jvp_p.def_abstract_eval(tp_fwd_jvp_abstract_eval)


# ==============================================================================
# 5. Transpose Rule (Implicit VJP)
# ==============================================================================

def tp_fwd_jvp_transpose(ct, X, Y, W, dX, dY, dW, *, L3_dim, kernel, hash):
    # This transpose corresponds to the Backward pass.
    # We assert that we are differentiating with respect to the input tangents.
    assert ad.is_undefined_primal(dX)
    assert ad.is_undefined_primal(dY)
    assert ad.is_undefined_primal(dW)

    # If the primals X, Y, W are being differentiated (undefined), we replace 
    # them with zeros for the purpose of this kernel call.
    if ad.is_undefined_primal(X):
        X = jnp.zeros(X.aval.shape, X.aval.dtype)
    if ad.is_undefined_primal(Y):
        Y = jnp.zeros(Y.aval.shape, Y.aval.dtype)
    if ad.is_undefined_primal(W):
        W = jnp.zeros(W.aval.shape, W.aval.dtype)

    print("STARTED HERE!")
    grad_X, grad_Y, grad_W = tp_bwd_p.bind(X, Y, W, ct, kernel=kernel, hash=hash)
    print("GOT OUT HERE!")

    # Return gradients for (X, Y, W, dX, dY, dW). 
    # Primals get None, tangents get the computed gradients.
    return (None, None, None, grad_X, grad_Y, grad_W)

ad.primitive_transposes[tp_fwd_jvp_p] = tp_fwd_jvp_transpose

def ensure_array(tan, primal):
    if type(tan) is ad.Zero:
        return jnp.zeros_like(primal)
    return tan

# ==============================================================================
# 6. JVP Rule for Original Forward Primitive
# ==============================================================================

def tp_fwd_jvp_rule(primals, tangents, *, L3_dim, kernel, hash):
    X, Y, W = primals
    dX, dY, dW = tangents
    
    dX = ensure_array(dX, X)
    dY = ensure_array(dY, Y)
    dW = ensure_array(dW, W)

    out_primal = tp_fwd_p.bind(X, Y, W, L3_dim=L3_dim, kernel=kernel, hash=hash)
    out_tangent = tp_fwd_jvp_p.bind(X, Y, W, dX, dY, dW, L3_dim=L3_dim, kernel=kernel, hash=hash)

    return out_primal, out_tangent

ad.primitive_jvps[tp_fwd_p] = tp_fwd_jvp_rule


# ==============================================================================
# 7. JVP Rule for Forward JVP Primitive (Higher Order)
# ==============================================================================

def tp_fwd_jvp_jvp_rule(primals, tangents, *, L3_dim, kernel, hash):
    tangents_clean = []
    for t, p in zip(tangents, primals):
        if type(t) is ad.Zero:
            tangents_clean.append(jnp.zeros_like(p))
        else:
            tangents_clean.append(t)
    tangents_clean = tuple(tangents_clean)

    def func(x, y, w, dx, dy, dw):
        return tp_fwd_jvp_impl(x, y, w, dx, dy, dw, L3_dim=L3_dim, kernel=kernel, hash=hash)

    return jax.jvp(func, primals, tangents_clean)

ad.primitive_jvps[tp_fwd_jvp_p] = tp_fwd_jvp_jvp_rule


# ==============================================================================
# 8. Backward JVP Primitive Definition 
# ==============================================================================

tp_bwd_jvp_p = core.Primitive("tp_bwd_jvp")

def tp_bwd_jvp_impl(X, Y, W, dZ, tX, tY, tW, tdZ, *, kernel, hash):
    term_dZ = tp_bwd_p.bind(X, Y, W, tdZ, kernel=kernel, hash=hash)
    term_X = tp_bwd_p.bind(tX, Y, W, dZ, kernel=kernel, hash=hash)
    term_Y = tp_bwd_p.bind(X, tY, W, dZ, kernel=kernel, hash=hash)
    term_W = tp_bwd_p.bind(X, Y, tW, dZ, kernel=kernel, hash=hash)
    
    out_dX = term_dZ[0] + term_Y[0] + term_W[0]
    out_dY = term_dZ[1] + term_X[1] + term_W[1]
    out_dW = term_dZ[2] + term_X[2] + term_Y[2]
    
    return out_dX, out_dY, out_dW

def tp_bwd_jvp_abstract_eval(X, Y, W, dZ, tX, tY, tW, tdZ, *, kernel, hash):
    irrep_dtype = X.dtype
    return (
        jax.core.ShapedArray(X.shape, irrep_dtype),
        jax.core.ShapedArray(Y.shape, irrep_dtype),
        jax.core.ShapedArray(W.shape, irrep_dtype),
    )

tp_bwd_jvp_p.def_impl(tp_bwd_jvp_impl)
tp_bwd_jvp_p.def_abstract_eval(tp_bwd_jvp_abstract_eval)


# ==============================================================================
# 9. Transpose Rule for Backward JVP
# ==============================================================================

def tp_bwd_jvp_transpose(ct, X, Y, W, dZ, tX, tY, tW, tdZ, *, kernel, hash):
    ddX, ddY, ddW = ct

    assert ad.is_undefined_primal(tX)
    assert ad.is_undefined_primal(tY)
    assert ad.is_undefined_primal(tW)
    assert ad.is_undefined_primal(tdZ)

    if ad.is_undefined_primal(X): X = jnp.zeros(X.aval.shape, X.aval.dtype)
    if ad.is_undefined_primal(Y): Y = jnp.zeros(Y.aval.shape, Y.aval.dtype)
    if ad.is_undefined_primal(W): W = jnp.zeros(W.aval.shape, W.aval.dtype)
    if ad.is_undefined_primal(dZ): dZ = jnp.zeros(dZ.aval.shape, dZ.aval.dtype)

    g_X, g_Y, g_W, g_dZ = tp_dbwd_p.bind(X, Y, W, dZ, ddX, ddY, ddW, kernel=kernel, hash=hash)

    return (None, None, None, None, g_X, g_Y, g_W, g_dZ)

ad.primitive_transposes[tp_bwd_jvp_p] = tp_bwd_jvp_transpose


# ==============================================================================
# 10. JVP Rule for Backward JVP Primitive (Higher Order)
# ==============================================================================

def tp_bwd_jvp_jvp_rule(primals, tangents, *, kernel, hash):
    tangents_clean = []
    for t, p in zip(tangents, primals):
        if type(t) is ad.Zero:
            tangents_clean.append(jnp.zeros_like(p))
        else:
            tangents_clean.append(t)
    tangents_clean = tuple(tangents_clean)

    def func(x, y, w, dz, tx, ty, tw, tdz):
        return tp_bwd_jvp_impl(x, y, w, dz, tx, ty, tw, tdz, kernel=kernel, hash=hash)

    return jax.jvp(func, primals, tangents_clean)

ad.primitive_jvps[tp_bwd_jvp_p] = tp_bwd_jvp_jvp_rule


# ==============================================================================
# 11. JVP Rule for Original Backward Primitive
# ==============================================================================

def tp_bwd_jvp_rule(primals, tangents, *, kernel, hash):
    X, Y, W, dZ = primals
    tX, tY, tW, tdZ = tangents

    tX, tY, tW, tdZ = [ensure_array(t, p) for t, p in zip(tangents, primals)]
    print("AM HERE!")
    print(type(kernel))
    print(hash)
    out_primal = tp_bwd_p.bind(X, Y, W, dZ, kernel=kernel, hash=hash)
    #out_tangent = tp_bwd_jvp_p.bind(X, Y, W, dZ, tX, tY, tW, tdZ, kernel=kernel, hash=hash)

    #return out_primal, out_tangent
    return out_primal, None

ad.primitive_jvps[tp_bwd_p] = tp_bwd_jvp_rule


class TensorProduct(LoopUnrollTP):
    r"""
    Identical to ``oeq.torch.TensorProduct`` with functionality in JAX.

    :param problem: Specification of the tensor product.
    """

    def __init__(self, problem: TPProblem):
        dp = extlib.DeviceProp(0)
        super().__init__(problem, dp, extlib.postprocess_kernel, torch_op=False)

        self.kernel = json.dumps({
            "kernel": self.jit_kernel,
            "forward_config": vars(self.forward_schedule.launch_config),
            "backward_config": vars(self.backward_schedule.launch_config),
            "double_backward_config": vars(self.double_backward_schedule.launch_config),
            "kernel_prop": self.kernelProp,
        })
        self.hash = self.kernel.__hash__() 

        self.weight_numel = problem.weight_numel
        self.L3_dim = self.config.irreps_out.dim

    def forward(
        self, X: jax.numpy.ndarray, Y: jax.numpy.ndarray, W: jax.numpy.ndarray
    ) -> jax.numpy.ndarray:
        return tp_fwd_p.bind(X, Y, W, L3_dim=self.L3_dim, kernel=self.kernel, hash=self.hash) 

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
