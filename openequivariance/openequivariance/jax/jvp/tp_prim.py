import json
import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import mlir, ad

# Implements the ladder of derivatives for tensor product
# via primitives, JVP, and transpose instead of custom_vjp.

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
tp_bwd_p.multiple_results = True 

def tp_bwd_impl(X, Y, W, dZ, *, kernel, hash):
    irrep_dtype = X.dtype
    out_shapes = (
        jax.ShapeDtypeStruct(X.shape, irrep_dtype),
        jax.ShapeDtypeStruct(Y.shape, irrep_dtype),
        jax.ShapeDtypeStruct(W.shape, irrep_dtype),
    )

    call = jax.ffi.ffi_call("tp_backward", out_shapes)
    result = call(X, Y, W, dZ, kernel=kernel, hash=hash)
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
tp_dbwd_p.multiple_results = True

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

    grad_X, grad_Y, grad_W = tp_bwd_p.bind(X, Y, W, ct, kernel=kernel, hash=hash)

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
tp_bwd_jvp_p.multiple_results = True

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
    out_primal = tp_bwd_p.bind(X, Y, W, dZ, kernel=kernel, hash=hash)
    out_tangent = tp_bwd_jvp_p.bind(X, Y, W, dZ, tX, tY, tW, tdZ, kernel=kernel, hash=hash)

    return out_primal, out_tangent 

ad.primitive_jvps[tp_bwd_p] = tp_bwd_jvp_rule