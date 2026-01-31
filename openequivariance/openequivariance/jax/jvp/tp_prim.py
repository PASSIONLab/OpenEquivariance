import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import mlir, ad
from openequivariance.jax.utils import clean_tensors

# ==============================================================================
# 1. Forward Primitive
# ==============================================================================

tp_fwd_p = core.Primitive("tp_fwd")

def tp_fwd_impl(X, Y, W, *, L3_dim, kernel, hash):
    irrep_dtype = X.dtype
    out_shape = jax.ShapeDtypeStruct((X.shape[0], L3_dim), irrep_dtype)
    call = jax.ffi.ffi_call("tp_forward", out_shape)
    return call(X, Y, W, kernel=kernel, hash=hash)

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
    return call(X, Y, W, dZ, kernel=kernel, hash=hash)

def tp_bwd_abstract_eval(X, Y, W, dZ, *, kernel, hash):
    irrep_dtype = X.dtype
    return (
        jax.core.ShapedArray(X.shape, irrep_dtype),
        jax.core.ShapedArray(Y.shape, irrep_dtype),
        jax.core.ShapedArray(W.shape, irrep_dtype),
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
    kwargs = dict(L3_dim=L3_dim, kernel=kernel, hash=hash)
    
    term1 = tp_fwd_p.bind(dX, Y, W, **kwargs)
    term2 = tp_fwd_p.bind(X, dY, W, **kwargs)
    term3 = tp_fwd_p.bind(X, Y, dW, **kwargs)
    return term1 + term2 + term3

def tp_fwd_jvp_abstract_eval(X, Y, W, dX, dY, dW, *, L3_dim, kernel, hash):
    return jax.core.ShapedArray((X.shape[0], L3_dim), X.dtype)

tp_fwd_jvp_p.def_impl(tp_fwd_jvp_impl)
tp_fwd_jvp_p.def_abstract_eval(tp_fwd_jvp_abstract_eval)


# ==============================================================================
# 5. Transpose Rule (Implicit VJP)
# ==============================================================================

def tp_fwd_jvp_transpose(ct, X, Y, W, dX, dY, dW, *, L3_dim, kernel, hash):
    X, Y, W = clean_tensors(X, Y, W)

    grad_X, grad_Y, grad_W = tp_bwd_p.bind(X, Y, W, ct, kernel=kernel, hash=hash)

    return (None, None, None, grad_X, grad_Y, grad_W)

ad.primitive_transposes[tp_fwd_jvp_p] = tp_fwd_jvp_transpose


# ==============================================================================
# 6. JVP Rule for Original Forward Primitive
# ==============================================================================

def tp_fwd_jvp_rule(primals, tangents, *, L3_dim, kernel, hash):
    X, Y, W = primals
    dX, dY, dW = tangents
    
    dX, dY, dW = clean_tensors(dX, dY, dW)

    out_primal = tp_fwd_p.bind(X, Y, W, L3_dim=L3_dim, kernel=kernel, hash=hash)
    out_tangent = tp_fwd_jvp_p.bind(X, Y, W, dX, dY, dW, L3_dim=L3_dim, kernel=kernel, hash=hash)

    return out_primal, out_tangent

ad.primitive_jvps[tp_fwd_p] = tp_fwd_jvp_rule


# ==============================================================================
# 7. JVP Rule for Forward JVP Primitive (Higher Order)
# ==============================================================================

def tp_fwd_jvp_jvp_rule(primals, tangents, *, L3_dim, kernel, hash):
    tangents_clean = tuple(clean_tensors(*tangents))

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
    kwargs = dict(kernel=kernel, hash=hash)

    term_dZ = tp_bwd_p.bind(X, Y, W, tdZ, **kwargs)
    term_X = tp_bwd_p.bind(tX, Y, W, dZ, **kwargs)
    term_Y = tp_bwd_p.bind(X, tY, W, dZ, **kwargs)
    term_W = tp_bwd_p.bind(X, Y, tW, dZ, **kwargs)
    
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

    if ad.is_undefined_primal(X): X = jnp.zeros(X.aval.shape, X.aval.dtype)
    if ad.is_undefined_primal(Y): Y = jnp.zeros(Y.aval.shape, Y.aval.dtype)
    if ad.is_undefined_primal(W): W = jnp.zeros(W.aval.shape, W.aval.dtype)
    if ad.is_undefined_primal(dZ): dZ = jnp.zeros(dZ.aval.shape, dZ.aval.dtype)

    tensors_clean = clean_tensors(X, Y, W, dZ, ddX, ddY, ddW)

    g_X, g_Y, g_W, g_dZ = tp_dbwd_p.bind(
        *tensors_clean, kernel=kernel, hash=hash
    )

    return (None, None, None, None, g_X, g_Y, g_W, g_dZ)

ad.primitive_transposes[tp_bwd_jvp_p] = tp_bwd_jvp_transpose


# ==============================================================================
# 10. JVP Rule for Backward JVP Primitive (Higher Order)
# ==============================================================================

def tp_bwd_jvp_jvp_rule(primals, tangents, *, kernel, hash):
    tangents_clean = tuple(clean_tensors(*tangents))

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

    tX, tY, tW, tdZ = clean_tensors(tX, tY, tW, tdZ)

    out_primal = tp_bwd_p.bind(X, Y, W, dZ, kernel=kernel, hash=hash)
    out_tangent = tp_bwd_jvp_p.bind(X, Y, W, dZ, tX, tY, tW, tdZ, kernel=kernel, hash=hash)

    return out_primal, out_tangent 

ad.primitive_jvps[tp_bwd_p] = tp_bwd_jvp_rule


# ==============================================================================
# 12. Slow Double Backward Implementation (Reference)
# ==============================================================================

def tp_dbwd_slow(X, Y, W, dZ, ddX, ddY, ddW, *, L3_dim, kernel, hash):
    kwargs = dict(kernel=kernel, hash=hash)
    
    op1 = tp_bwd_p.bind(ddX, ddY, W, dZ, **kwargs)
    op2 = tp_bwd_p.bind(X, Y, ddW, dZ, **kwargs)
    
    op3 = tp_fwd_p.bind(ddX, Y, W, L3_dim=L3_dim, **kwargs)
    op4 = tp_bwd_p.bind(ddX, Y, W, dZ, **kwargs)
    op5 = tp_bwd_p.bind(X, ddY, W, dZ, **kwargs)
    
    op6 = tp_fwd_p.bind(X, ddY, W, L3_dim=L3_dim, **kwargs)
    op7 = tp_fwd_p.bind(X, Y, ddW, L3_dim=L3_dim, **kwargs)

    grad_X = op1[0] + op2[0]
    grad_Y = op1[1] + op2[1]
    grad_W = op4[2] + op5[2]
    grad_dZ = op3 + op6 + op7

    return grad_X, grad_Y, grad_W, grad_dZ


# ==============================================================================
# 13. JVP rule for double backward (implicit) 
# ==============================================================================

def tp_dbwd_jvp_rule(primals, tangents, *, kernel, hash):
    dZ = primals[3] # Infer L3_dim from dZ (4th input)
    L3_dim = dZ.shape[1]

    def func(x, y, w, dz, ddx, ddy, ddw):
        return tp_dbwd_slow(x, y, w, dz, ddx, ddy, ddw, L3_dim=L3_dim, kernel=kernel, hash=hash)

    tangents_clean = tuple(clean_tensors(*tangents))
    return jax.jvp(func, primals, tangents_clean)

ad.primitive_jvps[tp_dbwd_p] = tp_dbwd_jvp_rule


# ==============================================================================
# 14. Transpose rule for double backward
# ==============================================================================

def tp_dbwd_transpose(ct, X, Y, W, dZ, ddX, ddY, ddW, *, kernel, hash):
    L3_dim = dZ.shape[1]

    X, Y, W, dZ, ddX, ddY, ddW = clean_tensors(X, Y, W, dZ, ddX, ddY, ddW)

    def func(x, y, w, dz, ddx, ddy, ddw):
        return tp_dbwd_slow(x, y, w, dz, ddx, ddy, ddw, L3_dim=L3_dim, kernel=kernel, hash=hash)

    _, vjp_fun = jax.vjp(func, X, Y, W, dZ, ddX, ddY, ddW)
    input_grads = vjp_fun(ct)

    return input_grads

ad.primitive_transposes[tp_dbwd_p] = tp_dbwd_transpose