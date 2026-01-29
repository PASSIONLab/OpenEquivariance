import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import mlir, ad

# ==============================================================================
# 0. Helpers
# ==============================================================================

def ensure_array(tan, primal):
    if type(tan) is ad.Zero:
        return jnp.zeros_like(primal)
    return tan

# ==============================================================================
# 1. Forward Primitive
# ==============================================================================

conv_fwd_p = core.Primitive("conv_fwd")

def conv_fwd_impl(X, Y, W, rows, cols, workspace, sender_perm, *, L3_dim, kernel, hash):
    irrep_dtype = X.dtype
    out_shape = jax.ShapeDtypeStruct((X.shape[0], L3_dim), irrep_dtype)
    call = jax.ffi.ffi_call("conv_forward", out_shape)
    return call(X, Y, W, rows, cols, workspace, sender_perm, kernel=kernel, hash=hash)

def conv_fwd_abstract_eval(X, Y, W, rows, cols, workspace, sender_perm, *, L3_dim, kernel, hash):
    return jax.core.ShapedArray((X.shape[0], L3_dim), X.dtype)

conv_fwd_p.def_impl(conv_fwd_impl)
conv_fwd_p.def_abstract_eval(conv_fwd_abstract_eval)
mlir.register_lowering(conv_fwd_p, mlir.lower_fun(conv_fwd_impl, multiple_results=False), platform="cuda")
mlir.register_lowering(conv_fwd_p, mlir.lower_fun(conv_fwd_impl, multiple_results=False), platform="rocm")


# ==============================================================================
# 2. Backward Primitive
# ==============================================================================

conv_bwd_p = core.Primitive("conv_bwd")
conv_bwd_p.multiple_results = True

def conv_bwd_impl(X, Y, W, dZ, rows, cols, workspace, sender_perm, *, kernel, hash):
    irrep_dtype = X.dtype
    out_shapes = (
        jax.ShapeDtypeStruct(X.shape, irrep_dtype),
        jax.ShapeDtypeStruct(Y.shape, irrep_dtype),
        jax.ShapeDtypeStruct(W.shape, irrep_dtype),
    )
    call = jax.ffi.ffi_call("conv_backward", out_shapes)
    return call(X, Y, W, dZ, rows, cols, workspace, sender_perm, kernel=kernel, hash=hash)

def conv_bwd_abstract_eval(X, Y, W, dZ, rows, cols, workspace, sender_perm, *, kernel, hash):
    irrep_dtype = X.dtype
    return (
        jax.core.ShapedArray(X.shape, irrep_dtype),
        jax.core.ShapedArray(Y.shape, irrep_dtype),
        jax.core.ShapedArray(W.shape, irrep_dtype),
    )

conv_bwd_p.def_impl(conv_bwd_impl)
conv_bwd_p.def_abstract_eval(conv_bwd_abstract_eval)
mlir.register_lowering(conv_bwd_p, mlir.lower_fun(conv_bwd_impl, multiple_results=True), platform="cuda")
mlir.register_lowering(conv_bwd_p, mlir.lower_fun(conv_bwd_impl, multiple_results=True), platform="rocm")


# ==============================================================================
# 3. Double Backward Primitive
# ==============================================================================

conv_dbwd_p = core.Primitive("conv_dbwd")
conv_dbwd_p.multiple_results = True

def conv_dbwd_impl(X, Y, W, dZ, ddX, ddY, ddW, rows, cols, workspace, sender_perm, *, kernel, hash):
    irrep_dtype = X.dtype
    out_shapes = (
        jax.ShapeDtypeStruct(X.shape, irrep_dtype),
        jax.ShapeDtypeStruct(Y.shape, irrep_dtype),
        jax.ShapeDtypeStruct(W.shape, irrep_dtype),
        jax.ShapeDtypeStruct(dZ.shape, irrep_dtype),
    )
    call = jax.ffi.ffi_call("conv_double_backward", out_shapes)
    return call(X, Y, W, dZ, ddX, ddY, ddW, rows, cols, workspace, sender_perm, kernel=kernel, hash=hash)

def conv_dbwd_abstract_eval(X, Y, W, dZ, ddX, ddY, ddW, rows, cols, workspace, sender_perm, *, kernel, hash):
    irrep_dtype = X.dtype
    return (
        jax.core.ShapedArray(X.shape, irrep_dtype),
        jax.core.ShapedArray(Y.shape, irrep_dtype),
        jax.core.ShapedArray(W.shape, irrep_dtype),
        jax.core.ShapedArray(dZ.shape, irrep_dtype),
    )

conv_dbwd_p.def_impl(conv_dbwd_impl)
conv_dbwd_p.def_abstract_eval(conv_dbwd_abstract_eval)
mlir.register_lowering(conv_dbwd_p, mlir.lower_fun(conv_dbwd_impl, multiple_results=True), platform="cuda")
mlir.register_lowering(conv_dbwd_p, mlir.lower_fun(conv_dbwd_impl, multiple_results=True), platform="rocm")


# ==============================================================================
# 4. Forward JVP Primitive Definition
# ==============================================================================

conv_fwd_jvp_p = core.Primitive("conv_fwd_jvp")

def conv_fwd_jvp_impl(X, Y, W, dX, dY, dW, rows, cols, workspace, sender_perm, *, L3_dim, kernel, hash):
    kwargs = dict(L3_dim=L3_dim, kernel=kernel, hash=hash)
    args_meta = (rows, cols, workspace, sender_perm)
    
    term1 = conv_fwd_p.bind(dX, Y, W, *args_meta, **kwargs)
    term2 = conv_fwd_p.bind(X, dY, W, *args_meta, **kwargs)
    term3 = conv_fwd_p.bind(X, Y, dW, *args_meta, **kwargs)
    return term1 + term2 + term3

def conv_fwd_jvp_abstract_eval(X, Y, W, dX, dY, dW, rows, cols, workspace, sender_perm, *, L3_dim, kernel, hash):
    return jax.core.ShapedArray((X.shape[0], L3_dim), X.dtype)

conv_fwd_jvp_p.def_impl(conv_fwd_jvp_impl)
conv_fwd_jvp_p.def_abstract_eval(conv_fwd_jvp_abstract_eval)


# ==============================================================================
# 5. Transpose Rule (Implicit VJP)
# ==============================================================================

def conv_fwd_jvp_transpose(ct, X, Y, W, dX, dY, dW, rows, cols, workspace, sender_perm, *, L3_dim, kernel, hash):
    assert ad.is_undefined_primal(dX)
    assert ad.is_undefined_primal(dY)
    assert ad.is_undefined_primal(dW)

    if ad.is_undefined_primal(X): X = jnp.zeros(X.aval.shape, X.aval.dtype)
    if ad.is_undefined_primal(Y): Y = jnp.zeros(Y.aval.shape, Y.aval.dtype)
    if ad.is_undefined_primal(W): W = jnp.zeros(W.aval.shape, W.aval.dtype)

    grad_X, grad_Y, grad_W = conv_bwd_p.bind(
        X, Y, W, ct, 
        rows, cols, workspace, sender_perm, 
        kernel=kernel, hash=hash
    )

    return (None, None, None, grad_X, grad_Y, grad_W, None, None, None, None)

ad.primitive_transposes[conv_fwd_jvp_p] = conv_fwd_jvp_transpose


# ==============================================================================
# 6. JVP Rule for Original Forward Primitive
# ==============================================================================

def conv_fwd_jvp_rule(primals, tangents, *, L3_dim, kernel, hash):
    X, Y, W, rows, cols, workspace, sender_perm = primals
    dX, dY, dW, drows, dcols, dworkspace, dsender_perm = tangents

    dX = ensure_array(dX, X)
    dY = ensure_array(dY, Y)
    dW = ensure_array(dW, W)

    out_primal = conv_fwd_p.bind(X, Y, W, rows, cols, workspace, sender_perm, L3_dim=L3_dim, kernel=kernel, hash=hash)
    out_tangent = conv_fwd_jvp_p.bind(X, Y, W, dX, dY, dW, rows, cols, workspace, sender_perm, L3_dim=L3_dim, kernel=kernel, hash=hash)

    return out_primal, out_tangent

ad.primitive_jvps[conv_fwd_p] = conv_fwd_jvp_rule


# ==============================================================================
# 7. JVP Rule for Forward JVP Primitive (Higher Order)
# ==============================================================================

def conv_fwd_jvp_jvp_rule(primals, tangents, *, L3_dim, kernel, hash):
    tangents_clean = []
    for t, p in zip(tangents, primals):
        if type(t) is ad.Zero:
            tangents_clean.append(jnp.zeros_like(p))
        else:
            tangents_clean.append(t)
    tangents_clean = tuple(tangents_clean)

    def func(x, y, w, dx, dy, dw, r, c, ws, sp):
        return conv_fwd_jvp_impl(
            x, y, w, dx, dy, dw, r, c, ws, sp, 
            L3_dim=L3_dim, kernel=kernel, hash=hash
        )

    return jax.jvp(func, primals, tangents_clean)

ad.primitive_jvps[conv_fwd_jvp_p] = conv_fwd_jvp_jvp_rule


# ==============================================================================
# 8. Backward JVP Primitive Definition 
# ==============================================================================

conv_bwd_jvp_p = core.Primitive("conv_bwd_jvp")
conv_bwd_jvp_p.multiple_results = True

def conv_bwd_jvp_impl(X, Y, W, dZ, tX, tY, tW, tdZ, rows, cols, workspace, sender_perm, *, kernel, hash):
    kwargs = dict(kernel=kernel, hash=hash)
    args_meta = (rows, cols, workspace, sender_perm)

    term_dZ = conv_bwd_p.bind(X, Y, W, tdZ, *args_meta, **kwargs)
    term_X = conv_bwd_p.bind(tX, Y, W, dZ, *args_meta, **kwargs)
    term_Y = conv_bwd_p.bind(X, tY, W, dZ, *args_meta, **kwargs)
    term_W = conv_bwd_p.bind(X, Y, tW, dZ, *args_meta, **kwargs)
    
    out_dX = term_dZ[0] + term_Y[0] + term_W[0]
    out_dY = term_dZ[1] + term_X[1] + term_W[1]
    out_dW = term_dZ[2] + term_X[2] + term_Y[2]
    
    return out_dX, out_dY, out_dW

def conv_bwd_jvp_abstract_eval(X, Y, W, dZ, tX, tY, tW, tdZ, rows, cols, workspace, sender_perm, *, kernel, hash):
    irrep_dtype = X.dtype
    return (
        jax.core.ShapedArray(X.shape, irrep_dtype),
        jax.core.ShapedArray(Y.shape, irrep_dtype),
        jax.core.ShapedArray(W.shape, irrep_dtype),
    )

conv_bwd_jvp_p.def_impl(conv_bwd_jvp_impl)
conv_bwd_jvp_p.def_abstract_eval(conv_bwd_jvp_abstract_eval)


# ==============================================================================
# 9. Transpose Rule for Backward JVP
# ==============================================================================

def conv_bwd_jvp_transpose(ct, X, Y, W, dZ, tX, tY, tW, tdZ, rows, cols, workspace, sender_perm, *, kernel, hash):
    ddX, ddY, ddW = ct

    assert ad.is_undefined_primal(tX)
    assert ad.is_undefined_primal(tY)
    assert ad.is_undefined_primal(tW)
    assert ad.is_undefined_primal(tdZ)

    if ad.is_undefined_primal(X): X = jnp.zeros(X.aval.shape, X.aval.dtype)
    if ad.is_undefined_primal(Y): Y = jnp.zeros(Y.aval.shape, Y.aval.dtype)
    if ad.is_undefined_primal(W): W = jnp.zeros(W.aval.shape, W.aval.dtype)
    if ad.is_undefined_primal(dZ): dZ = jnp.zeros(dZ.aval.shape, dZ.aval.dtype)

    g_X, g_Y, g_W, g_dZ = conv_dbwd_p.bind(
        X, Y, W, dZ, ddX, ddY, ddW, 
        rows, cols, workspace, sender_perm, 
        kernel=kernel, hash=hash
    )

    return (None, None, None, None, g_X, g_Y, g_W, g_dZ, None, None, None, None)

ad.primitive_transposes[conv_bwd_jvp_p] = conv_bwd_jvp_transpose


# ==============================================================================
# 10. JVP Rule for Backward JVP Primitive (Higher Order)
# ==============================================================================

def conv_bwd_jvp_jvp_rule(primals, tangents, *, kernel, hash):
    tangents_clean = []
    for t, p in zip(tangents, primals):
        if type(t) is ad.Zero:
            tangents_clean.append(jnp.zeros_like(p))
        else:
            tangents_clean.append(t)
    tangents_clean = tuple(tangents_clean)

    def func(x, y, w, dz, tx, ty, tw, tdz, r, c, ws, sp):
        return conv_bwd_jvp_impl(
            x, y, w, dz, tx, ty, tw, tdz, r, c, ws, sp, 
            kernel=kernel, hash=hash
        )

    return jax.jvp(func, primals, tangents_clean)

ad.primitive_jvps[conv_bwd_jvp_p] = conv_bwd_jvp_jvp_rule


# ==============================================================================
# 11. JVP Rule for Original Backward Primitive
# ==============================================================================

def conv_bwd_jvp_rule(primals, tangents, *, kernel, hash):
    X, Y, W, dZ, rows, cols, workspace, sender_perm = primals
    tX, tY, tW, tdZ, drows, dcols, dworkspace, dsender_perm = tangents

    tX, tY, tW, tdZ = [ensure_array(t, p) for t, p in zip((tX, tY, tW, tdZ), (X, Y, W, dZ))]
    
    out_primal = conv_bwd_p.bind(
        X, Y, W, dZ, rows, cols, workspace, sender_perm, 
        kernel=kernel, hash=hash
    )
    out_tangent = conv_bwd_jvp_p.bind(
        X, Y, W, dZ, tX, tY, tW, tdZ, rows, cols, workspace, sender_perm, 
        kernel=kernel, hash=hash
    )

    return out_primal, out_tangent 

ad.primitive_jvps[conv_bwd_p] = conv_bwd_jvp_rule


# ==============================================================================
# 12. Slow Double Backward Implementation (Reference)
# ==============================================================================

def conv_dbwd_slow(X, Y, W, dZ, ddX, ddY, ddW, rows, cols, workspace, sender_perm, *, L3_dim, kernel, hash):
    kwargs = dict(kernel=kernel, hash=hash)
    args_meta = (rows, cols, workspace, sender_perm)
    
    op1 = conv_bwd_p.bind(ddX, ddY, W, dZ, *args_meta, **kwargs)
    op2 = conv_bwd_p.bind(X, Y, ddW, dZ, *args_meta, **kwargs)
    
    op3 = conv_fwd_p.bind(ddX, Y, W, *args_meta, L3_dim=L3_dim, **kwargs)
    op4 = conv_bwd_p.bind(ddX, Y, W, dZ, *args_meta, **kwargs)
    op5 = conv_bwd_p.bind(X, ddY, W, dZ, *args_meta, **kwargs)
    
    op6 = conv_fwd_p.bind(X, ddY, W, *args_meta, L3_dim=L3_dim, **kwargs)
    op7 = conv_fwd_p.bind(X, Y, ddW, *args_meta, L3_dim=L3_dim, **kwargs)

    grad_X = op1[0] + op2[0]
    grad_Y = op1[1] + op2[1]
    grad_W = op4[2] + op5[2]
    grad_dZ = op3 + op6 + op7

    return grad_X, grad_Y, grad_W, grad_dZ


# ==============================================================================
# 13. JVP rule for double backward (implicit) 
# ==============================================================================

def conv_dbwd_jvp_rule(primals, tangents, *, kernel, hash):
    # Infer L3_dim from dZ (4th input)
    dZ = primals[3]
    L3_dim = dZ.shape[1]

    tangents_clean = []
    for t, p in zip(tangents, primals):
        if type(t) is ad.Zero:
            tangents_clean.append(jnp.zeros_like(p))
        else:
            tangents_clean.append(t)
    tangents_clean = tuple(tangents_clean)

    def func(x, y, w, dz, ddx, ddy, ddw, r, c, ws, sp):
        return conv_dbwd_slow(
            x, y, w, dz, ddx, ddy, ddw, r, c, ws, sp, 
            L3_dim=L3_dim, kernel=kernel, hash=hash
        )

    return jax.jvp(func, primals, tangents_clean)

ad.primitive_jvps[conv_dbwd_p] = conv_dbwd_jvp_rule


# ==============================================================================
# 14. Transpose rule for double backward
# ==============================================================================

def conv_dbwd_transpose(ct, X, Y, W, dZ, ddX, ddY, ddW, rows, cols, workspace, sender_perm, *, kernel, hash):
    # Infer L3_dim from dZ
    L3_dim = dZ.shape[1]

    if ad.is_undefined_primal(X): X = jnp.zeros(X.aval.shape, X.aval.dtype)
    if ad.is_undefined_primal(Y): Y = jnp.zeros(Y.aval.shape, Y.aval.dtype)
    if ad.is_undefined_primal(W): W = jnp.zeros(W.aval.shape, W.aval.dtype)
    if ad.is_undefined_primal(dZ): dZ = jnp.zeros(dZ.aval.shape, dZ.aval.dtype)
    if ad.is_undefined_primal(ddX): ddX = jnp.zeros(ddX.aval.shape, ddX.aval.dtype)
    if ad.is_undefined_primal(ddY): ddY = jnp.zeros(ddY.aval.shape, ddY.aval.dtype)
    if ad.is_undefined_primal(ddW): ddW = jnp.zeros(ddW.aval.shape, ddW.aval.dtype)

    def func(x, y, w, dz, ddx, ddy, ddw, r, c, ws, sp):
        return conv_dbwd_slow(
            x, y, w, dz, ddx, ddy, ddw, r, c, ws, sp, 
            L3_dim=L3_dim, kernel=kernel, hash=hash
        )

    _, vjp_fun = jax.vjp(func, X, Y, W, dZ, ddX, ddY, ddW, rows, cols, workspace, sender_perm)
    input_grads = vjp_fun(ct)
    
    return input_grads

ad.primitive_transposes[conv_dbwd_p] = conv_dbwd_transpose