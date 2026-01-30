import jax
import jax.numpy as jnp
from jax.extend import core
from functools import partial
from jax.interpreters import mlir, ad

def zeros_like(x):
    return jnp.zeros_like(x)

@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def forward(X, Y, W, rows, cols, workspace, sender_perm, L3_dim, kernel, hash):
    forward_call = jax.ffi.ffi_call(
        "conv_forward", jax.ShapeDtypeStruct((X.shape[0], L3_dim), X.dtype)
    )
    return forward_call(X, Y, W, rows, cols, workspace, sender_perm, kernel=kernel, hash=hash)


def forward_fwd(
    X, Y, W, rows, cols, workspace, sender_perm, L3_dim, kernel, hash
):
    out = forward(
        X, Y, W, rows, cols, workspace, sender_perm, L3_dim, kernel, hash 
    )
    return out, (X, Y, W, rows, cols)


def forward_bwd(workspace, sender_perm, L3_dim, kernel, hash, res, dZ):
    X, Y, W, rows, cols = res
    dX, dY, dW = backward(
        X, Y, W, dZ, rows, cols, workspace, sender_perm, kernel=kernel, hash=hash
    )
    return dX, dY, dW, None, None


forward.defvjp(forward_fwd, forward_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9))
def backward(X, Y, W, dZ, rows, cols, workspace, sender_perm, kernel, hash):
    backward_call = jax.ffi.ffi_call(
        "conv_backward",
        (
            jax.ShapeDtypeStruct(X.shape, X.dtype),
            jax.ShapeDtypeStruct(Y.shape, Y.dtype),
            jax.ShapeDtypeStruct(W.shape, W.dtype),
        ),
    )
    return backward_call(X, Y, W, dZ, rows, cols, workspace, sender_perm, kernel=kernel, hash=hash)


def backward_fwd(X, Y, W, dZ, rows, cols, workspace, sender_perm, kernel, hash):
    out = backward(X, Y, W, dZ, rows, cols, workspace, sender_perm, kernel, hash)
    return out, (X, Y, W, dZ, rows, cols)


def backward_bwd(workspace, sender_perm, kernel, hash, res, derivatives):
    X, Y, W, dZ, rows, cols = res
    ddX, ddY, ddW = derivatives

    gX, gY, gW, gdZ = double_backward(
        X,
        Y,
        W,
        dZ,
        ddX,
        ddY,
        ddW,
        rows,
        cols,
        workspace,
        sender_perm,
        kernel,
        hash,
    )

    return gX, gY, gW, gdZ, None, None


backward.defvjp(backward_fwd, backward_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12))
def double_backward(
    X, Y, W, dZ, ddX, ddY, ddW, rows, cols, workspace, sender_perm, kernel, hash 
):
    double_backward_call = jax.ffi.ffi_call(
        "conv_double_backward",
        (
            jax.ShapeDtypeStruct(X.shape, X.dtype),
            jax.ShapeDtypeStruct(Y.shape, Y.dtype),
            jax.ShapeDtypeStruct(W.shape, W.dtype),
            jax.ShapeDtypeStruct(dZ.shape, dZ.dtype),
        ),
    )
    return double_backward_call(
        X, Y, W, dZ, ddX, ddY, ddW, rows, cols, workspace, sender_perm, kernel=kernel, hash=hash
    )


def double_backward_fwd(
    X, Y, W, dZ, ddX, ddY, ddW, rows, cols, workspace, sender_perm, kernel, hash
):
    out = double_backward(
        X,
        Y,
        W,
        dZ,
        ddX,
        ddY,
        ddW,
        rows,
        cols,
        workspace,
        sender_perm,
        kernel,
        hash
    )
    return out, (X, Y, W, dZ, ddX, ddY, ddW, rows, cols)


def triple_backward(
    workspace,
    sender_perm,
    kernel,
    hash,
    residuals,
    tangent_outputs,
):
    X, Y, W, dZ, ddX, ddY, ddW, rows, cols = residuals
    t_dX, t_dY, t_dW, t_ddZ = tangent_outputs

    common_args = (rows, cols, workspace, sender_perm, kernel, hash)

    op1_inputs = (ddX, ddY, W, dZ, t_dX, t_dY, zeros_like(W))
    g1_ddX, g1_ddY, g1_W, g1_dZ = double_backward(*op1_inputs, *common_args)

    op2_inputs = (X, Y, ddW, dZ, t_dX, t_dY, zeros_like(ddW))
    g2_X, g2_Y, g2_ddW, g2_dZ = double_backward(*op2_inputs, *common_args)

    op3_inputs = (ddX, Y, W, dZ, zeros_like(ddX), zeros_like(Y), t_dW)
    g3_ddX, g3_Y, g3_W, g3_dZ = double_backward(*op3_inputs, *common_args)

    op4_inputs = (X, ddY, W, dZ, zeros_like(X), zeros_like(ddY), t_dW)
    g4_X, g4_ddY, g4_W, g4_dZ = double_backward(*op4_inputs, *common_args)

    g5_ddX, g5_Y, g5_W = backward(ddX, Y, W, t_ddZ, *common_args)
    g6_X, g6_ddY, g6_W = backward(X, ddY, W, t_ddZ, *common_args)
    g7_X, g7_Y, g7_ddW = backward(X, Y, ddW, t_ddZ, *common_args)

    grad_X = g2_X + g4_X + g6_X + g7_X
    grad_Y = g2_Y + g3_Y + g5_Y + g7_Y
    grad_W = g1_W + g3_W + g4_W + g5_W + g6_W
    grad_dZ = g1_dZ + g2_dZ + g3_dZ + g4_dZ

    grad_ddX = g1_ddX + g3_ddX + g5_ddX
    grad_ddY = g1_ddY + g4_ddY + g6_ddY
    grad_ddW = g2_ddW + g7_ddW

    return grad_X, grad_Y, grad_W, grad_dZ, grad_ddX, grad_ddY, grad_ddW, None, None


double_backward.defvjp(double_backward_fwd, triple_backward)