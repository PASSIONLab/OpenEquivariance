import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Optional
from openequivariance.jax import extlib

from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.LoopUnrollConv import LoopUnrollConv
from openequivariance.core.utils import hash_attributes
from openequivariance.jax.utils import reorder_jax

from openequivariance.benchmark.logging_utils import getLogger

logger = getLogger()


def zeros_like(x):
    return jnp.zeros_like(x)


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def forward(X, Y, W, rows, cols, workspace, sender_perm, L3_dim, irrep_dtype, attrs):
    forward_call = jax.ffi.ffi_call(
        "conv_forward", jax.ShapeDtypeStruct((X.shape[0], L3_dim), irrep_dtype)
    )
    return forward_call(X, Y, W, rows, cols, workspace, sender_perm, **attrs)


def forward_fwd(
    X, Y, W, rows, cols, workspace, sender_perm, L3_dim, irrep_dtype, attrs
):
    out = forward(
        X, Y, W, rows, cols, workspace, sender_perm, L3_dim, irrep_dtype, attrs
    )
    return out, (X, Y, W, rows, cols)


def forward_bwd(workspace, sender_perm, L3_dim, irrep_dtype, attrs, res, dZ):
    X, Y, W, rows, cols = res
    dX, dY, dW = backward(
        X, Y, W, dZ, rows, cols, workspace, sender_perm, irrep_dtype, attrs
    )
    return dX, dY, dW, None, None


forward.defvjp(forward_fwd, forward_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9))
def backward(X, Y, W, dZ, rows, cols, workspace, sender_perm, irrep_dtype, attrs):
    backward_call = jax.ffi.ffi_call(
        "conv_backward",
        (
            jax.ShapeDtypeStruct(X.shape, irrep_dtype),
            jax.ShapeDtypeStruct(Y.shape, irrep_dtype),
            jax.ShapeDtypeStruct(W.shape, irrep_dtype),
        ),
    )
    return backward_call(X, Y, W, dZ, rows, cols, workspace, sender_perm, **attrs)


def backward_fwd(X, Y, W, dZ, rows, cols, workspace, sender_perm, irrep_dtype, attrs):
    out = backward(X, Y, W, dZ, rows, cols, workspace, sender_perm, irrep_dtype, attrs)
    return out, (X, Y, W, dZ, rows, cols)


def backward_bwd(workspace, sender_perm, irrep_dtype, attrs, res, derivatives):
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
        irrep_dtype,
        attrs,
    )

    return gX, gY, gW, gdZ, None, None


backward.defvjp(backward_fwd, backward_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12))
def double_backward(
    X, Y, W, dZ, ddX, ddY, ddW, rows, cols, workspace, sender_perm, irrep_dtype, attrs
):
    double_backward_call = jax.ffi.ffi_call(
        "conv_double_backward",
        (
            jax.ShapeDtypeStruct(X.shape, irrep_dtype),
            jax.ShapeDtypeStruct(Y.shape, irrep_dtype),
            jax.ShapeDtypeStruct(W.shape, irrep_dtype),
            jax.ShapeDtypeStruct(dZ.shape, irrep_dtype),
        ),
    )
    return double_backward_call(
        X, Y, W, dZ, ddX, ddY, ddW, rows, cols, workspace, sender_perm, **attrs
    )


def double_backward_fwd(
    X, Y, W, dZ, ddX, ddY, ddW, rows, cols, workspace, sender_perm, irrep_dtype, attrs
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
        irrep_dtype,
        attrs,
    )
    return out, (X, Y, W, dZ, ddX, ddY, ddW, rows, cols)


def triple_backward(
    workspace,
    sender_perm,
    irrep_dtype,
    attrs,
    residuals,
    tangent_outputs,
):
    X, Y, W, dZ, ddX, ddY, ddW, rows, cols = residuals
    t_dX, t_dY, t_dW, t_ddZ = tangent_outputs

    common_args = (rows, cols, workspace, sender_perm, irrep_dtype, attrs)

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


class TensorProductConv(LoopUnrollConv):
    def __init__(
        self, config: TPProblem, deterministic: bool = False, kahan: bool = False
    ):
        dp = extlib.DeviceProp(0)
        super().__init__(
            config,
            dp,
            extlib.postprocess_kernel,
            idx_dtype=np.int32,
            torch_op=False,
            deterministic=deterministic,
            kahan=kahan,
        )

        self.attrs = {
            "kernel": self.jit_kernel,
            "forward_config": vars(self.forward_schedule.launch_config),
            "backward_config": vars(self.backward_schedule.launch_config),
            "double_backward_config": vars(self.double_backward_schedule.launch_config),
            "kernel_prop": self.kernel_prop,
        }
        hash_attributes(self.attrs)

        self.weight_numel = config.weight_numel
        self.L3_dim = self.config.irreps_out.dim

        self.workspace = jnp.zeros((self.workspace_size,), dtype=jnp.uint8)
        logger.info(
            f"Convolution requires {self.workspace_size // (2**20)}MB of workspace."
        )
        self.dummy_transpose_perm = jnp.zeros((1,), dtype=jnp.int32)

    def forward(
        self,
        X: jax.numpy.ndarray,
        Y: jax.numpy.ndarray,
        W: jax.numpy.ndarray,
        rows: jax.numpy.ndarray,
        cols: jax.numpy.ndarray,
        sender_perm: Optional[jax.numpy.ndarray] = None,
    ) -> jax.numpy.ndarray:
        if not self.deterministic:
            sender_perm = self.dummy_transpose_perm
        else:
            assert sender_perm is not None, (
                "Must provide sender_perm for deterministic convolutions."
            )

        return forward(
            X,
            Y,
            W,
            rows,
            cols,
            self.workspace,
            sender_perm,
            self.L3_dim,
            self.config.irrep_dtype,
            self.attrs,
        )

    def __call__(
        self,
        X: jax.numpy.ndarray,
        Y: jax.numpy.ndarray,
        W: jax.numpy.ndarray,
        rows: jax.numpy.ndarray,
        cols: jax.numpy.ndarray,
        sender_perm: Optional[jax.numpy.ndarray] = None,
    ) -> jax.numpy.ndarray:
        return self.forward(X, Y, W, rows, cols, sender_perm)

    def reorder_weights_from_e3nn(self, weights, has_batch_dim=True):
        return reorder_jax(self.forward_schedule, weights, "forward", has_batch_dim)

    def reorder_weights_to_e3nn(self, weights, has_batch_dim=True):
        return reorder_jax(self.forward_schedule, weights, "backward", has_batch_dim)

    def forward_cpu(self, L1_in, L2_in, weights, L3_out, graph):
        rows = graph.rows.astype(np.int32)
        cols = graph.cols.astype(np.int32)
        sender_perm = graph.transpose_perm.astype(np.int32)
        weights = self.reorder_weights_from_e3nn(
            weights, has_batch_dim=not self.config.shared_weights
        )
        result = self.forward(
            jax.numpy.asarray(L1_in),
            jax.numpy.asarray(L2_in),
            jax.numpy.asarray(weights),
            jax.numpy.asarray(rows),
            jax.numpy.asarray(cols),
            jax.numpy.asarray(sender_perm),
        )
        L3_out[:] = np.asarray(result)

    def backward_cpu(
        self,
        L1_in,
        L1_grad,
        L2_in,
        L2_grad,
        L3_grad,
        weights,
        weights_grad,
        graph,
    ):
        rows = graph.rows.astype(np.int32)
        cols = graph.cols.astype(np.int32)
        sender_perm = graph.transpose_perm.astype(np.int32)
        weights = self.reorder_weights_from_e3nn(
            weights, has_batch_dim=not self.config.shared_weights
        )

        backward_fn = jax.vjp(
            lambda X, Y, W: self.forward(
                X,
                Y,
                W,
                jax.numpy.asarray(rows),
                jax.numpy.asarray(cols),
                jax.numpy.asarray(sender_perm),
            ),
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
        self, in1, in2, out_grad, weights, weights_dgrad, in1_dgrad, in2_dgrad, graph
    ):
        in1_jax = jax.numpy.asarray(in1)
        in2_jax = jax.numpy.asarray(in2)
        weights_jax = jax.numpy.asarray(weights)
        out_grad_jax = jax.numpy.asarray(out_grad)
        in1_dgrad_jax = jax.numpy.asarray(in1_dgrad)
        in2_dgrad_jax = jax.numpy.asarray(in2_dgrad)
        weights_dgrad_jax = jax.numpy.asarray(weights_dgrad)

        rows_jax = jax.numpy.asarray(graph.rows.astype(self.idx_dtype))
        cols_jax = jax.numpy.asarray(graph.cols.astype(self.idx_dtype))
        sender_perm_jax = jax.numpy.asarray(graph.transpose_perm.astype(self.idx_dtype))

        in1_grad, in2_grad, weights_grad, out_dgrad = jax.vjp(
            lambda x, y, w, o: jax.vjp(
                lambda a, b, c: self.forward(
                    a, b, c, rows_jax, cols_jax, sender_perm_jax
                ),
                x,
                y,
                w,
            )[1](o),
            in1_jax,
            in2_jax,
            weights_jax,
            out_grad_jax,
        )[1]((in1_dgrad_jax, in2_dgrad_jax, weights_dgrad_jax))

        return (
            np.asarray(in1_grad),
            np.asarray(in2_grad),
            np.asarray(weights_grad),
            np.asarray(out_dgrad),
        )
