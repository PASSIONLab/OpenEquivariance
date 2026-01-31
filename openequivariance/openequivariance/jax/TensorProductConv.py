import jax
import json
import jax.numpy as jnp
import numpy as np
from typing import Optional
from openequivariance.jax import extlib

from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.LoopUnrollConv import LoopUnrollConv
from openequivariance.jax.utils import reorder_jax

from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.jax.jvp import conv_prim
from openequivariance.jax.vjp import conv_func


logger = getLogger()


class TensorProductConv(LoopUnrollConv):
    r"""
    Identical to ``oeq.torch.TensorProductConv`` with functionality in JAX, with one
    key difference: integer arrays passed to this function must have dtype
    ``np.int32`` (as opposed to ``np.int64`` in the PyTorch version).

    :param problem: Specification of the tensor product.
    :param deterministic: if ``False``, uses atomics for the convolution. If ``True``, uses a deterministic
           fixup-based algorithm. `Default`: ``False``.
    :param kahan: If ``True``, uses Kahan summation to improve accuracy during aggregation. To use this option,
           the input tensors must be in float32 precision AND you must set ``deterministic=True``. *Default*: ``False``.
    """

    def __init__(
        self,
        config: TPProblem,
        deterministic: bool = False,
        kahan: bool = False,
        requires_jvp: bool = True,
    ):
        dp = extlib.DeviceProp(0)
        self.requires_jvp = requires_jvp
        super().__init__(
            config,
            dp,
            extlib.postprocess_kernel,
            idx_dtype=np.int32,
            torch_op=False,
            deterministic=deterministic,
            kahan=kahan,
        )

        self.kernel = json.dumps(
            {
                "kernel": self.jit_kernel,
                "forward_config": vars(self.forward_schedule.launch_config),
                "backward_config": vars(self.backward_schedule.launch_config),
                "double_backward_config": vars(
                    self.double_backward_schedule.launch_config
                ),
                "kernel_prop": self.kernel_prop,
            }
        )
        self.hash = self.kernel.__hash__()

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

        func = conv_prim.conv_fwd_p.bind

        if not self.requires_jvp:
            func = conv_func.forward

        return func(
            X,
            Y,
            W,
            rows,
            cols,
            self.workspace,
            sender_perm,
            L3_dim=self.L3_dim,
            kernel=self.kernel,
            hash=self.hash,
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

        jit_fwd = jax.jit(self.forward)
        result = jit_fwd(
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

        backward_fn = jax.jit(
            jax.vjp(
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
        )

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

        in1_grad, in2_grad, weights_grad, out_dgrad = jax.jit(
            jax.vjp(
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
            )[1]
        )((in1_dgrad_jax, in2_dgrad_jax, weights_dgrad_jax))

        return (
            np.asarray(in1_grad),
            np.asarray(in2_grad),
            np.asarray(weights_grad),
            np.asarray(out_dgrad),
        )
