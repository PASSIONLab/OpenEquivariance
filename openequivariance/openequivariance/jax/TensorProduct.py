import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from openequivariance.jax import extlib
from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.LoopUnrollTP import LoopUnrollTP
from openequivariance.jax.utils import reorder_jax 
from openequivariance.jax.jvp.tp_prim import tp_fwd_p
import json

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
