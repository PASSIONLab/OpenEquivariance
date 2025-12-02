import numpy as np
from functools import partial
from openequivariance.impl_jax import extlib

from openequivariance.core.e3nn_lite import TPProblem, Irreps
from openequivariance.core.LoopUnrollConv import LoopUnrollConv 
from openequivariance.core.utils import hash_attributes

import jax
import jax.numpy as jnp

from openequivariance.benchmark.logging_utils import getLogger
logger = getLogger()

class TensorProductConv(LoopUnrollConv):
    def __init__(self, config, deterministic=False, kahan=False):
        dp = extlib.DeviceProp(0)
        super().__init__(
            self,
            config, 
            dp, extlib.postprocess_kernel,
            idx_dtype=np.int64,
            torch_op=False,
            deterministic=deterministic,
            kahan=kahan
        )

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

        self.workspace = jnp.zeros((self.workspace_size,), dtype=jnp.uint8)
        logger.info(f"Convolution requires {self.workspace_size // (2 ** 20)}MB of workspace.")
        self.dummy_transpose_perm = jnp.zeros((1,), dtype=jnp.int64)


if __name__=="__main__":
    X_ir, Y_ir, Z_ir = Irreps("1x2e"), Irreps("1x3e"), Irreps("1x2e") 
    instructions=[(0, 0, 0, "uvu", True)]
    problem = TPProblem(X_ir, Y_ir, Z_ir, 
                        instructions, 
                        shared_weights=False, 
                        internal_weights=False)

    conv = TensorProductConv(problem, deterministic=False, kahan=False)
    print("COMPLETE!")