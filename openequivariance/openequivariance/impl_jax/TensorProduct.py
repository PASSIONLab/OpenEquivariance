import numpy as np

import jax
from openequivariance.impl_jax import extlib
import hashlib
from openequivariance.core.e3nn_lite import TPProblem, Irreps
from openequivariance.core.LoopUnrollTP import LoopUnrollTP

def hash_attributes(attrs):
    m = hashlib.sha256()

    for key in sorted(attrs.keys()):
        m.update(attrs[key].__repr__().encode("utf-8"))

    hash = int(m.hexdigest()[:16], 16) >> 1
    attrs["hash"] = hash

class TensorProduct(LoopUnrollTP):
    def __init__(self, config):
        dp = extlib.DeviceProp(0)
        super().__init__(config, dp, extlib.postprocess_kernel, torch_op=False)

        self.attrs = {
            "kernel": self.jit_kernel,
            "forward_config": vars(self.forward_schedule.launch_config),
            "backward_config": vars(self.backward_schedule.launch_config),
            "double_backward_config": vars(self.double_backward_schedule.launch_config),
            "kernel_prop": self.kernelProp
        }
        hash_attributes(self.attrs)

        self.forward_call = jax.ffi.ffi_call(
            "tp_forward", 
            jax.ShapeDtypeStruct((), jax.numpy.int32))

    def forward(self, X, Y, W):
        self.forward_call(X, Y, W, **self.attrs)


if __name__ == "__main__":
    tp_problem = None
    X_ir, Y_ir, Z_ir = Irreps("1x2e"), Irreps("1x3e"), Irreps("1x2e") 
    instructions=[(0, 0, 0, "uvu", True)]
    problem = TPProblem(X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False)
    tensor_product = TensorProduct(problem)

    print("COMPLETE!")