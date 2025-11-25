import numpy as np

import jax
import openequivariance_extjax as oeq_extjax 
import hashlib
#from openequivariance.core.e3nn_lite import TPProblem

for name, target in oeq_extjax.registrations().items():
    print(name, target)
    jax.ffi.register_ffi_target(name, target, platform="CUDA")

def hash_attributes(attrs):
    m = hashlib.sha256()

    for key in sorted(attrs.keys()):
        m.update(attrs[key].__repr__().encode("utf-8"))

    hash = int(m.hexdigest()[:16], 16) >> 1
    attrs["hash"] = hash

class TensorProduct:
    def __init__(self, problem):
        self.problem = problem

        self.kernel = "BLAH" 
        self.forward_config = {"num_blocks": 42, "num_threads": 256, "smem": 8192 }
        self.backward_config = {}
        self.double_backward_config = {}
        self.kernel_prop = {}
        self.attrs = {
            "kernel": self.kernel,
            "forward_config": self.forward_config,
            "backward_config": self.backward_config,
            "double_backward_config": self.double_backward_config,
            "kernel_prop": self.kernel_prop
        }
        hash_attributes(self.attrs)

        self.forward_call = jax.ffi.ffi_call(
            "tp_forward", 
            jax.ShapeDtypeStruct((), jax.numpy.int32))
        
        self.forward_call(**self.attrs)


if __name__ == "__main__":
    tp_problem = None 
    tensor_product = TensorProduct(tp_problem)
    print("COMPLETE!")