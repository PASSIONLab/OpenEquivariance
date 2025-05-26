__all__ = [
    "E3NNTensorProduct",
    "E3NNTensorProductCompiled",
    "E3NNTensorProductCompiledCUDAGraphs",
    "E3NNTensorProductCompiledMaxAutotuneCUDAGraphs",
]

import os
import pathlib
import numpy as np

from openequivariance.implementations.TensorProductBase import TensorProductBase
from openequivariance.implementations.e3nn_lite import TPProblem
from openequivariance.benchmark.logging_utils import getLogger

TORCH_COMPILE_AUTOTUNING_DIR = pathlib.Path("triton_autotuning")

logger = getLogger()


class E3NNTensorProduct(TensorProductBase):
    def __init__(self, config: TPProblem, torch_op=True):
        super().__init__(config, torch_op=torch_op)
        assert self.torch_op

        global torch
        global e3nn
        import torch
        import e3nn
        from e3nn import o3

        e3nn.set_optimization_defaults(jit_script_fx=False)

        assert config.irrep_dtype == config.weight_dtype
        if config.irrep_dtype == np.float64:
            torch.set_default_dtype(torch.float64)

        self.e3nn_tp = o3.TensorProduct(
            config.irreps_in1,
            config.irreps_in2,
            config.irreps_out,
            config.instructions_raw,
            in1_var=config.in1_var,
            in2_var=config.in2_var,
            out_var=config.out_var,
            irrep_normalization=config.irrep_normalization,
            path_normalization=config.path_normalization,
            internal_weights=config.internal_weights,
            shared_weights=config.shared_weights,
        ).to(device="cuda")

        if config.irrep_dtype == np.float64:
            torch.set_default_dtype(torch.float32)  # Reset to default

        self.forward = self.e3nn_tp.__call__

    def forward_cpu(
        self,
        L1_in: np.ndarray,
        L2_in: np.ndarray,
        L3_out: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        torch_L1_in = torch.tensor(L1_in, device="cuda")
        torch_L2_in = torch.tensor(L2_in, device="cuda")
        torch_weights = torch.tensor(weights, device="cuda")

        torch_L3_out = self.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)

        L3_out[:] = torch_L3_out.numpy(force=True)

    def backward_cpu(
        self,
        L1_in: np.ndarray,
        L1_grad: np.ndarray,
        L2_in: np.ndarray,
        L2_grad: np.ndarray,
        L3_grad: np.ndarray,
        weights: np.ndarray,
        weights_grad: np.ndarray,
    ) -> None:
        torch_L1_in = torch.tensor(L1_in, requires_grad=True, device="cuda")
        torch_L2_in = torch.tensor(L2_in, requires_grad=True, device="cuda")
        torch_weights = torch.tensor(weights, requires_grad=True, device="cuda")

        torch_out = self.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)

        torch_L3_grad_in = torch.tensor(L3_grad, device="cuda")

        torch_out.backward(gradient=torch_L3_grad_in)

        L1_grad[:] = torch_L1_in.grad.numpy(force=True)
        L2_grad[:] = torch_L2_in.grad.numpy(force=True)
        weights_grad[:] = torch_weights.grad.numpy(force=True)

    @classmethod
    def name(cls):
        return cls.__name__


class E3NNTensorProductCompiled(E3NNTensorProduct):
    def __init__(
        self,
        config: TPProblem,
        torch_compile_kwargs: dict,
        torch_op: bool = True,
    ):
        super().__init__(config, torch_op=torch_op)
        self.torch_compile_kwargs = torch_compile_kwargs

        logger.debug("Torch compiling e3nn TP")
        logger.debug(msg=f"{torch_compile_kwargs}")
        self.e3nn_tp = torch.compile(self.e3nn_tp, **self.torch_compile_kwargs)
        logger.debug("e3nn TP torch compiled")

        self.forward = self.e3nn_tp.__call__


class E3NNTensorProductCompiledCUDAGraphs(E3NNTensorProductCompiled):
    def __init__(self, config: TPProblem, torch_op=True):
        global torch
        import torch

        torch._dynamo.config.cache_size_limit = 64

        torch_compile_kwargs = {
            "fullgraph": True,
            "backend": "inductor",
            "options": {"triton.cudagraphs": True},
        }
        super().__init__(config, torch_compile_kwargs, torch_op=torch_op)


class E3NNTensorProductCompiledMaxAutotuneCUDAGraphs(E3NNTensorProductCompiled):
    def __init__(self, config: TPProblem, torch_op=True):
        global torch
        import torch

        TORCH_COMPILE_AUTOTUNING_DIR.mkdir(exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(TORCH_COMPILE_AUTOTUNING_DIR)
        os.environ["TRITON_CACHE_DIR"] = str(TORCH_COMPILE_AUTOTUNING_DIR)
        torch._dynamo.config.cache_size_limit = 64

        torch_compile_kwargs = {
            "fullgraph": True,
            "backend": "inductor",
            "options": {
                "max_autotune": True,
                "triton.cudagraphs": True,
                "triton.unique_kernel_names": False,
                "coordinate_descent_tuning": False,
            },
        }
        super().__init__(config, torch_compile_kwargs, torch_op=torch_op)
