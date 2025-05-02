from openequivariance import extlib
from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP
import torch

class TensorProduct(torch.nn.Module, LoopUnrollTP):
    '''
    PyTorch-specialized dispatcher class that selects the right implementation based on problem
    configuration. 
    '''
    def __init__(self, problem, torch_op=True):
        torch.nn.Module.__init__(self)
        LoopUnrollTP.__init__(self, problem, torch_op)
        self.weight_numel = self.problem.weight_numel

    @staticmethod
    def name():
        return LoopUnrollTP.name()

    def forward(self, L1: torch.Tensor, L2: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.ops.torch_tp_jit.jit_tp_forward(self.internal, L1, L2, W)