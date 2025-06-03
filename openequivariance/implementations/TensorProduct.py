from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP
from openequivariance import TPProblem
import torch


class TensorProduct(torch.nn.Module, LoopUnrollTP):
    """
    Drop-in replacement for o3.TensorProduct from e3nn. Supports forward,
    backward, and double-backward passes using JIT-compiled kernels.
    """

    def __init__(self, problem: TPProblem, torch_op=True):
        '''
        :param problem: TPProblem instance containing the problem configuration.
        '''
        torch.nn.Module.__init__(self)
        LoopUnrollTP.__init__(self, problem, torch_op)
        self.weight_numel = problem.weight_numel

    @staticmethod
    def name():
        return LoopUnrollTP.name()

    def forward(
        self, L1: torch.Tensor, L2: torch.Tensor, W: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.libtorch_tp_jit.jit_tp_forward(self.internal, L1, L2, W)
