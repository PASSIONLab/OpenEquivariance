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

    @staticmethod
    def name():
        return LoopUnrollTP.name() 