from openequivariance.implementations.convolution.LoopUnrollConv import *

class TensorProductConv(torch.nn.Module, LoopUnrollConv):
    '''
    PyTorch-specialized dispatcher class.
    '''
    def __init__(self, config, torch_op=True, deterministic=False):
        torch.nn.Module.__init__(self)
        LoopUnrollConv.__init__(self, config, idx_dtype=np.int64,
                torch_op=torch_op, deterministic=deterministic)

    @staticmethod
    def name():
        return LoopUnrollConv.name()


    def forward(self,   L1_in: torch.Tensor, L2_in: 
                        torch.Tensor, W: torch.Tensor, 
                        rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        return torch.ops.torch_tp_jit.jit_conv_forward(self.internal, L1_in, L2_in, W, rows, cols, self.workspace_buffer)