from openequivariance.implementations.convolution.LoopUnrollConv import *

class TensorProductConv(torch.nn.Module, LoopUnrollConv):
    '''
    PyTorch-specialized dispatcher class.
    '''
    def __init__(self, config, torch_op=True, deterministic=False):
        torch.nn.Module.__init__(self)
        LoopUnrollConv.__init__(self, config, idx_dtype=np.int64,
                torch_op=torch_op, deterministic=deterministic)

        if not self.deterministic:
            self.dummy_transpose_perm = torch.tensor([0], dtype=torch.int64)

    @staticmethod
    def name():
        return LoopUnrollConv.name()

    def forward(self,   L1_in: torch.Tensor, L2_in: 
                        torch.Tensor, W: torch.Tensor, 
                        rows: torch.Tensor, cols: torch.Tensor, sender_perm=None) -> torch.Tensor:
        if not self.deterministic:
            sender_perm = self.dummy_transpose_perm
        else:
            assert sender_perm is not None

        return torch.ops.torch_tp_jit.jit_conv_forward(self.internal, L1_in, L2_in, W, rows, cols, self.workspace_buffer, sender_perm) 