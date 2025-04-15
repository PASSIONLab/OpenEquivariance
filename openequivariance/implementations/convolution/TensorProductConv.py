from openequivariance.implementations.convolution.LoopUnrollConv import *

class TensorProductConv(LoopUnrollConv):
    '''
    Dispatcher class for convolutions. Right now, we just subclass
    LoopUnrollConv.
    '''
    def __init__(self, config, torch_op=True, deterministic=False):
        super().__init__(config, idx_dtype=np.int64,
                torch_op=torch_op, deterministic=deterministic)