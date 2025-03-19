import numpy as np
import numpy.linalg as la

from openequivariance.implementations.convolution.ConvolutionBase import *
from openequivariance.implementations.E3NNTensorProduct import *

class NumpyConv(ConvolutionBase):
    def __init__(self, config):
        self.config = config 
        self.L1, self.L2, self.L3 = config.irreps_in1, config.irreps_in2, config.irreps_out
        self.reference_tp = E3NNTensorProduct(config)

    @staticmethod
    def name():
        return "NumpyConvolution" 

    def forward_cpu(self, L1_in, L2_in, weights, L3_out, graph):
        '''
        L1_in:   [nodes, features_L1]
        L2_in:   [edges, features_L2]
        weights: [edges, weight_numel]
        L3_out:  [nodes, features_L3]
        '''
        tp_outputs = np.zeros((graph.nnz, self.L3.dim), dtype=np.float32)
        self.reference_tp.forward_cpu(L1_in[graph.cols], L2_in, tp_outputs, weights)
        np.add.at(L3_out, graph.rows, tp_outputs)