import torch

from openequivariance.extlib import *

#class STPOpt:
#    def __init__(self, num_elements):
#        self.group_mm = GroupMM(num_elements)

    

if __name__ == '__main__':
    num_elements = 1
    batch_size = 1 # Selected arbitrarily, assume that the tensor is not ragged in its last dimension

    M = 4
    K = 2
    A = torch.randn(num_elements, M, K).to('cuda')
    B = torch.randn(num_elements * batch_size, K).to('cuda')
    C = torch.ones(num_elements * batch_size, M).to('cuda')

    A[:] = 1.0
    B[:] = 2.0

    offsets = torch.zeros(num_elements + 1, dtype=torch.int64, device='cpu')

    for i in range(num_elements):
        offsets[i+1] = (i+1) * batch_size

    ground_truth = torch.zeros_like(C) 

    # Test the forward pass 
    for i in range(num_elements):
        B_slice = B[batch_size * i:batch_size * (i+1)]
        ground_truth[batch_size * i:batch_size * (i+1)] = (A[i] @ B_slice.T).T

    print(ground_truth)

    group_mm = GroupMM_F32(num_elements)
    group_mm.group_gemm(A.contiguous().data_ptr(), 
                        B.contiguous().data_ptr(),
                        C.data_ptr(), offsets.data_ptr(),
                        M, K, 0)

    print(C)