import torch

from openequivariance.extlib import *

#class STPOpt:
#    def __init__(self, num_elements):
#        self.group_mm = GroupMM(num_elements)

    

if __name__ == '__main__':
    num_elements = 1
    batch_size = 1 # Selected arbitrarily, assume that the tensor is not ragged in its last dimension
    group_mm = GroupMM(num_elements)

    M = 4
    K = 2
    A = torch.randn(num_elements, M, K)
    B = torch.randn(num_elements * batch_size, K)
    C = torch.randn(num_elements * batch_size, M)

    offsets = torch.zeros(num_elements + 1, dtype=torch.int64)

    for i in range(num_elements):
        offsets[i+1] = i * batch_size


    ground_truth = torch.zeros_like(C) 

    # Test the forward pass 
    for i in range(num_elements):
        B_slice = B[batch_size * i:batch_size * (i+1)]
        ground_truth[batch_size * i:batch_size * (i+1)] = A[i] @ B_slice

    print(ground_truth)


    group_mm.group_gemm(A.contiguous().data_ptr(), 
                        B.contiguous().data_ptr(),
                        C.data_ptr(), offsets,
                        M, K, 0)


    print(C)

    