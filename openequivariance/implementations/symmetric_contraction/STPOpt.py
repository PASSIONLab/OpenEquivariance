import torch

from openequivariance.extlib import *

#class STPOpt:
#    def __init__(self, num_elements):
#        self.group_mm = GroupMM(num_elements)

    

if __name__ == '__main__':
    num_elements = 10
    batch_size = 30 # Selected arbitrarily, assume that the tensor is not ragged in its last dimension

    M = 64 
    K = 128
    A = torch.randn(num_elements, M, K).to('cuda')
    B = torch.randn(num_elements * batch_size, K).to('cuda')
    C = torch.zeros(num_elements * batch_size, M).to('cuda')

    ragged_counts = torch.zeros(num_elements, dtype=torch.int64, device='cpu')

    for i in range(num_elements):
        ragged_counts[i] = batch_size 

    ground_truth = torch.zeros_like(C) 

    # Test the forward pass 
    for i in range(num_elements):
        B_slice = B[batch_size * i:batch_size * (i+1)]
        ground_truth[batch_size * i:batch_size * (i+1)] = (A[i] @ B_slice.T).T

    group_mm = GroupMM_F32(num_elements)
    group_mm.group_gemm(A.contiguous().data_ptr(), 
                        B.contiguous().data_ptr(),
                        C.data_ptr(), ragged_counts.data_ptr(),
                        M, K, 0)

    print(torch.norm(ground_truth - C))

    #print(ground_truth)
    #print(C)

    C_g = torch.randn(num_elements * batch_size, M).to('cuda')
    ground_truth_grad = torch.zeros_like(A)

    for i in range(num_elements):
        Cg_slice = C_g[batch_size * i:batch_size * (i+1)]
        B_slice = B[batch_size * i:batch_size * (i+1)]
        ground_truth_grad[i] = Cg_slice.T @ B_slice 

    Ag = torch.zeros_like(A)

    group_mm.group_gemm(C_g.contiguous().data_ptr(), 
                        B.contiguous().data_ptr(),
                        Ag.data_ptr(), ragged_counts.data_ptr(),
                        M, K, 1)
    
    print(torch.norm(ground_truth_grad))
    print(torch.norm(Ag))

    print(torch.norm(ground_truth_grad - Ag))