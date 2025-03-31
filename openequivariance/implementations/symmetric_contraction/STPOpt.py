import torch

from openequivariance.extlib import *

class GroupMM:
    next_id = 0
    def __init__(self, dtype, num_elements):
        self.id = GroupMM.next_id
        self.num_elements = num_elements
        GroupMM.next_id += 1

        if dtype==torch.float32:
            self.internal = GroupMM_F32(num_elements) 
        else:
            self.internal = GroupMM_F64(num_elements)


        @torch.library.custom_op(f"openequivariance::group_gemm{self.id}", mutates_args=(), device_types="cuda")
        def group_gemm(A: torch.Tensor, B: torch.Tensor, 
                        ragged_counts: torch.Tensor, M: int, K: int, ragged_inner: int) -> torch.Tensor:
            '''
            If ragged_inner == 0:
                A is 3D, num_weights x M x K
                B is batch_size x K
                C is batch_size x M
            If ragged_inner == 1:    (needed for the backward pass)
                A is batch_size x M
                B is batch_size x K
                C is 3D, num_weights x M x K 
            '''
            shape = None
            if ragged_inner == 0:
                shape = (B.shape[0], M)
            elif ragged_inner == 1:
                shape = (num_elements, M, K)

            C = torch.zeros(shape, device='cuda')
            self.internal.group_gemm(A.contiguous().data_ptr(), 
                                B.contiguous().data_ptr(),
                                C.data_ptr(), ragged_counts.data_ptr(),
                                M, K, ragged_inner)
            return C

        @group_gemm.register_fake
        def _(A, B, ragged_counts, M, K, ragged_inner):
            if ragged_inner == 0:
                return A.empty_like(B.shape[0], M)
            elif ragged_inner == 1:
                return A.empty_like(num_elements, M, K)

        self.group_gemm = group_gemm

        def setup_context(ctx, inputs, output):
            ctx.A, ctx.B, ctx.ragged_counts, ctx.M, ctx.K, ctx.ragged_inner = inputs

        def backward(ctx, grad_output):
            grad_A, grad_B = None, None
            if ctx.ragged_inner == 0:
                grad_A = group_gemm(grad_output, ctx.B, ctx.ragged_counts, ctx.M, ctx.K, 1)
                grad_B = group_gemm(ctx.A.transpose(1, 2), grad_output, ctx.ragged_counts, ctx.K, ctx.M, 0)
            elif ctx.ragged_inner == 1:
                grad_A = group_gemm(grad_output, ctx.B, ctx.ragged_counts, ctx.K, ctx.M, 0)
                grad_B = group_gemm(grad_output.transpose(1, 2), ctx.A, ctx.ragged_counts, ctx.M, ctx.K, 0)

            return grad_A, grad_B, None, None, None, None 

        self.group_gemm.register_autograd(backward, setup_context=setup_context)

    def forward(self, weights, vectors, bincounts):
        return self.group_gemm(weights, vectors, bincounts, weights.shape[1], weights.shape[2], 0)

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