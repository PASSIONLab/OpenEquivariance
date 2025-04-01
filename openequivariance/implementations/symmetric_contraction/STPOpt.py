import torch

from openequivariance.extlib import *

class GroupMM:
    next_id = 0
    def __init__(self, dtype, num_elements, batch_size):
        self.id = GroupMM.next_id
        self.num_elements = num_elements
        GroupMM.next_id += 1

        if dtype==torch.float32:
            self.internal = GroupMM_F32(num_elements, batch_size) 
        else:
            self.internal = GroupMM_F64(num_elements, batch_size)


        @torch.library.custom_op(f"openequivariance::group_gemm{self.id}", mutates_args=(), device_types="cuda")
        def group_gemm(A: torch.Tensor, B: torch.Tensor, 
                        ragged_counts: torch.Tensor, M: int, K: int, ragged_inner: int) -> torch.Tensor:
            '''
            If ragged_inner == 0:
                A is 3D, num_weights x num_features x M x K
                B is batch_size x num_features x K
                C is batch_size x num_features x M
            If ragged_inner == 1:    (needed for the backward pass)
                A is batch_size x num_features x M
                B is batch_size x num_features K
                C is 3D, num_weights x num_features M x K 
            '''
            shape = None
            if ragged_inner == 0:
                shape = (B.shape[0], B.shape[1], M)
            elif ragged_inner == 1:
                shape = (num_elements, B.shape[1], M, K)

            C = torch.zeros(shape, device='cuda', dtype=A.dtype)
            self.internal.group_gemm(A.contiguous().data_ptr(), 
                                B.contiguous().data_ptr(),
                                C.data_ptr(), ragged_counts.data_ptr(),
                                M, K, ragged_inner)
            return C

        @group_gemm.register_fake
        def _(A, B, ragged_counts, M, K, ragged_inner):
            if ragged_inner == 0:
                return A.empty_like(B.shape[0], B.shape[1], M)
            elif ragged_inner == 1:
                return A.empty_like(num_elements, B.shape[1], batch_size, M, K)

        self.group_gemm = group_gemm

        def setup_context(ctx, inputs, output):
            ctx.A, ctx.B, ctx.ragged_counts, ctx.M, ctx.K, ctx.ragged_inner = inputs

        def backward(ctx, grad_output):
            grad_A, grad_B = None, None

            if ctx.ragged_inner == 0:
                grad_A = group_gemm(grad_output, ctx.B, ctx.ragged_counts, ctx.M, ctx.K, 1)
                grad_B = group_gemm(ctx.A.transpose(2, 3), grad_output, ctx.ragged_counts, ctx.K, ctx.M, 0)
            elif ctx.ragged_inner == 1:
                grad_A = group_gemm(grad_output, ctx.B, ctx.ragged_counts, ctx.M, ctx.K, 0)
                grad_B = group_gemm(grad_output.transpose(2, 3), ctx.A, ctx.ragged_counts, ctx.K, ctx.M, 0)

            return grad_A, grad_B, None, None, None, None 

        self.group_gemm.register_autograd(backward, setup_context=setup_context)

    def forward(self, weights, vectors, bincounts):
        return self.group_gemm(weights, vectors, bincounts, weights.shape[2], weights.shape[3], 0)



def test_group_matmul():
    torch.manual_seed(0)
    num_elements = 10
    vpe= 30 # Vectors per element, uniform just for testing 
    num_features = 20

    M = 64 
    K = 123
    ragged_counts = torch.zeros(num_elements, dtype=torch.int64, device='cpu')

    for i in range(num_elements):
        ragged_counts[i] = vpe 

    def test_backward_0():
        group_mm = GroupMM(torch.float32, num_elements, num_features)
        A = torch.randn(num_elements, num_features, M, K).to('cuda')
        B = torch.randn(num_elements * vpe, num_features, K).to('cuda')

        A.requires_grad = True
        B.requires_grad = True

        ground_truth = torch.zeros(num_elements * vpe, num_features, M, device='cuda')

        # Test the forward pass 
        for i in range(num_elements):
            B_slice = B[vpe * i:vpe * (i+1)]
            ground_truth[vpe * i:vpe * (i+1)] = (A[i] @ B_slice.permute(1, 2, 0)).permute(2, 0, 1)

        C_g = torch.randn(num_elements * vpe, num_features, M).to('cuda')
        C_g.requires_grad = True

        ground_truth.backward(C_g, inputs=[A, B]) 

        A_grad_gt = A.grad.detach().clone()
        B_grad_gt = B.grad.detach().clone()

        A.grad[:] = 0.0
        B.grad[:] = 0.0

        C = group_mm.group_gemm(A, B, ragged_counts, M, K, 0)

        print(torch.norm(ground_truth - C))

        C.backward(C_g, inputs=[A, B])
        print(torch.norm(A_grad_gt - A.grad))
        print(torch.norm(B_grad_gt - B.grad))

    def test_backward_1():
        print("TESTING BACKWARD_1!")
        group_mm = GroupMM(torch.float32, num_elements, num_features)

        A = torch.zeros(num_elements * vpe, num_features, M, device='cuda')
        B = torch.randn(num_elements * vpe, num_features, K).to('cuda')
        A.requires_grad = True
        B.requires_grad = True

        ground_truth = torch.zeros(num_elements, num_features, M, K).to('cuda')

        for i in range(num_elements):
            A_slice = A[vpe * i:vpe * (i+1)]
            B_slice = B[vpe * i:vpe * (i+1)]

            ground_truth[i] = A_slice.permute(1, 2, 0) @ B_slice.permute(1, 0, 2)

        C = group_mm.group_gemm(A, B, ragged_counts, M, K, 1)

        print(torch.norm(C - ground_truth)) 

        C_g = torch.randn(num_elements, M, K).to('cuda')
        C_g.requires_grad = True

        ground_truth.backward(C_g, inputs=[A, B])

        A_grad_gt = A.grad.detach().clone()
        B_grad_gt = B.grad.detach().clone()

        A.grad[:] = 0.0
        B.grad[:] = 0.0

        C.backward(C_g, inputs=[A, B])

        print(torch.norm(A.grad - A_grad_gt))
        print(torch.norm(B.grad - B_grad_gt))

    #test_backward_0()
    test_backward_1()
    #test_double_backward()


if __name__=='__main__':
    test_group_matmul()