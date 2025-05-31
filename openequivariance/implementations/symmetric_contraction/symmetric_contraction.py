# ruff: noqa : E402
import torch

from openequivariance.extlib import GroupMM_F32, GroupMM_F64


class GroupMM:
    next_id = 0

    def __init__(self, dtype, num_elements, batch_size):
        self.id = GroupMM.next_id
        self.num_elements = num_elements
        GroupMM.next_id += 1

        if dtype == torch.float32:
            self.internal = GroupMM_F32(num_elements, batch_size)
        else:
            self.internal = GroupMM_F64(num_elements, batch_size)

        @torch.library.custom_op(
            f"openequivariance::group_gemm{self.id}",
            mutates_args=(),
            device_types="cuda",
        )
        def group_gemm(
            A: torch.Tensor,
            B: torch.Tensor,
            ragged_counts: torch.Tensor,
            M: int,
            K: int,
            ragged_inner: int,
        ) -> torch.Tensor:
            """
            If ragged_inner == 0:
                A is 3D, num_weights x num_features x M x K
                B is batch_size x num_features x K
                C is batch_size x num_features x M
            If ragged_inner == 1:    (needed for the backward pass)
                A is batch_size x num_features x M
                B is batch_size x num_features K
                C is 3D, num_weights x num_features M x K
            """
            shape = None
            if ragged_inner == 0:
                shape = (B.shape[0], B.shape[1], M)
            elif ragged_inner == 1:
                shape = (num_elements, B.shape[1], M, K)

            C = torch.zeros(shape, device="cuda", dtype=A.dtype)
            self.internal.group_gemm(
                A.contiguous().data_ptr(),
                B.contiguous().data_ptr(),
                C.data_ptr(),
                ragged_counts.data_ptr(),
                M,
                K,
                ragged_inner,
            )
            return C

        @group_gemm.register_fake
        def _(A, B, ragged_counts, M, K, ragged_inner):
            if ragged_inner == 0:
                return A.new_empty(B.shape[0], B.shape[1], M)
            elif ragged_inner == 1:
                return A.new_empty(num_elements, batch_size, M, K)

        self.group_gemm = group_gemm

        def setup_context(ctx, inputs, output):
            ctx.A, ctx.B, ctx.ragged_counts, ctx.M, ctx.K, ctx.ragged_inner = inputs

        def backward(ctx, grad_output):
            grad_A, grad_B = None, None

            if ctx.ragged_inner == 0:
                grad_A = group_gemm(
                    grad_output, ctx.B, ctx.ragged_counts, ctx.M, ctx.K, 1
                )
                grad_B = group_gemm(
                    ctx.A.transpose(2, 3),
                    grad_output,
                    ctx.ragged_counts,
                    ctx.K,
                    ctx.M,
                    0,
                )
            elif ctx.ragged_inner == 1:
                grad_A = group_gemm(
                    grad_output, ctx.B, ctx.ragged_counts, ctx.M, ctx.K, 0
                )
                grad_B = group_gemm(
                    grad_output.transpose(2, 3),
                    ctx.A,
                    ctx.ragged_counts,
                    ctx.K,
                    ctx.M,
                    0,
                )

            return grad_A, grad_B, None, None, None, None

        self.group_gemm.register_autograd(backward, setup_context=setup_context)

    def forward(self, weights, vectors, bincounts):
        return self.group_gemm(
            weights, vectors, bincounts, weights.shape[2], weights.shape[3], 0
        )


# --------------------------------------------------------------------------
# The following segment of code was copied from MACE's repo at https://github.com/ACEsuit/mace/blob/b5faaa076c49778fc17493edfecebcabeb960155/mace/tools/cg.py#L106

import collections
from typing import Dict, Optional, Union, List

from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode

_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")


def _wigner_nj(
    irrepss: List[o3.Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(
                    ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim
                )
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            E,
                        )
                    ]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: x[0])


def U_matrix_real(
    irreps_in: Union[str, o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irreps_out = o3.Irreps(irreps_out)
    irrepss = [o3.Irreps(irreps_in)] * correlation

    if correlation == 4:
        filter_ir_mid = [(i, 1 if i % 2 == 0 else -1) for i in range(12)]

    wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)

    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])

    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    out += [last_ir, stack]
    return out


@compile_mode("script")
class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_elements = num_elements
        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        dtype = torch.get_default_dtype()
        for nu in range(1, correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])
        self.groupMM = GroupMM(
            torch.get_default_dtype(), num_elements, self.num_features
        )
        self.num_equivariance = 2 * irrep_out.lmax + 1

        for i in range(correlation, 0, -1):
            # Shapes defining
            num_params = self.U_tensors(i).size()[-1]

            if i == correlation:
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights_max = w
            else:
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights.append(w)

        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

        # Permute the U matrices
        for i in range(correlation, 0, -1):
            U = self.U_tensors(i)
            num_params = U.shape[-1]
            permutation = [U.dim() - 1] + list(range(U.dim()))[:-1]
            U_permuted = U.permute(permutation).reshape(num_params, -1)
            self.register_buffer(f"U_permuted_{i}", U_permuted)

    def forward(
        self, x: torch.Tensor, bincount: torch.Tensor, sorted_indices: torch.Tensor
    ):
        U = self.U_tensors(self.correlation)
        num_params = U.shape[-1]
        num_ell = U.shape[-2]
        U_weights = self.weights_max.transpose(1, 2).reshape(
            -1, num_params
        ) @ self.U_permuted(self.correlation)

        out = self.groupMM.forward(
            U_weights.view(self.num_elements, self.num_features, -1, num_ell),
            x,
            bincount,
        )
        out = out.view([x.shape[0], self.num_features] + list(U.shape[:-2]))

        for i, weight in enumerate(self.weights):
            U = self.U_tensors(self.correlation - i - 1)
            U_perm = self.U_permuted(self.correlation - i - 1)
            c_tensor = weight.transpose(1, 2).reshape(-1, weight.shape[1]) @ U_perm
            c_tensor = c_tensor.view(
                [weight.shape[0], weight.shape[2]] + list(U.shape[:-1])
            )
            c_tensor = c_tensor[sorted_indices] + out

            s = c_tensor.shape
            out = torch.sum(
                c_tensor.view(s[0] * s[1], -1, s[-1]) * x.view(s[0] * s[1], 1, s[-1]),
                dim=2,
            ).view(s[:-1])
            # out = torch.bmm(c_tensor.view(s[0] * s[1], -1, s[-1]), x.view(s[0] * s[1], s[-1], 1)).view(s[:-1])

        return out.view(out.shape[0], -1)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]

    def U_permuted(self, nu: int):
        return dict(self.named_buffers())[f"U_permuted_{nu}"]


@compile_mode("script")
class SymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.num_elements = num_elements
        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        del irreps_in, irreps_out

        if not isinstance(correlation, tuple):
            corr = correlation
            correlation = {}
            for irrep_out in self.irreps_out:
                correlation[irrep_out] = corr

        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        del internal_weights, shared_weights

        self.contractions = torch.nn.ModuleList()
        for irrep_out in self.irreps_out:
            self.contractions.append(
                Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=o3.Irreps(str(irrep_out.ir)),
                    correlation=correlation[irrep_out],
                    internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    weights=self.shared_weights,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        indices = torch.argmax(y, dim=1)
        bincount = torch.bincount(indices, minlength=self.num_elements).to("cpu")
        permutation = torch.argsort(indices)
        inverse_perm = torch.argsort(permutation)
        sorted_indices = indices[permutation]
        x = x[permutation]
        outs = [
            contraction(x, bincount, sorted_indices)
            for contraction in self.contractions
        ]
        outs_cat = torch.cat(outs, dim=-1)[inverse_perm]
        return outs_cat


# --------------------------------------------------------------------------


def test_group_matmul():
    torch.manual_seed(0)
    num_elements = 10
    vpe = 30  # Vectors per element, uniform just for testing
    num_features = 20

    M = 64
    K = 123
    ragged_counts = torch.zeros(num_elements, dtype=torch.int64, device="cpu")

    for i in range(num_elements):
        ragged_counts[i] = vpe

    def test_backward_0():
        group_mm = GroupMM(torch.float32, num_elements, num_features)
        A = torch.randn(num_elements, num_features, M, K).to("cuda")
        B = torch.randn(num_elements * vpe, num_features, K).to("cuda")

        A.requires_grad = True
        B.requires_grad = True

        ground_truth = torch.zeros(num_elements * vpe, num_features, M, device="cuda")

        # Test the forward pass
        for i in range(num_elements):
            B_slice = B[vpe * i : vpe * (i + 1)]
            ground_truth[vpe * i : vpe * (i + 1)] = (
                A[i] @ B_slice.permute(1, 2, 0)
            ).permute(2, 0, 1)

        C_g = torch.randn(num_elements * vpe, num_features, M).to("cuda")
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

        A = torch.zeros(num_elements * vpe, num_features, M, device="cuda")
        B = torch.randn(num_elements * vpe, num_features, K).to("cuda")
        A.requires_grad = True
        B.requires_grad = True

        ground_truth = torch.zeros(num_elements, num_features, M, K).to("cuda")

        for i in range(num_elements):
            A_slice = A[vpe * i : vpe * (i + 1)]
            B_slice = B[vpe * i : vpe * (i + 1)]

            ground_truth[i] = A_slice.permute(1, 2, 0) @ B_slice.permute(1, 0, 2)

        C = group_mm.group_gemm(A, B, ragged_counts, M, K, 1)

        print(torch.norm(C - ground_truth))

        C_g = torch.randn(num_elements, num_features, M, K).to("cuda")
        C_g.requires_grad = True

        ground_truth.backward(C_g, inputs=[A, B])

        A_grad_gt = A.grad.detach().clone()
        B_grad_gt = B.grad.detach().clone()

        A.grad[:] = 0.0
        B.grad[:] = 0.0

        C.backward(C_g, inputs=[A, B])

        print(torch.norm(A.grad - A_grad_gt))
        print(torch.norm(B.grad - B_grad_gt))

    def test_double_backward():
        torch.autograd.set_detect_anomaly(True)
        GroupMM(torch.float32, num_elements, num_features)
        A = torch.randn(num_elements, num_features, M, K).to("cuda")
        B = torch.randn(num_elements * vpe, num_features, K).to("cuda")

        A.requires_grad = True
        B.requires_grad = True

        ground_truth = torch.zeros(num_elements * vpe, num_features, M, device="cuda")

        # Test the forward pass
        for i in range(num_elements):
            B_slice = B[vpe * i : vpe * (i + 1)]
            ground_truth[vpe * i : vpe * (i + 1)] = (
                A[i] @ B_slice.permute(1, 2, 0)
            ).permute(2, 0, 1)

        C_g = torch.randn(num_elements * vpe, num_features, M).to("cuda")
        C_g.requires_grad = True

        ground_truth.backward(C_g, inputs=[A, B], create_graph=True, retain_graph=True)
        dummy = torch.norm(A.grad) + torch.norm(B.grad)
        dummy_grad = torch.randn_like(dummy)

        dummy.backward(gradient=dummy_grad, inputs=[C_g, A, B])

        A_grad_gt = A.grad
        B_grad_gt = B.grad
        C_grad_gt = C_g.grad

        print(torch.norm(A_grad_gt))
        print(torch.norm(B_grad_gt))
        print(torch.norm(C_grad_gt))

    test_backward_0()
    test_backward_1()
    test_double_backward()


if __name__ == "__main__":
    test_group_matmul()
