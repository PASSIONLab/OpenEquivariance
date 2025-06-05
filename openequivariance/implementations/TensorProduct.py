from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP
from openequivariance import TPProblem
import torch


class TensorProduct(torch.nn.Module, LoopUnrollTP):
    """
    Drop-in replacement for ``o3.TensorProduct`` from e3nn. Supports forward,
    backward, and double-backward passes using JIT-compiled kernels. Initialization
    fails if:

    * There are no visible GPUs.
    * The provided tensor product specification is unsupported.

    :param problem: Specification of the tensor product.
    """

    def __init__(self, problem: TPProblem, torch_op=True):
        torch.nn.Module.__init__(self)
        LoopUnrollTP.__init__(self, problem, torch_op)
        self.weight_numel = problem.weight_numel

    @staticmethod
    def name():
        return LoopUnrollTP.name()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, W: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes :math:`W (x \otimes_{\\textrm{CG}} y)`, identical to
        ``o3.TensorProduct.forward``.

        :param x: Tensor of shape ``[batch_size, problem.irreps_in1.dim()]``, datatype
                  ``problem.irrep_dtype``.
        :param y: Tensor of shape ``[batch_size, problem.irreps_in2.dim()]``, datatype
                  ``problem.irrep_dtype``.
        :param W: Tensor of datatype ``problem.weight_dtype`` and shape

            * ``[batch_size, problem.weight_numel]`` if ``problem.shared_weights=False``
            * ``[problem.weight_numel]`` if ``problem.shared_weights=True``

        :return: Tensor of shape ``[batch_size, problem.irreps_out.dim()]``, datatype ``problem.irrep_dtype``.
        """
        return torch.ops.libtorch_tp_jit.jit_tp_forward(self.internal, x, y, W)
