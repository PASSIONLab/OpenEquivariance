import numpy as np
import json
import os
import itertools
from typing import Iterator

from openequivariance.core.TensorProductBase import TensorProductBase
from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.logging import getLogger
from openequivariance.benchmark.problems import (
    ChannelwiseTPP,
    FullyConnectedTPProblem,
    SingleInstruction,
)

os.environ["CUEQUIVARIANCE_OPS_USE_JIT"] = "1"

logger = getLogger()


class CUETensorProduct(TensorProductBase):
    def __init__(self, config: TPProblem, torch_op=True):
        super().__init__(config, torch_op=torch_op)

        global torch
        import torch
        import cuequivariance as cue
        import cuequivariance_torch as cuet
        import e3nn.o3 as o3

        # To-do: abstract and place into the TensorProduct class
        self.is_uvw = config.instructions[0].connection_mode == "uvw"

        supported_tpp_types = [
            ChannelwiseTPP,
            FullyConnectedTPProblem,
            SingleInstruction,
        ]

        assert config.irrep_dtype == config.weight_dtype

        np_to_torch_dtype = {np.float32: torch.float32, np.float64: torch.float64}

        class O3_e3nn(cue.O3):
            def __mul__(  # pylint: disable=no-self-argument
                rep1: "O3_e3nn", rep2: "O3_e3nn"
            ) -> Iterator["O3_e3nn"]:
                return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

            @classmethod
            def clebsch_gordan(
                cls, rep1: "O3_e3nn", rep2: "O3_e3nn", rep3: "O3_e3nn"
            ) -> np.ndarray:
                rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

                if rep1.p * rep2.p == rep3.p:
                    return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
                        rep3.dim
                    )
                return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

            def __lt__(  # pylint: disable=no-self-argument
                rep1: "O3_e3nn", rep2: "O3_e3nn"
            ) -> bool:
                rep2 = rep1._from(rep2)
                return (rep1.l, rep1.p) < (rep2.l, rep2.p)

            @classmethod
            def iterator(cls) -> Iterator["O3_e3nn"]:
                for l in itertools.count(0):
                    yield O3_e3nn(l=l, p=1 * (-1) ** l)
                    yield O3_e3nn(l=l, p=-1 * (-1) ** l)

        self.cue_tp = None
        torch_dtype = np_to_torch_dtype[config.irrep_dtype]

        assert any(
            [isinstance(config, supported_ttp_type)]
            for supported_ttp_type in supported_tpp_types
        )
        if isinstance(config, ChannelwiseTPP) or isinstance(config, SingleInstruction):
            self.cue_tp = cuet.ChannelWiseTensorProduct(
                cue.Irreps(O3_e3nn, str(config.irreps_in1)),
                cue.Irreps(O3_e3nn, str(config.irreps_in2)),
                cue.Irreps(O3_e3nn, str(config.irreps_out)),
                layout=cue.ir_mul,
                shared_weights=config.shared_weights,
                internal_weights=config.internal_weights,
                dtype=torch_dtype,
                math_dtype=torch_dtype,
            )

            torch._dynamo.config.cache_size_limit = 64

            self.cue_tp.to("cuda")
            # self.cue_tp = torch.compile(self.cue_tp, fullgraph=True, mode="default")
            self.tp_correctness = self.cue_tp
            self.forward = self.cue_tp.__call__
            self.forward_correctness = lambda x, y, W: self.tp_correctness(x, y, W)

        if isinstance(config, FullyConnectedTPProblem):
            e = cue.descriptors.fully_connected_tensor_product(
                cue.Irreps("O3", str(config.irreps_in1)),
                cue.Irreps("O3", str(config.irreps_in2)),
                cue.Irreps("O3", str(config.irreps_out)),
            )

            assert config.weight_numel == e.inputs[0].irreps.dim
            self.cue_tp = cuet.EquivariantTensorProduct(
                e, layout=cue.ir_mul, math_dtype=np_to_torch_dtype[config.irrep_dtype]
            )

            self.cue_tp.to("cuda")
            self.forward = lambda x, y, W: self.cue_tp(W, x, y)

            # Only used for correctness
            self.tp_correctness = cuet.EquivariantTensorProduct(
                e, layout=cue.mul_ir, math_dtype=np_to_torch_dtype[config.irrep_dtype]
            )
            self.tp_correctness.to("cuda")
            self.forward_correctness = lambda x, y, W: self.tp_correctness(W, x, y)

        self.kernel_names = [
            "TensorProductUniform1dKernel",
            "channelwise_kernel_fwd",
            "channelwise_kernel_bwd",
        ]

    def forward_cpu(
        self,
        L1_in: np.ndarray,
        L2_in: np.ndarray,
        L3_out: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        torch_L1_in = torch.tensor(L1_in, device="cuda")
        torch_L2_in = torch.tensor(L2_in, device="cuda")
        torch_weights = torch.tensor(weights, device="cuda")

        torch_L3_out = self.forward_correctness(torch_L1_in, torch_L2_in, torch_weights)

        L3_out[:] = torch_L3_out.numpy(force=True)

    def backward_cpu(
        self,
        L1_in: np.ndarray,
        L1_grad: np.ndarray,
        L2_in: np.ndarray,
        L2_grad: np.ndarray,
        L3_grad: np.ndarray,
        weights: np.ndarray,
        weights_grad: np.ndarray,
    ) -> None:
        torch_L1_in = torch.tensor(L1_in, requires_grad=True, device="cuda")
        torch_L2_in = torch.tensor(L2_in, requires_grad=True, device="cuda")
        torch_weights = torch.tensor(weights, requires_grad=True, device="cuda")

        torch_out = self.forward_correctness(torch_L1_in, torch_L2_in, torch_weights)

        torch_L3_grad_in = torch.tensor(L3_grad, device="cuda")

        torch_out.backward(gradient=torch_L3_grad_in)

        L1_grad[:] = torch_L1_in.grad.numpy(force=True)
        L2_grad[:] = torch_L2_in.grad.numpy(force=True)
        weights_grad[:] = torch_weights.grad.numpy(force=True)

    def analyze_trace(self, trace_file):
        """
        Need to update this function for the uvw case.
        """
        assert not self.is_uvw

        trace = None
        with open(trace_file, "r") as f:
            trace = json.load(f)

        tp_time = 0.0
        total = 0.0

        for event in trace["traceEvents"]:
            if "args" in event and "stream" in event["args"]:
                event_time_ms = event["dur"] / 1000
                total += event_time_ms

                if (
                    "TensorProductUniform1dKernel" in event["name"]
                    or "channelwise_kernel_fwd" in event["name"]
                    or "channelwise_kernel_bwd" in event["name"]
                ):
                    tp_time += event_time_ms

        return tp_time

    def benchmark_forward(
        self,
        num_warmup: int,
        num_iter: int,
        L1_in: np.ndarray,
        L2_in: np.ndarray,
        L3_buffer: np.ndarray,
        weights: np.ndarray,
        with_torch_overhead: bool = True,
    ) -> np.ndarray:
        return super().benchmark_forward(
            num_warmup,
            num_iter,
            L1_in,
            L2_in,
            L3_buffer,
            weights,
            with_torch_overhead,
            kernel_names=["TensorProductUniform1DKernel", "channelwise_kernel_"],
        )

    def benchmark_backward(
        self,
        num_warmup: int,
        num_iter: int,
        L1_in: np.ndarray,
        L2_in: np.ndarray,
        L3_buffer: np.ndarray,
        weights: np.ndarray,
        with_torch_overhead: bool = True,
    ) -> np.ndarray:
        return super().benchmark_backward(
            num_warmup,
            num_iter,
            L1_in,
            L2_in,
            L3_buffer,
            weights,
            with_torch_overhead,
            kernel_names=self.kernel_names,
        )

    @staticmethod
    def name():
        return "CUETensorProduct"
