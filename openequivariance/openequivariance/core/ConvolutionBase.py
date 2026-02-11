import numpy as np
from openequivariance.core.random_buffer_utils import (
    get_random_buffers_forward_conv,
    get_random_buffers_backward_conv,
    get_random_buffers_double_backward_conv,
)

from openequivariance.core.logging_utils import getLogger, bcolors
from openequivariance.core.e3nn_lite import wigner_3j
from openequivariance.core.utils import benchmark

logger = getLogger()


def flops_data_per_tp(config, direction):
    """
    Assumes all interactions are "uvu" for now
    Returns (flops_per_tp, data_per_tp, nnz)
    """
    bytes_per_word = np.dtype(config.irrep_dtype).itemsize

    assert not config.shared_weights
    L1, L2, L3 = config.irreps_in1, config.irreps_in2, config.irreps_out
    ops_per_nz, words_per_tp = None, None
    if direction == "forward":
        ops_per_nz = 3
        words_per_tp = L1.dim + L2.dim + L3.dim + config.weight_numel
    elif direction == "backward":
        ops_per_nz = 9
        words_per_tp = (
            L1.dim
            + L2.dim
            + L3.dim
            + config.weight_numel
            + L1.dim
            + L2.dim
            + config.weight_numel
        )  # Output gradients

    ops_per_tp = 0
    nnz = 0
    for u, v, w, connection_mode, *others in config.instructions:
        tensor = wigner_3j(L1[u].ir.l, L2[v].ir.l, L3[w].ir.l)
        local_nnz = np.count_nonzero(tensor)
        nnz += local_nnz
        ops_per_tp += (
            ops_per_nz * local_nnz * L1[u].mul * L2[v].mul
        )  # Assumes L3.mult(w) = L1.mult(u) * L2.mult(v)

        if connection_mode == "uvu":
            ops_per_tp += L3[w].mul * (2 * L3[w].ir.l + 1)
        elif connection_mode == "uvw":
            ops_per_tp += L1[u].mul * L2[v].mul * L3[w].ir.dim * L3[w].mul

    return ops_per_tp, words_per_tp * bytes_per_word, nnz


class CoordGraph:
    def __init__(self, coords, rows, cols, name):
        """
        Because graphs may change constantly, this class is designed
        to be as light as possible. A directed edge from node
        u to v is indicated by the presence of an index i such that
        rows[i] = u, rows[i] = v.
        """
        assert len(rows) == len(cols)
        self.nnz = len(rows)  # Counts every nonzero in the adjacency matrix
        self.node_count = coords.shape[0]
        self.coords = coords
        self.name = name

        # Sort the original rows / cols
        triples = [(rows[i], cols[i], i) for i in range(self.nnz)]
        triples.sort(key=lambda x: (x[0], x[1]))
        rows = np.array([x[0] for x in triples], dtype=rows.dtype)
        cols = np.array([x[1] for x in triples], dtype=cols.dtype)

        self.rows = rows
        self.cols = cols

        triples = [(cols[i], rows[i], i) for i in range(self.nnz)]
        triples.sort(key=lambda x: (x[0], x[1]))
        self.transpose_perm = np.array([x[2] for x in triples], dtype=self.rows.dtype)


class ConvolutionBase:
    next_conv_id = 0  # Used to assign unique IDs to each conv instance

    def __init__(
        self,
        config,
        *,
        idx_dtype: type[np.generic] = np.int64,
        torch_op=False,
        deterministic=False,
    ):
        config = config.clone()
        self.config = config
        self.L1, self.L2, self.L3 = (
            config.irreps_in1,
            config.irreps_in2,
            config.irreps_out,
        )
        self.internal = None
        self.torch_op = torch_op
        self.idx_dtype = idx_dtype
        self.deterministic = deterministic

        self.conv_id = ConvolutionBase.next_conv_id
        ConvolutionBase.next_conv_id += 1

        if torch_op:
            global torch
            import torch

        self.workspace_ptr = 0
        self.workspace_size = 0

    def reorder_weights_from_e3nn(self, weights, has_batch_dim=True):
        r"""
        See :py:func:`oeq.TensorProduct.reorder_weights_from_e3nn`.
        """
        return weights

    def reorder_weights_to_e3nn(self, weights, has_batch_dim=True):
        r"""
        See :py:func:`oeq.TensorProduct.reorder_weights_to_e3nn`.
        """
        return weights

    @staticmethod
    def name():
        raise NotImplementedError()

    def benchmark_forward(
        self, num_warmup, num_iter, graph, prng_seed=12345, kernel_names=["forward"]
    ):
        direction = "forward"
        L1_in, L2_in, weights, L3_buffer = get_random_buffers_forward_conv(
            self.config, graph.node_count, graph.nnz, prng_seed
        )

        assert graph.rows.dtype == self.idx_dtype
        assert graph.cols.dtype == self.idx_dtype

        torch_L1_in = torch.tensor(L1_in, device="cuda")
        torch_L2_in = torch.tensor(L2_in, device="cuda")
        torch_weights = torch.tensor(weights, device="cuda")

        torch_rows = torch.tensor(graph.rows, device="cuda")
        torch_cols = torch.tensor(graph.cols, device="cuda")
        torch_transpose_perm = (
            torch.tensor(graph.transpose_perm, device="cuda")
            if self.deterministic
            else None
        )

        mode = "gpu_time" if self.torch_op else "torch_kernel_time"

        time_millis = benchmark(
            (
                lambda: self.forward(
                    torch_L1_in,
                    torch_L2_in,
                    torch_weights,
                    torch_rows,
                    torch_cols,
                    torch_transpose_perm,
                )
            ),
            num_warmup,
            num_iter,
            mode=mode,
            kernel_names=kernel_names,
        )

        ops_per_tp, data_per_tp, _ = flops_data_per_tp(self.config, direction)
        ops_per_tp += self.config.irreps_out.dim

        return self.calculate_bench_stats(
            direction,
            ops_per_tp,
            data_per_tp,
            time_millis,
            graph,
            num_warmup,
            num_iter,
            prng_seed,
        )

    def benchmark_backward(
        self, num_warmup, num_iter, graph, prng_seed=12345, kernel_names=["backward"]
    ):
        direction = "backward"
        in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = (
            get_random_buffers_backward_conv(
                self.config, graph.node_count, graph.nnz, prng_seed
            )
        )

        assert graph.rows.dtype == self.idx_dtype
        assert graph.cols.dtype == self.idx_dtype

        torch_L1_in = torch.tensor(in1, device="cuda", requires_grad=True)
        torch_L2_in = torch.tensor(in2, device="cuda", requires_grad=True)
        torch_weights = torch.tensor(weights, device="cuda", requires_grad=True)

        torch_rows = torch.tensor(graph.rows, device="cuda").detach()
        torch_cols = torch.tensor(graph.cols, device="cuda").detach()
        torch_transpose_perm = torch.tensor(graph.transpose_perm, device="cuda")

        fwd_args = [torch_L1_in, torch_L2_in, torch_weights, torch_rows, torch_cols]
        if self.deterministic:
            fwd_args.append(torch_transpose_perm)
        torch_out = self.forward(*fwd_args)
        torch_L3_grad = torch.tensor(out_grad, device="cuda")

        mode = "gpu_time" if self.torch_op else "torch_kernel_time"

        time_millis = benchmark(
            (
                lambda: torch_out.backward(
                    torch_L3_grad,
                    retain_graph=True,
                    inputs=[torch_L1_in, torch_L2_in, torch_weights],
                )
            ),
            num_warmup,
            num_iter,
            mode=mode,
            kernel_names=kernel_names,
        )

        ops_per_tp, data_per_tp, _ = flops_data_per_tp(self.config, direction)
        ops_per_tp += self.config.irreps_out.dim

        return self.calculate_bench_stats(
            direction,
            ops_per_tp,
            data_per_tp,
            time_millis,
            graph,
            num_warmup,
            num_iter,
            prng_seed,
        )

    def calculate_bench_stats(
        self,
        direction,
        ops_per_tp,
        data_per_tp,
        time_millis,
        graph,
        num_warmup,
        num_iter,
        prng_seed,
    ):
        throughputs_gflops = [
            float(el) for el in graph.nnz * ops_per_tp / (time_millis * 1e6)
        ]
        bandwidth_gbps = [
            float(el) for el in graph.nnz * data_per_tp / (time_millis * 1e6)
        ]
        time_millis = [float(el) for el in time_millis]

        result = {
            "direction": direction,
            "flops_per_tp": int(ops_per_tp),
            "data_per_tp": int(data_per_tp),
            "time_millis": list(time_millis),
            "throughputs_gflops": list(throughputs_gflops),
            "bandwidth_gbps": list(bandwidth_gbps),
            "L1": str(self.config.irreps_in1),
            "L2": str(self.config.irreps_in2),
            "L3": str(self.config.irreps_out),
            "graph_node_count": graph.node_count,
            "graph_adj_nnz": graph.nnz,
            "num_warmup": num_warmup,
            "num_iter": num_iter,
            "prng_seed": prng_seed,
        }

        logger.info(
            f"{bcolors.OKCYAN}Avg. Throughput: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(throughputs_gflops):.2f} ± {np.std(throughputs_gflops):.2f} GFLOPs{bcolors.ENDC}"
        )
        logger.info(
            f"{bcolors.OKCYAN}Avg. Bandwidth: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(bandwidth_gbps):.2f} ± {np.std(bandwidth_gbps):.2f} GBPs{bcolors.ENDC}"
        )
        return result



def scatter_add_wrapper(messages, rows, target_dim):
    L3_dim = messages.size(1)
    idx = rows.unsqueeze(1).expand(-1, L3_dim)
    out = messages.new_zeros((target_dim, L3_dim))
    return torch.scatter_add(
        input=out,
        dim=0,
        index=idx,
        src=messages,
    )
