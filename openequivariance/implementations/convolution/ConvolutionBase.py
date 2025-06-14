import copy
import numpy as np
from openequivariance.extlib import DeviceBuffer, GPUTimer
from openequivariance.benchmark.random_buffer_utils import (
    get_random_buffers_forward_conv,
    get_random_buffers_backward_conv,
)
from openequivariance.implementations.TensorProductBase import TensorProductBase

from openequivariance.benchmark.logging_utils import getLogger, bcolors
from openequivariance.benchmark.correctness_utils import check_similiarity

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
        tensor = TensorProductBase.load_cg_tensor(L1[u].ir.l, L2[v].ir.l, L3[w].ir.l)
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

    def __init__(self, config, idx_dtype, torch_op=False, deterministic=False):
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

    def allocate_workspace(self, size_bytes):
        self.workspace_size = size_bytes
        if self.torch_op:
            self.workspace_buffer = torch.zeros(
                size_bytes, dtype=torch.uint8, device="cuda"
            )
        else:
            self.workspace_buffer = DeviceBuffer(size_bytes)
        self.workspace_ptr = self.workspace_buffer.data_ptr()
        logger.info(f"Convolution requires {size_bytes // 1000000}MB of workspace.")

    @staticmethod
    def name():
        raise NotImplementedError()

    def forward_cpu(self, L1_in, L2_in, weights, L3_out, graph):
        assert graph.rows.dtype == self.idx_dtype
        assert graph.cols.dtype == self.idx_dtype

        weights_chunked = np.zeros_like(weights)
        if self.reorder_weights_e3nn_to_oeq is not None:
            self.reorder_weights_e3nn_to_oeq(
                weights, weights_chunked, not self.config.shared_weights
            )
        else:
            weights_chunked = weights

        L1_d, L2_d, weights_d = (
            DeviceBuffer(L1_in),
            DeviceBuffer(L2_in),
            DeviceBuffer(weights_chunked),
        )
        L3_d = DeviceBuffer(L3_out)

        rows_d = DeviceBuffer(graph.rows)
        cols_d = DeviceBuffer(graph.cols)

        self.internal.exec_conv_rawptrs(
            L1_d.data_ptr(),
            L2_d.data_ptr(),
            weights_d.data_ptr(),
            L3_d.data_ptr(),
            rows_d.data_ptr(),
            cols_d.data_ptr(),
            graph.nnz,
            graph.node_count,
            self.workspace_ptr,
        )

        L3_d.copy_to_host()

    def backward_cpu(
        self, L1_in, L1_grad, L2_in, L2_grad, weights, weights_grad, L3_grad, graph
    ):
        assert graph.rows.dtype == self.idx_dtype
        assert graph.cols.dtype == self.idx_dtype

        weights_chunked = np.zeros_like(weights)
        if self.reorder_weights_e3nn_to_oeq is not None:
            self.reorder_weights_e3nn_to_oeq(
                weights, weights_chunked, not self.config.shared_weights
            )
        else:
            weights_chunked = weights

        L1_d = DeviceBuffer(L1_in)
        L2_d = DeviceBuffer(L2_in)
        weights_d = DeviceBuffer(weights_chunked)
        L3_d = DeviceBuffer(L3_grad)
        rows_d = DeviceBuffer(graph.rows)
        cols_d = DeviceBuffer(graph.cols)

        L1_grad_d = DeviceBuffer(L1_grad)
        L2_grad_d = DeviceBuffer(L2_grad)
        weights_grad_d = DeviceBuffer(weights_grad)

        transpose_perm_d = None
        transpose_perm_ptr = 0
        if self.deterministic:
            transpose_perm_d = DeviceBuffer(graph.transpose_perm)
            transpose_perm_ptr = transpose_perm_d.data_ptr()

        self.internal.backward_rawptrs(
            L1_d.data_ptr(),
            L1_grad_d.data_ptr(),
            L2_d.data_ptr(),
            L2_grad_d.data_ptr(),
            weights_d.data_ptr(),
            weights_grad_d.data_ptr(),
            L3_d.data_ptr(),
            rows_d.data_ptr(),
            cols_d.data_ptr(),
            graph.nnz,
            graph.node_count,
            self.workspace_ptr,
            transpose_perm_ptr,
        )

        L1_grad_d.copy_to_host()
        L2_grad_d.copy_to_host()
        weights_grad_d.copy_to_host()

        if self.reorder_weights_oeq_to_e3nn is not None:
            weights_grad_copy = weights_grad.copy()
            self.reorder_weights_oeq_to_e3nn(
                weights_grad_copy, weights_grad, not self.config.shared_weights
            )

        return L1_grad, L2_grad, weights_grad

    def test_correctness_forward(
        self,
        graph,
        thresh,
        prng_seed,
        reference_implementation=None,
        check_reproducible=True,
        high_precision_ref=False,
    ):
        if reference_implementation is None:
            from openequivariance.implementations.convolution.E3NNConv import E3NNConv

            reference_implementation = E3NNConv

        result = {"thresh": thresh}

        in1, in2, weights, out = get_random_buffers_forward_conv(
            self.config, graph.node_count, graph.nnz, prng_seed
        )
        ref_in1, ref_in2, ref_weights, ref_out = [
            buf.copy() for buf in [in1, in2, weights, out]
        ]

        reference_config = self.config
        if high_precision_ref:
            reference_config = copy.deepcopy(self.config)
            reference_config.irrep_dtype = np.float64
            reference_config.weight_dtype = np.float64
            ref_in1, ref_in2, ref_weights, ref_out = [
                np.array(el, dtype=np.float64)
                for el in [ref_in1, ref_in2, ref_weights, ref_out]
            ]

        args = {
            "L1_in": ref_in1,
            "L2_in": ref_in2,
            "weights": ref_weights,
            "rows": graph.rows,
            "cols": graph.cols,
        }

        ref_tp = reference_implementation(reference_config)
        if ref_tp.deterministic:
            args["transpose_perm"] = graph.transpose_perm

        for key in args:
            args[key] = torch.tensor(args[key], device="cuda")

        ref_out[:] = ref_tp.forward(**args).cpu().numpy()

        test_out = out.copy()
        self.forward_cpu(
            L1_in=in1.copy(),
            L2_in=in2.copy(),
            weights=weights.copy(),
            L3_out=test_out,
            graph=graph,
        )

        for name, to_check, ground_truth in [("output", ref_out, test_out)]:
            result[name] = check_similiarity(name, to_check, ground_truth, thresh)

        if check_reproducible:
            num_trials = 5
            for name in ["output"]:
                result[name]["num_reproducibility_trials"] = num_trials
                result[name]["bitwise_reproducible"] = True

            for i in range(num_trials):
                repeated_run = out.copy()
                self.forward_cpu(
                    L1_in=in1.copy(),
                    L2_in=in2.copy(),
                    weights=weights.copy(),
                    L3_out=repeated_run,
                    graph=graph,
                )

                for name, to_check, ground_truth in [
                    ("output", repeated_run, test_out)
                ]:
                    result[name]["bitwise_reproducible"] = bool(
                        result[name]["bitwise_reproducible"]
                        and np.all(repeated_run == test_out)
                    )

        return result

    def benchmark_forward(self, num_warmup, num_iter, graph, prng_seed=12345):
        direction = "forward"
        L1_in, L2_in, weights, L3_buffer = get_random_buffers_forward_conv(
            self.config, graph.node_count, graph.nnz, prng_seed
        )

        assert graph.rows.dtype == self.idx_dtype
        assert graph.cols.dtype == self.idx_dtype

        time_millis = np.zeros(num_iter, dtype=np.float32)
        timer = GPUTimer()

        if self.torch_op:
            torch_L1_in = torch.tensor(L1_in, device="cuda")
            torch_L2_in = torch.tensor(L2_in, device="cuda")
            torch_weights = torch.tensor(weights, device="cuda")

            torch_rows = torch.tensor(graph.rows, device="cuda")
            torch_cols = torch.tensor(graph.cols, device="cuda")
            torch_transpose_perm = torch.tensor(graph.transpose_perm, device="cuda")

            if not self.deterministic:
                for i in range(num_warmup):
                    self.forward(
                        torch_L1_in, torch_L2_in, torch_weights, torch_rows, torch_cols
                    )

                for i in range(num_iter):
                    timer.clear_L2_cache()
                    timer.start()
                    self.forward(
                        torch_L1_in, torch_L2_in, torch_weights, torch_rows, torch_cols
                    )
                    time_millis[i] = timer.stop_clock_get_elapsed()
            else:
                for i in range(num_warmup):
                    self.forward(
                        torch_L1_in,
                        torch_L2_in,
                        torch_weights,
                        torch_rows,
                        torch_cols,
                        torch_transpose_perm,
                    )

                for i in range(num_iter):
                    timer.clear_L2_cache()
                    timer.start()
                    self.forward(
                        torch_L1_in,
                        torch_L2_in,
                        torch_weights,
                        torch_rows,
                        torch_cols,
                        torch_transpose_perm,
                    )
                    time_millis[i] = timer.stop_clock_get_elapsed()

        elif not self.torch_op:
            L1_d, L2_d, weights_d = (
                DeviceBuffer(L1_in),
                DeviceBuffer(L2_in),
                DeviceBuffer(weights),
            )
            L3_d = DeviceBuffer(L3_buffer)
            rows_d = DeviceBuffer(graph.rows)
            cols_d = DeviceBuffer(graph.cols)

            for i in range(num_warmup):
                self.internal.exec_conv_rawptrs(
                    L1_d.data_ptr(),
                    L2_d.data_ptr(),
                    weights_d.data_ptr(),
                    L3_d.data_ptr(),
                    rows_d.data_ptr(),
                    cols_d.data_ptr(),
                    graph.nnz,
                    graph.node_count,
                    self.workspace_ptr,
                )

            for i in range(num_iter):
                timer.clear_L2_cache()
                timer.start()
                self.internal.exec_conv_rawptrs(
                    L1_d.data_ptr(),
                    L2_d.data_ptr(),
                    weights_d.data_ptr(),
                    L3_d.data_ptr(),
                    rows_d.data_ptr(),
                    cols_d.data_ptr(),
                    graph.nnz,
                    graph.node_count,
                    self.workspace_ptr,
                )
                time_millis[i] = timer.stop_clock_get_elapsed()

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

    def benchmark_backward(self, num_warmup, num_iter, graph, prng_seed=12345):
        direction = "backward"
        in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = (
            get_random_buffers_backward_conv(
                self.config, graph.node_count, graph.nnz, prng_seed
            )
        )

        assert graph.rows.dtype == self.idx_dtype
        assert graph.cols.dtype == self.idx_dtype

        time_millis = np.zeros(num_iter, dtype=np.float32)
        timer = GPUTimer()

        if self.torch_op:
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

            for i in range(num_warmup):
                torch_out.backward(
                    torch_L3_grad,
                    retain_graph=True,
                    inputs=[torch_L1_in, torch_L2_in, torch_weights],
                )

            for i in range(num_iter):
                torch_L1_in.grad.zero_()
                torch_L2_in.grad.zero_()
                torch_weights.grad.zero_()

                timer.clear_L2_cache()
                timer.start()
                torch_out.backward(
                    torch_L3_grad,
                    retain_graph=True,
                    inputs=[torch_L1_in, torch_L2_in, torch_weights],
                )
                time_millis[i] = timer.stop_clock_get_elapsed()

        elif not self.torch_op:
            L1_d = DeviceBuffer(in1)
            L2_d = DeviceBuffer(in2)
            weights_d = DeviceBuffer(weights)
            L3_d = DeviceBuffer(out_grad)
            rows_d = DeviceBuffer(graph.rows)
            cols_d = DeviceBuffer(graph.cols)

            L1_grad_d = DeviceBuffer(in1_grad)
            L2_grad_d = DeviceBuffer(in2_grad)
            weights_grad_d = DeviceBuffer(weights_grad)

            transpose_perm_d = None
            transpose_perm_ptr = 0
            if self.deterministic:
                transpose_perm_d = DeviceBuffer(graph.transpose_perm)
                transpose_perm_ptr = transpose_perm_d.data_ptr()

            for i in range(num_warmup):
                self.internal.backward_rawptrs(
                    L1_d.data_ptr(),
                    L1_grad_d.data_ptr(),
                    L2_d.data_ptr(),
                    L2_grad_d.data_ptr(),
                    weights_d.data_ptr(),
                    weights_grad_d.data_ptr(),
                    L3_d.data_ptr(),
                    rows_d.data_ptr(),
                    cols_d.data_ptr(),
                    graph.nnz,
                    graph.node_count,
                    self.workspace_ptr,
                    transpose_perm_ptr,
                )

            for i in range(num_iter):
                timer.clear_L2_cache()
                timer.start()
                self.internal.backward_rawptrs(
                    L1_d.data_ptr(),
                    L1_grad_d.data_ptr(),
                    L2_d.data_ptr(),
                    L2_grad_d.data_ptr(),
                    weights_d.data_ptr(),
                    weights_grad_d.data_ptr(),
                    L3_d.data_ptr(),
                    rows_d.data_ptr(),
                    cols_d.data_ptr(),
                    graph.nnz,
                    graph.node_count,
                    self.workspace_ptr,
                    transpose_perm_ptr,
                )
                time_millis[i] = timer.stop_clock_get_elapsed()

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
            "flops_per_tp": ops_per_tp,
            "data_per_tp": data_per_tp,
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

    def test_correctness_backward(
        self,
        graph,
        thresh,
        prng_seed,
        reference_implementation=None,
        high_precision_ref=False,
    ):
        if reference_implementation is None:
            from openequivariance.implementations.convolution.E3NNConv import E3NNConv

            reference_implementation = E3NNConv

        result = {"thresh": thresh}

        buffers = get_random_buffers_backward_conv(
            self.config, graph.node_count, graph.nnz, prng_seed
        )
        reference_buffers = [buf.copy() for buf in buffers]
        reference_problem = self.config

        if high_precision_ref:
            reference_problem = copy.deepcopy(self.config)
            reference_problem.irrep_dtype = np.float64
            reference_problem.weight_dtype = np.float64
            reference_buffers = [
                np.array(el, dtype=np.float64) for el in reference_buffers
            ]

        ref_tp = reference_implementation(reference_problem)
        in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = buffers
        (
            ref_in1,
            ref_in2,
            ref_out_grad,
            ref_weights,
            ref_weights_grad,
            ref_in1_grad,
            ref_in2_grad,
        ) = reference_buffers

        ref_tp.backward_cpu(
            L1_in=ref_in1,
            L1_grad=ref_in1_grad,
            L2_in=ref_in2,
            L2_grad=ref_in2_grad,
            L3_grad=ref_out_grad,
            weights=ref_weights,
            weights_grad=ref_weights_grad,
            graph=graph,
        )

        # run test version
        test_weights_grad = weights_grad.copy()
        test_in1_grad = in1_grad.copy()
        test_in2_grad = in2_grad.copy()

        self.backward_cpu(
            L1_in=in1.copy(),
            L1_grad=test_in1_grad,
            L2_in=in2.copy(),
            L2_grad=test_in2_grad,
            L3_grad=out_grad.copy(),
            weights=weights.copy(),
            weights_grad=test_weights_grad,
            graph=graph,
        )

        for name, to_check, ground_truth, threshold in [
            ("weight_grad", test_weights_grad, ref_weights_grad, thresh),
            ("in1_grad", test_in1_grad, ref_in1_grad, thresh),
            ("in2_grad", test_in2_grad, ref_in2_grad, thresh),
        ]:
            result[name] = check_similiarity(name, to_check, ground_truth, threshold)

        return result

    def test_correctness_double_backward(
        self,
        graph,
        thresh,
        prng_seed,
        reference_implementation=None,
        high_precision_ref=False,
    ):
        global torch
        import torch

        assert self.torch_op
        buffers = get_random_buffers_backward_conv(
            self.config, graph.node_count, graph.nnz, prng_seed
        )

        rng = np.random.default_rng(seed=prng_seed * 2)
        dummy_grad_value = rng.standard_normal(1)[0]

        if reference_implementation is None:
            from openequivariance.implementations.convolution.E3NNConv import E3NNConv

            reference_implementation = E3NNConv

        reference_problem = self.config
        if high_precision_ref:
            reference_problem = copy.deepcopy(self.config)
            reference_problem.irrep_dtype = np.float64
            reference_problem.weight_dtype = np.float64

        reference_tp = reference_implementation(reference_problem, torch_op=True)

        result = {"thresh": thresh}
        tensors = []
        for i, tp in enumerate([self, reference_tp]):
            in1, in2, out_grad, weights, _, _, _ = [buf.copy() for buf in buffers]

            if i == 1 and high_precision_ref:
                in1, in2, out_grad, weights, _, _, _ = [
                    np.array(el, dtype=np.float64) for el in buffers
                ]

            in1_torch = torch.tensor(in1, device="cuda", requires_grad=True)
            in2_torch = torch.tensor(in2, device="cuda", requires_grad=True)

            weights_reordered = np.zeros_like(weights)
            if i == 0 and self.reorder_weights_e3nn_to_oeq is not None:
                self.reorder_weights_e3nn_to_oeq(
                    weights, weights_reordered, not self.config.shared_weights
                )
            else:
                weights_reordered[:] = weights
            weights_torch = torch.tensor(
                weights_reordered, device="cuda", requires_grad=True
            )

            torch_rows = torch.tensor(graph.rows, device="cuda")
            torch_cols = torch.tensor(graph.cols, device="cuda")
            torch_transpose_perm = torch.tensor(graph.transpose_perm, device="cuda")

            fwd_args = [in1_torch, in2_torch, weights_torch, torch_rows, torch_cols]
            if tp.deterministic:
                fwd_args.append(torch_transpose_perm)

            out_torch = tp.forward(*fwd_args)
            out_grad_torch = torch.tensor(out_grad, device="cuda", requires_grad=True)

            in1_grad, in2_grad, w_grad = torch.autograd.grad(
                outputs=[out_torch],
                inputs=[in1_torch, in2_torch, weights_torch],
                grad_outputs=[out_grad_torch],
                create_graph=True,
            )

            dummy = torch.norm(in1_grad) + torch.norm(in2_grad) + torch.norm(w_grad)
            dummy_grad = torch.tensor(
                float(dummy_grad_value), device="cuda", requires_grad=True
            )
            dummy.backward(
                dummy_grad, inputs=[out_grad_torch, in1_torch, in2_torch, weights_torch]
            )

            weights_grad = weights_torch.grad.detach().cpu().numpy()
            if i == 0 and self.reorder_weights_oeq_to_e3nn is not None:
                weights_grad_copy = weights_grad.copy()
                self.reorder_weights_oeq_to_e3nn(
                    weights_grad_copy, weights_grad, not self.config.shared_weights
                )

            tensors.append(
                (
                    out_grad_torch.grad.detach().cpu().numpy().copy(),
                    in1_torch.grad.detach().cpu().numpy().copy(),
                    in2_torch.grad.detach().cpu().numpy().copy(),
                    weights_grad.copy(),
                )
            )

        for name, to_check, ground_truth in [
            ("output_grad", tensors[0][0], tensors[1][0]),
            ("in1_grad", tensors[0][1], tensors[1][1]),
            ("in2_grad", tensors[0][2], tensors[1][2]),
            ("weights_grad", tensors[0][3], tensors[1][3]),
        ]:
            result[name] = check_similiarity(name, to_check, ground_truth, thresh)

        return result
