import numpy as np
import numpy.linalg as la
from build.kernel_wrapper import *
from src.benchmark.random_buffer_utils import * 
from src.implementations.TensorProduct import *

from src.benchmark.logging_utils import getLogger, bcolors 
logger = getLogger()

def flops_data_per_tp(config, bytes_per_word, direction):
    '''
    Assumes all interactions are "uvu" for now

    Returns (flops_per_tp, data_per_tp, nnz)
    '''
    assert(not config.shared_weights)
    L1, L2, L3 = config.irreps_in1, config.irreps_in2, config.irreps_out
    ops_per_nz, words_per_tp = None, None
    if direction == "forward":
        ops_per_nz = 3
        words_per_tp = L1.dim + L2.dim + L3.dim + config.weight_numel 
    elif direction == "backward":
        ops_per_nz = 9
        words_per_tp = L1.dim + L2.dim + L3.dim + config.weight_numel \
                + L1.dim + L2.dim + config.weight_numel # Output gradients

    ops_per_tp = 0
    nnz = 0
    for (u, v, w, connection_mode, *others) in config.instructions:
        tensor = TensorProduct.load_cg_tensor(L1[u].ir.l, L2[v].ir.l, L3[w].ir.l)
        local_nnz = np.count_nonzero(tensor)
        nnz += local_nnz
        ops_per_tp += ops_per_nz * local_nnz * L1[u].mul * L2[v].mul # Assumes L3.mult(w) = L1.mult(u) * L2.mult(v)

        if connection_mode == "uvu":
            ops_per_tp += L3[w].mul * (2 * L3[w].ir.l + 1) 
        elif connection_mode == "uvw":
            ops_per_tp += L1[u].mul * L2[v].mul * L3[w].ir.dim * L3[w].mul

    return ops_per_tp, words_per_tp * bytes_per_word, nnz

class CoordGraph:
    def __init__(self, coords, rows, cols, name):
        '''
        Because graphs may change constantly, this class is designed
        to be as light as possible. A directed edge from node
        u to v is indicated by the presence of an index i such that
        rows[i] = u, rows[i] = v.
        '''
        assert(len(rows) == len(cols))
        self.nnz = len(rows) # Counts every nonzero in the adjacency matrix 
        self.node_count = coords.shape[0]
        self.rows = rows
        self.cols = cols
        self.coords = coords
        self.name = name

        self.cached_sp_graph = None # Cached scipy sparse matrix 

class Convolution:
    '''
    Inputs: L1 for input node features
            L2 for edge features
            L3 for output node features 
    '''
    def __init__(self, config):
        self.config = config 
        self.L1, self.L2, self.L3 = config.irreps_in1, config.irreps_in2, config.irreps_out
        self.internal = None

    @staticmethod
    def name():
        raise NotImplementedError()

    def forward_cpu(self, 
            L1_in, L2_in, weights, L3_out,
            graph, disable_tensor_op=False):

        L1_d, L2_d, weights_d = DeviceBuffer(L1_in), DeviceBuffer(L2_in), DeviceBuffer(weights)
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
            disable_tensor_op)

        L3_d.copy_to_host()


    def backward_cpu(self, 
            L1_in, L2_in, weights, L3_grad,
            graph, disable_tensor_op=False):
        '''
        We break from convention here by allocating and returning
        the appropriate buffers. 
        '''
        L1_grad = np.zeros_like(L1_in)
        L2_grad = np.zeros_like(L2_in)
        weights_grad = np.zeros_like(weights)

        L1_d = DeviceBuffer(L1_in)
        L2_d = DeviceBuffer(L2_in)
        weights_d = DeviceBuffer(weights)
        L3_d = DeviceBuffer(L3_grad)
        rows_d = DeviceBuffer(graph.rows)
        cols_d = DeviceBuffer(graph.cols)
        
        L1_grad_d = DeviceBuffer(L1_grad)
        L2_grad_d = DeviceBuffer(L2_grad)
        weights_grad_d = DeviceBuffer(weights_grad)

        self.internal.backward_rawptrs(
            L1_d.data_ptr(), L1_grad_d.data_ptr(),
            L2_d.data_ptr(), L2_grad_d.data_ptr(),
            weights_d.data_ptr(), weights_grad_d.data_ptr(),
            L3_d.data_ptr(),
            rows_d.data_ptr(), cols_d.data_ptr(),
            graph.nnz, graph.node_count,
            disable_tensor_op)

        L1_grad_d.copy_to_host()
        L2_grad_d.copy_to_host()
        weights_grad_d.copy_to_host()

        return L1_grad, L2_grad, weights_grad

    def test_correctness(self, L1_in, L2_in, weights, L3_out_comp, graph, conv_reference_impl, disable_tensor_op):
        L1, L2, L3 = self.L1, self.L2, self.L3

        ground_truth = np.zeros((graph.node_count, L3.dim), dtype=np.float32)
        conv_reference = conv_reference_impl(self.config)

        if disable_tensor_op:
            logger.warning(f"{bcolors.WARNING}Tensor product disabled within convolution, performing SpMM.{bcolors.ENDC}")

        logger.info(f"Starting reference convolution {bcolors.OKCYAN}{conv_reference.name()}{bcolors.ENDC}.")
        conv_reference.forward_cpu(L1_in, L2_in, weights, ground_truth, graph, disable_tensor_op) 
        logger.info("Finished reference convolution.")

        thresh = 5e-6 # AtomicAdd nondeterminism may require higher threshold 
        result = {
            "disable_tensor_op": disable_tensor_op,
            "shape_match": False,
            "diff_Linf_norm": np.inf,
            "thresh": thresh, # Above floating point interval machine epsilon 
            "pass": False
        }

        if L3_out_comp.shape != ground_truth.shape:
            result["shape_match"] = False
            logger.error(f"{bcolors.FAIL}Ground truth shape does not match input! {L3_out_comp.shape=}, {ground_truth.shape=} {bcolors.ENDC}")
        else:
            result["shape_match"] = True 
            diff_Linf_norm = float(la.norm((ground_truth - L3_out_comp).flatten(), ord=np.inf))
            result["diff_Linf_norm"] = diff_Linf_norm 
            result["pass"] = bool(diff_Linf_norm < thresh) 

            if result["pass"]:
                logger.info(f"{bcolors.OKGREEN}Convolution correctness check pass, {diff_Linf_norm=:.2g}, {thresh=:.2g}. {bcolors.ENDC}")
            else:
                logger.error(f"{bcolors.FAIL}Convolution correctness check fail! {diff_Linf_norm=:.2g}, {thresh=:.2g} {bcolors.ENDC}")

        return result, ground_truth

    def benchmark_forward(self, 
            num_warmup, 
            num_iter, 
            graph, 
            disable_tensor_op, 
            prng_seed=12345):
        '''
        This function only works for scalar L-values right now, need to change
        to handle any multiplicity.
        '''
        L1_in, L2_in, weights, L3_buffer = get_random_buffers_forward_conv(self.config, graph.node_count, graph.nnz, prng_seed)

        L1_d, L2_d, weights_d = DeviceBuffer(L1_in), DeviceBuffer(L2_in), DeviceBuffer(weights)
        L3_d = DeviceBuffer(L3_buffer)
        rows_d = DeviceBuffer(graph.rows)
        cols_d = DeviceBuffer(graph.cols)

        time_millis = np.zeros(num_iter, dtype=np.float32)
        timer = GPUTimer()

        for i in range(num_warmup):
            self.internal.exec_conv_rawptrs(
                L1_d.data_ptr(), L2_d.data_ptr(), weights_d.data_ptr(), L3_d.data_ptr(),
                rows_d.data_ptr(), cols_d.data_ptr(), graph.nnz, graph.node_count,
                disable_tensor_op)

        for i in range(num_iter):
            timer.start()
            self.internal.exec_conv_rawptrs(
                L1_d.data_ptr(), L2_d.data_ptr(), weights_d.data_ptr(), L3_d.data_ptr(),
                rows_d.data_ptr(), cols_d.data_ptr(), graph.nnz, graph.node_count,
                disable_tensor_op)
            time_millis[i] = timer.stop_clock_get_elapsed() 

        ops_per_tp, data_per_tp, nnz = flops_data_per_tp(self.config, 4, "forward")
        if disable_tensor_op:
            ops_per_tp = 2 * self.config.irreps_out.dim
        else:
            ops_per_tp += self.config.irreps_out.dim # Output accumulation... should check this 

        throughputs_gflops = [float(el) for el in graph.nnz * ops_per_tp / (time_millis * 1e6)]

        # Rough calculation of bandwidth assumes output is touched only once, but input rows are read as many times as nnz 
        bandwidth_gbps = [float(el) for el in graph.nnz * data_per_tp / (time_millis * 1e6)]
        time_millis = [float(el) for el in time_millis] 

        result = {
            "direction": "forward",
            "total_cg_nnz": nnz,
            "flops_per_tp": ops_per_tp,
            "data_per_tp": data_per_tp,

            "disable_tensor_op": disable_tensor_op,
            "L1": str(self.config.irreps_in1),
            "L2": str(self.config.irreps_in2), 
            "L3": str(self.config.irreps_out),
            "graph_node_count": graph.node_count,
            "graph_adj_nnz": graph.nnz,
            "num_warmup": num_warmup,
            "num_iter": num_iter,
            "prng_seed": prng_seed,
            "time_millis": time_millis,
            "throughputs_gflops": throughputs_gflops,
            "bandwidth_gbps": bandwidth_gbps
        }

        disable_op_str = ""
        if disable_tensor_op:
            disable_op_str = " (Tensor Op Disabled)"

        logger.info(f"{bcolors.OKCYAN}Avg. Throughput{disable_op_str}: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(throughputs_gflops):.2f} ± {np.std(throughputs_gflops):.2f} GFLOPs{bcolors.ENDC}")
        logger.info(f"{bcolors.OKCYAN}Avg. Bandwidth{disable_op_str}: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(bandwidth_gbps):.2f} ± {np.std(bandwidth_gbps):.2f} GBPs{bcolors.ENDC}")
        return result