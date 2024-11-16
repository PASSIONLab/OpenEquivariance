import numpy as np

from src.benchmark.e3nn_lite_utils import calc_weight_offsets
from src.benchmark.e3nn_lite_utils import Irrep, _MulIr, Irreps, TPProblem, Instruction
from src.implementations.TensorProduct import TensorProduct, GPUInfo
from src.benchmark.logging_utils import getLogger, bcolors 
from jinja2 import Environment, FileSystemLoader

from build.kernel_wrapper import KernelLaunchConfig, JITTPImpl

logger = getLogger()

def raise_helper(msg):
    raise Exception(msg)

def divide(numerator, denominator):
    return numerator // denominator 

def sizeof(dtype):
    if dtype in ["float", "int", "unsigned int"]:
        return 4
    else:
        raise Exception("Provided undefined datatype to sizeof!")

class MultiplicityOuterProductTP(TensorProduct):
    def __init__(self, config : TPProblem, torch_op : bool = False):
        super().__init__(config, torch_op)

        for ins in config.instructions: # type : Instruction
            assert isinstance(ins, Instruction)
            assert ins.connection_mode == 'uvw'
            assert ins.path_shape[0] <= 32
            assert ins.path_shape[1] <= 32
            assert ins.path_shape[2] <= 32

        irreps_in1 = config.irreps_in1
        irreps_in2 = config.irreps_in2
        irreps_out = config.irreps_out 

        # ==================================================================================

        env = Environment(loader=FileSystemLoader("src/templates"), extensions=['jinja2.ext.do'])
        env.globals['raise'] = raise_helper 
        env.globals['divide'] = divide 
        env.globals['sizeof'] = sizeof
        env.globals['range'] = range
        env.globals['enumerate'] = enumerate 
        env.globals['len'] = len
        main_template = env.get_template("subkernel_per_interaction_multirep.cuh")
        # forward_subkernel_template = env.get_template("subkernel_forward_thread.cu.jinja2")
        # backward_subkernel_template = env.get_template("subkernel_backward_thread.cu.jinja2")
        
        # =====================================================================
        # Updated to work with TensorProductProblem
    
        class RepData:
            def __init__(self, irreps : Irreps):
                assert isinstance(irreps, Irreps)
                self.rep_len = irreps.dim
                self.irrep_lengths = [mul_irrep.ir.dim for mul_irrep in irreps]
                self.mults = [mul_irrep.mul for mul_irrep in irreps]

                offset = 0
                self.offsets = []
                for mul_irrep in irreps: 
                    self.offsets.append(offset)
                    offset += mul_irrep.dim

        # =====================================================================
        # Strictly Copied from Loop Unroll TP

        class CGTensor:
            def __init__(self, l1, l2, l3):
                tensor = load_cg_tensor(l1, l2, l3)
                coord1, coord2, coord3 = [arr.astype(np.int32).copy() for arr in np.nonzero(tensor)]
                float_values = tensor[np.nonzero(tensor)].astype(np.float32).copy()
                values = [str(float.hex(float(val))) + "f" for val in float_values]

                self.tuples = [(coord1[i], coord2[i], coord3[i], values[i]) for i in range(len(values))]
              # self.tuples.sort(key=lambda tup: (tup[1], tup[0], tup[2]))
                self.nnz = len(values)

        # =====================================================================
        # FORWARD MEMORY ANALYSIS 
        forward_shared_memory_per_batch_element = (irreps_in1.dim + irreps_in2.dim + irreps_out.dim) * sizeof("float")
        forward_batch_elements_per_SM = 1 # HARDCODED NONSENSE 
        # IT WAS GOING TO BE THIS  
        #  # GPUInfo.max_smem // (GPUInfo.A100_SMS * forward_shared_memory_per_batch_element)
        forward_thread_blocks_per_SM = 1 # HARDCODED NONSENSE 
        forward_threads_per_batch_element = 32 # HARDCODED NONSENSE
        forward_threads_per_thread_block = forward_threads_per_batch_element * forward_batch_elements_per_SM

        # =====================================================================

        forward_launch_config = KernelLaunchConfig()
        forward_launch_config.num_blocks = GPUInfo.A100_SMS * forward_thread_blocks_per_SM
        forward_launch_config.num_threads = forward_threads_per_thread_block

        # IMPORTANT! 
        smem_gemm_n_max = 1
        smem_gemm_L3_scratch = smem_gemm_n_max * max(RepData(config.irreps_out).irrep_lengths) # this has space for the largest output size * 32
        smem_gemm_weights_scratch = 32 * smem_gemm_n_max

        smem_gemm_info = {
            'n_max' : smem_gemm_n_max,
            'n_to_try' : 1,
            'L3_scratch_elems' : smem_gemm_L3_scratch,
            'weight_scratch_elems' : smem_gemm_weights_scratch,
        }
        # END OF IMPORTANT

        forward_launch_config.smem = (
            (irreps_in1.dim + irreps_in2.dim + irreps_out.dim + smem_gemm_L3_scratch + smem_gemm_weights_scratch) 
            * sizeof("float") 
            * forward_launch_config.num_threads // forward_launch_config.warp_size
            ) 

        logger.info(f"Forward pass needs {forward_launch_config.smem} bytes of shared memory.")

        if forward_launch_config.smem > GPUInfo.max_smem:
            raise Exception(f"Error, requested shared memory {forward_launch_config.smem}B hits or exceeds maximum, {GPUInfo.max_smem}B !")
        
        # =====================================================================

        backward_launch_config = KernelLaunchConfig()
        backward_launch_config.num_blocks = GPUInfo.A100_SMS * 4
        backward_launch_config.num_threads = 192
        backward_launch_config.smem = (2 * irreps_in1.dim + 2 * irreps_in2.dim + 2 * + irreps_out.dim)  * sizeof("float") * backward_launch_config.num_threads // backward_launch_config.warp_size 
        logger.info(f"Backward pass needs {backward_launch_config.smem} bytes of shared memory.")

        if backward_launch_config.smem > GPUInfo.max_smem:
            raise Exception(f"Error, requested shared memory {backward_launch_config.smem}B hits or exceeds maximum, {GPUInfo.max_smem}B !")

        # =====================================================================     

        self.forward_config = forward_launch_config
        self.backward_config = backward_launch_config 
        load_cg_tensor = self.load_cg_tensor

        # =====================================================================
        # weights_offsets
        weight_offsets = calc_weight_offsets(config)
        assert isinstance(weight_offsets, list)
        assert len(weight_offsets) == len(list(config.instructions))

        # =====================================================================
        # tranform "e3nn instructions" into "interactions"
        instructions : list[Instruction] = config.instructions 
        interactions = []
        for ins in instructions:
            u = ins.i_in1
            v = ins.i_in2
            w = ins.i_out
            interaction = (u, v, w, CGTensor(irreps_in1[u].ir.l, irreps_in2[v].ir.l, irreps_out[w].ir.l))
            interactions.append(interaction)
        # interactions.sort(key=lambda x: (x[2], x[0], x[1]))


        assert len(interactions) != 0

        # =====================================================================
        # Strictly Copied from Loop Unroll TP
        kernel_text = main_template.render(
            L1=RepData(config.irreps_in1), 
            L2=RepData(config.irreps_in2), 
            L3=RepData(config.irreps_out),
            weight_offsets=weight_offsets,
            instructions=instructions,
            interactions=interactions,
            smem_gemm_info=smem_gemm_info,
            forward_config=forward_launch_config,
            backward_config=backward_launch_config
        )

        self.jit_kernel = kernel_text
        
        logger.debug(kernel_text)

        logger.info("Starting NVRTC")
        self.internal = JITTPImpl(self.jit_kernel, self.forward_config, self.backward_config)
        logger.info("Kernel compiled!")

    def forward_cpu(
        self, 
        L1_in: np.ndarray, 
        L2_in: np.ndarray, 
        L3_out: np.ndarray, 
        weights: np.ndarray
        ) -> None:
        '''
        All state initialization for the internal class occurs inside the
        constructor. 
        '''
        self.internal.exec_tensor_product_cpu(L1_in, L2_in, L3_out, weights)
        


    def backward_cpu(self, L1_in, L2_in, L3_grad, weights):
        return NotImplementedError("This doesn't begin to work")
