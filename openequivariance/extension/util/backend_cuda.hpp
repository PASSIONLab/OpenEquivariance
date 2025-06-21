#pragma once

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <vector>

using namespace std;
using Stream = cudaStream_t;

#define NVRTC_SAFE_CALL(x)                                      \
do {                                                            \
   nvrtcResult result = x;                                      \
   if (result != NVRTC_SUCCESS) {                               \
      std::cerr << "\nerror: " #x " failed with error "         \
               << nvrtcGetErrorString(result) << '\n';          \
      exit(1);                                                  \
   }                                                            \
} while(0)

#define CUDA_SAFE_CALL(x)                                     \
do {                                                            \
   CUresult result = x;                                         \
   if (result != CUDA_SUCCESS) {                                \
      const char *msg;                                          \
      cuGetErrorName(result, &msg);                             \
      std::cerr << "\nerror: " #x " failed with error "         \
               << msg << '\n';                                  \
      exit(1);                                                  \
   }                                                            \
} while(0)

#define CUDA_ERRCHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class CUDA_Allocator {
public:
    static void* gpu_alloc (size_t size) {
        void* ptr;
        CUDA_ERRCHK( cudaMalloc((void**) &ptr, size ))
        return ptr;
    }

    static void gpu_free (void* ptr) {
        CUDA_ERRCHK( cudaFree(ptr))
    }

    static void copy_host_to_device (void* host, void* device, size_t size) {
        CUDA_ERRCHK( cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
    }

    static void copy_device_to_host (void* host, void* device, size_t size) {
        CUDA_ERRCHK( cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    }
};

class GPUTimer {
    cudaEvent_t start_evt, stop_evt;

public:
    GPUTimer() {  
        cudaEventCreate(&start_evt);
        cudaEventCreate(&stop_evt);
    }

    void start() {
        cudaEventRecord(start_evt);
    }

    float stop_clock_get_elapsed() {
        float time_millis;
        cudaEventRecord(stop_evt);
        cudaEventSynchronize(stop_evt);
        cudaEventElapsedTime(&time_millis, start_evt, stop_evt);
        return time_millis; 
    }

    void clear_L2_cache() {
        size_t element_count = 25000000;

        int* ptr = (int*) (CUDA_Allocator::gpu_alloc(element_count * sizeof(int)));
        CUDA_ERRCHK(cudaMemset(ptr, 42, element_count * sizeof(int)))
        CUDA_Allocator::gpu_free(ptr);
        cudaDeviceSynchronize();
    }
    
    ~GPUTimer() {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
    }
};

class __attribute__((visibility("default"))) DeviceProp {
public:
    std::string name; 
    int warpsize;
    int major, minor;
    int multiprocessorCount;
    int maxSharedMemPerBlock;
    int maxSharedMemoryPerMultiprocessor; 

    DeviceProp(int device_id) {
        cudaDeviceProp prop; 
        cudaGetDeviceProperties(&prop, device_id);
        name = std::string(prop.name);
        CUDA_ERRCHK(cudaDeviceGetAttribute(&maxSharedMemoryPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_id));
        CUDA_ERRCHK(cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
        CUDA_ERRCHK(cudaDeviceGetAttribute(&warpsize, cudaDevAttrWarpSize, device_id));
        CUDA_ERRCHK(cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, device_id));
        CUDA_ERRCHK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
        CUDA_ERRCHK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
    }
};

class __attribute__((visibility("default"))) KernelLaunchConfig {
public:
    uint32_t num_blocks = 0;
    uint32_t num_threads = 0;
    uint32_t warp_size = 32;
    uint32_t smem = 0;
    CUstream hStream = NULL;

    KernelLaunchConfig() = default;
    ~KernelLaunchConfig() = default;

    KernelLaunchConfig(uint32_t num_blocks, uint32_t num_threads_per_block, uint32_t smem) :
        num_blocks(num_blocks),
        num_threads(num_threads_per_block),
        smem(smem) 
    { }

    KernelLaunchConfig(int64_t num_blocks_i, int64_t num_threads_i, int64_t smem_i) :
        KernelLaunchConfig( static_cast<uint32_t>(num_blocks_i),
                            static_cast<uint32_t>(num_threads_i),
                            static_cast<uint32_t>(smem_i)) 
    { }
};

/*
* This page is a useful resource on NVRTC: 
* https://docs.nvidia.com/cuda/nvrtc/index.html#example-using-nvrtcgettypename
*/

class __attribute__((visibility("default"))) CUJITKernel {
private:
    nvrtcProgram prog;

    bool compiled = false;
    char* code = nullptr;
    int cu_major, cu_minor;

    CUlibrary library;

    vector<string> kernel_names;
    vector<CUkernel> kernels;

public:
    string kernel_plaintext;
    CUJITKernel(string plaintext) :
        kernel_plaintext(plaintext) {

        NVRTC_SAFE_CALL(
        nvrtcCreateProgram( &prog,                     // prog
                            kernel_plaintext.c_str(),  // buffer
                            "kernel.cu",               // name
                            0,                         // numHeaders
                            NULL,                      // headers
                            NULL));                    // includeNames
    }

    void compile(string kernel_name, const vector<int> template_params, int opt_level=3) {
        vector<string> kernel_names = {kernel_name};
        vector<vector<int>> template_param_list = {template_params};
        compile(kernel_names, template_param_list);
    }

    void compile(vector<string> kernel_names_i, vector<vector<int>> template_param_list, int opt_level=3) {
        DeviceProp dp(0); // We only query the first device on the system at the moment
        cu_major = dp.major;
        cu_minor = dp.minor;

        if(compiled) {
            throw std::logic_error("JIT object has already been compiled!");
        }

        if(kernel_names_i.size() != template_param_list.size()) {
            throw std::logic_error("Kernel names and template parameters must have the same size!");
        }

        for(unsigned int kernel = 0; kernel < kernel_names_i.size(); kernel++) {
            string kernel_name = kernel_names_i[kernel];
            vector<int> &template_params = template_param_list[kernel];

            // Step 1: Generate kernel names from the template parameters 
            if(template_params.size() == 0) {
                kernel_names.push_back(kernel_name);
            }
            else {
                std::string result = kernel_name + "<";
                for(unsigned int i = 0; i < template_params.size(); i++) {
                    result += std::to_string(template_params[i]); 
                    if(i != template_params.size() - 1) {
                        result += ",";
                    }
                }
                result += ">";
                kernel_names.push_back(result);
            }

        }
        
        std::string sm = "-arch=sm_" + std::to_string(cu_major) + std::to_string(cu_minor);

        std::vector<const char*> opts = {
            "--std=c++17",
            sm.c_str(),
            "--split-compile=0",
            "--use_fast_math"
        };    

        // =========================================================
        // Step 2: Add name expressions, compile 
        for(size_t i = 0; i < kernel_names.size(); ++i)
            NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernel_names[i].c_str()));

        nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                        static_cast<int>(opts.size()),     // numOptions
                                                        opts.data()); // options

        size_t logSize;
        NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
        char *log = new char[logSize];
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));

        if (compileResult != NVRTC_SUCCESS) {
            throw std::logic_error("NVRTC Fail, log: " + std::string(log));
        } 
        delete[] log;
        compiled = true;

        // =========================================================
        // Step 3: Get PTX, initialize device, context, and module 

        size_t codeSize;
        NVRTC_SAFE_CALL(nvrtcGetCUBINSize(prog, &codeSize));
        code = new char[codeSize];
        NVRTC_SAFE_CALL(nvrtcGetCUBIN(prog, code));

        CUDA_SAFE_CALL(cuInit(0));
        CUDA_SAFE_CALL(cuLibraryLoadData(&library, code, 0, 0, 0, 0, 0, 0));

        for (size_t i = 0; i < kernel_names.size(); i++) {
            const char *name;

            NVRTC_SAFE_CALL(nvrtcGetLoweredName(
                                    prog,
                    kernel_names[i].c_str(), // name expression
                    &name                    // lowered name
                    ));

            kernels.emplace_back();
            CUDA_SAFE_CALL(cuLibraryGetKernel(&(kernels[i]), library, name));
        }
    }

    void set_max_smem(int kernel_id, uint32_t max_smem_bytes) {
        if(!compiled)
            throw std::logic_error("JIT object has not been compiled!");
        if(kernel_id >= kernels.size())
            throw std::logic_error("Kernel index out of range!");

        int device_count;
        CUDA_SAFE_CALL(cuDeviceGetCount(&device_count));

        for(int i = 0; i < device_count; i++) {
            DeviceProp dp(i);
            if(dp.major == cu_major && dp.minor == cu_minor) {
                CUdevice dev;
                CUDA_SAFE_CALL(cuDeviceGet(&dev, i));
                CUDA_SAFE_CALL(cuKernelSetAttribute(
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        max_smem_bytes,
                        kernels[kernel_id],
                        dev));
            }
        }
    }

    void execute(int kernel_id, void* args[], KernelLaunchConfig config) {
        if(kernel_id >= kernels.size())
            throw std::logic_error("Kernel index out of range!");

        CUcontext pctx = NULL; 
        CUDA_SAFE_CALL(cuCtxGetCurrent(&pctx));

        if(pctx == NULL) {
            int device_id;
            CUdevice dev;
            CUDA_ERRCHK(cudaGetDevice(&device_id));
            CUDA_SAFE_CALL(cuDeviceGet(&dev, device_id));
            CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&pctx, dev));
            CUDA_SAFE_CALL(cuCtxSetCurrent(pctx));
        }

        CUDA_SAFE_CALL(
            cuLaunchKernel( (CUfunction) (kernels[kernel_id]),
                            config.num_blocks, 1, 1,    // grid dim
                            config.num_threads, 1, 1,   // block dim
                            config.smem, config.hStream,       // shared mem and stream
                            args, NULL)          // arguments
        );            
    }

    ~CUJITKernel() {
        if(compiled) {
            CUDA_SAFE_CALL(cuLibraryUnload(library));
            delete[] code;
        }
        NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    }
};