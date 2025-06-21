#pragma once

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <vector>
#include <string>
#include <iostream>
#include <memory>

using namespace std;
using Stream = hipStream_t; 

#define HIPRTC_SAFE_CALL(x)                                     \
do {                                                            \
   hiprtcResult result = x;                                     \
   if (result != HIPRTC_SUCCESS) {                              \
      std::cerr << "\nerror: " #x " failed with error "         \
               << hiprtcGetErrorString(result) << '\n';            \
      exit(1);                                                  \
   }                                                            \
} while(0)

#define HIP_SAFE_CALL(x)                                        \
do {                                                            \
   hipResult_t result = x;                                         \
   if (result != HIP_SUCCESS) {                                \
      const char *msg;                                          \
      hipGetErrorName(result, &msg);                            \
      std::cerr << "\nerror: " #x " failed with error "         \
               << msg << '\n';                                  \
      exit(1);                                                  \
   }                                                            \
} while(0)

#define HIP_ERRCHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class HIP_Allocator {
public:
    static void* gpu_alloc (size_t size) {
        void* ptr;
        HIP_ERRCHK( hipMalloc((void**) &ptr, size ))
        return ptr;
    }

    static void gpu_free (void* ptr) {
        HIP_ERRCHK( hipFree(ptr))
    }

    static void copy_host_to_device (void* host, void* device, size_t size) {
        HIP_ERRCHK( hipMemcpyHtoD(device, host, size));
    }

    static void copy_device_to_host (void* host, void* device, size_t size) {
        HIP_ERRCHK( hipMemcpyDtoH(host, device, size));
    }
};

class GPUTimer {
    hipEvent_t start_evt, stop_evt;

public:
    GPUTimer() {  
        HIP_ERRCHK(hipEventCreate(&start_evt))
        HIP_ERRCHK(hipEventCreate(&stop_evt));
    }

    void start() {
        HIP_ERRCHK(hipEventRecord(start_evt));
    }

    float stop_clock_get_elapsed() {
        float time_millis;
        HIP_ERRCHK(hipEventRecord(stop_evt));
        HIP_ERRCHK(hipEventSynchronize(stop_evt));
        HIP_ERRCHK(hipEventElapsedTime(&time_millis, start_evt, stop_evt));
        return time_millis; 
    }

    void clear_L2_cache() {
        size_t element_count = 25000000;

        int* ptr = (int*) (HIP_Allocator::gpu_alloc(element_count * sizeof(int)));
        HIP_ERRCHK(hipMemset(ptr, 42, element_count * sizeof(int)))
        HIP_Allocator::gpu_free(ptr);
        HIP_ERRCHK(hipDeviceSynchronize());
    }
    
    ~GPUTimer() {
        HIP_ERRCHK(hipEventDestroy(start_evt));
        HIP_ERRCHK(hipEventDestroy(stop_evt));
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
        hipDeviceProp_t prop; 
        HIP_ERRCHK(hipGetDeviceProperties(&prop, device_id));
        name = std::string(prop.name);
        HIP_ERRCHK(hipDeviceGetAttribute(&maxSharedMemoryPerMultiprocessor, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device_id));
        HIP_ERRCHK(hipDeviceGetAttribute(&maxSharedMemPerBlock, hipDeviceAttributeMaxSharedMemoryPerBlock, device_id));
        HIP_ERRCHK(hipDeviceGetAttribute(&warpsize, hipDeviceAttributeWarpSize, device_id));
        HIP_ERRCHK(hipDeviceGetAttribute(&multiprocessorCount, hipDeviceAttributeMultiprocessorCount, device_id));
    }
};

/*
* Guide to HIPRTC: https://rocm.docs.amd.com/projects/HIP/en/docs-5.7.1/user_guide/hip_rtc.html
*/

class __attribute__((visibility("default"))) KernelLaunchConfig {
public:
    uint32_t num_blocks = 0;
    uint32_t num_threads = 0;
    uint32_t warp_size = 64;
    uint32_t smem = 0;
    hipStream_t hStream = NULL;

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

class __attribute__((visibility("default"))) KernelLibrary {
    hipModule_t library;
    vector<hipFunction_t> kernels;

public:
    int device;
    KernelLibrary(hiprtcProgram &prog, vector<char> &kernel_binary, vector<string> &kernel_names) {
        HIP_ERRCHK(hipGetDevice(&device));
        HIP_ERRCHK(hipModuleLoadData(&library, kernel_binary.data()));

        for (size_t i = 0; i < kernel_names.size(); i++) {
            const char *name;

            HIPRTC_SAFE_CALL(hiprtcGetLoweredName(
                    prog,
                    kernel_names[i].c_str(), // name expression
                    &name                    // lowered name
                    ));

            kernels.emplace_back();
            HIP_ERRCHK(hipModuleGetFunction(&(kernels[i]), library, name));
        }
    }

    hipFunction_t operator[](int kernel_id) {
        if(kernel_id >= kernels.size())
            throw std::logic_error("Kernel index out of range!");

        return kernels[kernel_id];
    }

    ~KernelLibrary() { 
        HIP_ERRCHK(hipModuleUnload(library)); 
    }
}; 

class __attribute__((visibility("default"))) HIPJITKernel {
private:
    hiprtcProgram prog;
    bool compiled = false;

    vector<string> kernel_names;
    unique_ptr<KernelLibrary> kernels;

public:
    string kernel_plaintext;
    vector<char> kernel_binary;

    HIPJITKernel(string plaintext) :
        kernel_plaintext(plaintext) {

        HIPRTC_SAFE_CALL(
        hiprtcCreateProgram( &prog,                    // prog
                            kernel_plaintext.c_str(),  // buffer
                            "kernel.hip",              // name
                            0,                         // numHeaders
                            NULL,                      // headers
                            NULL));                    // includeNames
    }

    void compile(string kernel_name, const vector<int> template_params, int opt_level=3) {
        vector<string> kernel_names = {kernel_name};
        vector<vector<int>> template_param_list = {template_params};
        compile(kernel_names, template_param_list, opt_level);
    }

    void compile(vector<string> kernel_names_i, vector<vector<int>> template_param_list, int opt_level=3) {
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

        hipDeviceProp_t props;
        int device = 0;
        HIP_ERRCHK(hipGetDeviceProperties(&props, device));
        std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;  
        std::string opt_arg = "-O" + std::to_string(opt_level);

        std::vector<const char*> opts = {
            "--std=c++17",
            opt_arg.c_str(),
            sarg.c_str()
        }; 

        // =========================================================
        // Step 2: Add name expressions, compile 
        for(size_t i = 0; i < kernel_names.size(); ++i)
            HIPRTC_SAFE_CALL(hiprtcAddNameExpression(prog, kernel_names[i].c_str()));

        hiprtcResult compileResult = hiprtcCompileProgram(prog,  // prog
                                                        static_cast<int>(opts.size()),     // numOptions
                                                        opts.data()); // options

        size_t logSize;
        HIPRTC_SAFE_CALL(hiprtcGetProgramLogSize(prog, &logSize));
        char *log = new char[logSize];
        HIPRTC_SAFE_CALL(hiprtcGetProgramLog(prog, log));

        if (compileResult != HIPRTC_SUCCESS) {
            throw std::logic_error("HIPRTC Fail, log: " + std::string(log));
        } 
        delete[] log;
        compiled = true;

        // =========================================================
        // Step 3: Get PTX, initialize device, context, and module 

        size_t codeSize;
        HIPRTC_SAFE_CALL(hiprtcGetCodeSize(prog, &codeSize));
        kernel_binary.resize(codeSize);
        hiprtcGetCode(prog, kernel_binary.data());
        kernels.reset(new KernelLibrary(prog, kernel_binary, kernel_names));
    }

    void set_max_smem(int kernel_id, uint32_t max_smem_bytes) {
        // Ignore for AMD GPUs 
    }

    void execute(int kernel_id, void* args[], KernelLaunchConfig config) {
        int device_id; HIP_ERRCHK(hipGetDevice(&device_id));
        if(device_id != kernels->device) {
            kernels.reset();
            kernels.reset(new KernelLibrary(prog, kernel_binary, kernel_names)); 
        }

        HIP_ERRCHK(
            hipModuleLaunchKernel( ((*kernels)[kernel_id]),
                            config.num_blocks, 1, 1,    // grid dim
                            config.num_threads, 1, 1,   // block dim
                            config.smem, config.hStream,       // shared mem and stream
                            args, NULL)          // arguments
        );            
    }

    ~HIPJITKernel() {
        HIPRTC_SAFE_CALL(hiprtcDestroyProgram(&prog));
    }
};