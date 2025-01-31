#pragma once

#include <hip_runtime.h>
#include <hip/hiprtc.h>

#include <string>
#include <iostream>
#include "jit.hpp"

using namespace std;

#define HIPRTC_SAFE_CALL(x)                                     \
do {                                                            \
   hiprtcResult result = x;                                     \
   if (result != HIPRTC_SUCCESS) {                              \
      std::cerr << "\nerror: " #x " failed with error "         \
               << hipGetErrorString(result) << '\n';            \
      exit(1);                                                  \
   }                                                            \
} while(0)

#define HIP_SAFE_CALL(x)                                        \
do {                                                            \
   CUresult result = x;                                         \
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
        hipEventCreate(&start_evt);
        hipEventCreate(&stop_evt);
    }

    void start() {
        hipEventRecord(start_evt);
    }

    float stop_clock_get_elapsed() {
        float time_millis;
        hipEventRecord(stop_evt);
        hipEventSynchronize(stop_evt);
        hipEventElapsedTime(&time_millis, start_evt, stop_evt);
        return time_millis; 
    }

    void clear_L2_cache() {
        size_t element_count = 25000000;

        int* ptr = (int*) (HIP_Allocator::gpu_alloc(element_count * sizeof(int)));
        HIP_ERRCHK(hipMemset(ptr, 42, element_count * sizeof(int)))
        HIP_Allocator::gpu_free(ptr);
        hipDeviceSynchronize();
    }
    
    ~GPUTimer() {
        hipEventDestroy(start_evt);
        hipEventDestroy(stop_evt);
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
        hipGetDeviceProperties(&prop, device_id);
        name = std::string(prop.name);
        HIP_ERRCHK(hipDeviceGetAttribute(&maxSharedMemoryPerMultiprocessor, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device_id));
        HIP_ERRCHK(hipDeviceGetAttribute(&maxSharedMemPerBlock, hipDeviceAttributeMaxSharedMemoryPerBlockOptin, device_id));
        HIP_ERRCHK(hipDeviceGetAttribute(&warpsize, hipDeviceAttributeWarpSize, device_id));
        HIP_ERRCHK(hipDeviceGetAttribute(&multiprocessorCount, hipDeviceAttributeMultiProcessorCount, device_id));
    }
};

/*
* Guide to HIPRTC: https://rocm.docs.amd.com/projects/HIP/en/docs-5.7.1/user_guide/hip_rtc.html
*/

class __attribute__((visibility("default"))) HIPJITKernel {
private:
    string kernel_plaintext;
    hiprtcProgram prog;

    bool compiled = false;
    char* code = nullptr;

    hipModule_t library; 

    vector<string> kernel_names;
    vector<hipFunction_t> kernels;

public:
    CUJITKernel(string plaintext) :
        kernel_plaintext(plaintext) {

        //HIP_ERRCHK(hipFree(0)); // No-op to initialize the primary context 
        HIPRTC_SAFE_CALL(
        hiprtcCreateProgram( &prog,                    // prog
                            kernel_plaintext.c_str(),  // buffer
                            "kernel.hip",              // name
                            0,                         // numHeaders
                            NULL,                      // headers
                            NULL));                    // includeNames
    }

    void compile(string kernel_name, const vector<int> template_params) {
        vector<string> kernel_names = {kernel_name};
        vector<vector<int>> template_param_list = {template_params};
        compile(kernel_names, template_param_list);
    }

    void compile(vector<string> kernel_names_i, vector<vector<int>> template_param_list) {
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
        HIP_CHECK(hipGetDeviceProperties(&props, device));
        std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;  

        std::vector<const char*> opts = {
            "--std=c++17",
            sarg.c_str(),
            "--split-compile=0",
            "--use_fast_math"
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
        code = new char[codeSize];
        HIPRTC_SAFE_CALL(hiprtcGetCode(prog, code));

        //HIP_SAFE_CALL(cuInit(0));
        HIP_SAFE_CALL(hipModuleLoadData(&library, code, 0, 0, 0, 0, 0, 0));

        for (size_t i = 0; i < kernel_names.size(); i++) {
            const char *name;

            HIPRTC_SAFE_CALL(hiprtcGetLoweredName(
                    prog,
                    kernel_names[i].c_str(), // name expression
                    &name                    // lowered name
                    ));

            kernels.emplace_back();
            HIP_SAFE_CALL(hipModuleGetFunction(&(kernels[i]), library, name));
        }
    }

    void set_max_smem(int kernel_id, uint32_t max_smem_bytes) {
        if(kernel_id >= kernels.size())
            throw std::logic_error("Kernel index out of range!");

        HIP_SAFE_CALL(hipFuncSetAttribute(
                hipFuncAttributeMaxDynamicSharedMemorySize, 
                max_smem_bytes,
                kernels[kernel_id]));
    }

    void execute(int kernel_id, void* args[], KernelLaunchConfig config) {
        if(kernel_id >= kernels.size())
            throw std::logic_error("Kernel index out of range!");

        HIP_SAFE_CALL(
            hipModuleLaunchKernel( (kernels[kernel_id]),
                            config.num_blocks, 1, 1,    // grid dim
                            config.num_threads, 1, 1,   // block dim
                            config.smem, config.hStream,       // shared mem and stream
                            args, NULL)          // arguments
        );            
    }

    ~CUJITKernel() {
        if(compiled) {
            HIP_SAFE_CALL(hipModuleUnload(library));
            delete[] code;
        }
        HIPRTC_SAFE_CALL(hiprtcDestroyProgram(&prog));
    }
};

