#pragma once

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
// ===== NEW: caching includes =====
#include <fstream>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <sys/stat.h>
// ===== END NEW =====

using namespace std;
using Stream = cudaStream_t;

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                             \
    nvrtcResult result = x;                                        \
    if (result != NVRTC_SUCCESS) {                                 \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';            \
      exit(1);                                                     \
    }                                                              \
  } while(0)

#define CUDA_SAFE_CALL(x)                                          \
  do {                                                             \
    CUresult result = x;                                           \
    if (result != CUDA_SUCCESS) {                                  \
      const char *msg;                                             \
      cuGetErrorName(result, &msg);                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                    \
      exit(1);                                                     \
    }                                                              \
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
    static void* gpu_alloc(size_t size) {
        void* ptr;
        CUDA_ERRCHK( cudaMalloc((void**) &ptr, size) )
        return ptr;
    }

    static void gpu_free(void* ptr) {
        CUDA_ERRCHK( cudaFree(ptr) )
    }

    static void copy_host_to_device(void* host, void* device, size_t size) {
        CUDA_ERRCHK( cudaMemcpy(device, host, size, cudaMemcpyHostToDevice) );
    }

    static void copy_device_to_host(void* host, void* device, size_t size) {
        CUDA_ERRCHK( cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost) );
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
        int* ptr = (int*)(CUDA_Allocator::gpu_alloc(element_count * sizeof(int)));
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
    size_t codeSize = 0;           // ===== NEW: stored as member for caching =====
    int cu_major, cu_minor;

    CUlibrary library;

    vector<int> supported_archs;

    vector<string> kernel_names;
    vector<CUkernel> kernels;
    vector<string> lowered_names;  // ===== NEW: stored for caching =====

    // ===== NEW: Cache helper methods =====

    // Get cache directory from CUDA_CACHE_PATH env var (or default)
    static std::string get_cache_dir() {
        const char* env = std::getenv("CUDA_CACHE_PATH");
        if (env && std::strlen(env) > 0) {
            return std::string(env);
        }
        return "./cuda_jit_cache";
    }

    // FNV-1a 64-bit hash
    static std::string fnv1a_hash(const std::string& input) {
        uint64_t hash = 14695981039346656037ULL;
        for (unsigned char c : input) {
            hash ^= static_cast<uint64_t>(c);
            hash *= 1099511628211ULL;
        }
        char buf[17];
        std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)hash);
        return std::string(buf);
    }

    // Compute a unique cache key from source + arch + kernel names + NVRTC version
    std::string compute_cache_key(const std::string& sm_flag) {
        int nvrtc_major, nvrtc_minor;
        nvrtcVersion(&nvrtc_major, &nvrtc_minor);

        std::string combined;
        combined += "cache_v1\n";
        combined += "nvrtc=" + std::to_string(nvrtc_major) + "." + std::to_string(nvrtc_minor) + "\n";
        combined += "arch=" + sm_flag + "\n";
        combined += "src_len=" + std::to_string(kernel_plaintext.size()) + "\n";
        combined += kernel_plaintext;
        combined += "\n";
        for (const auto& name : kernel_names) {
            combined += "kern=" + name + "\n";
        }

        return "sm_" + std::to_string(cu_major) + std::to_string(cu_minor)
               + "_" + fnv1a_hash(combined);
    }

    // Create directories recursively (POSIX)
    static void mkdir_recursive(const std::string& path) {
        std::string current;
        for (size_t i = 0; i < path.size(); i++) {
            current += path[i];
            if (path[i] == '/') {
                ::mkdir(current.c_str(), 0755);
            }
        }
        if (!current.empty()) {
            ::mkdir(current.c_str(), 0755);
        }
    }

    // Try to load cached CUBIN + lowered names from disk.
    // Returns true on success (kernels are ready), false on cache miss or any error.
    bool try_load_from_cache(const std::string& cubin_path, const std::string& names_path) {
        // 1. Open both files
        std::ifstream cubin_file(cubin_path, std::ios::binary | std::ios::ate);
        if (!cubin_file.is_open()) return false;

        std::ifstream names_file(names_path);
        if (!names_file.is_open()) return false;

        // 2. Read lowered names
        std::vector<std::string> cached_names;
        std::string line;
        while (std::getline(names_file, line)) {
            if (!line.empty()) {
                cached_names.push_back(line);
            }
        }

        if (cached_names.size() != kernel_names.size()) {
            return false;  // Mismatch: cache is stale or corrupted
        }

        // 3. Read CUBIN binary
        size_t size = static_cast<size_t>(cubin_file.tellg());
        if (size == 0) return false;
        cubin_file.seekg(0, std::ios::beg);

        char* cached_code = new char[size];
        cubin_file.read(cached_code, size);
        if (!cubin_file.good()) {
            delete[] cached_code;
            return false;
        }

        // 4. Load CUBIN into CUDA
        CUDA_SAFE_CALL(cuInit(0));

        CUresult load_result = cuLibraryLoadData(&library, cached_code, 0, 0, 0, 0, 0, 0);
        if (load_result != CUDA_SUCCESS) {
            delete[] cached_code;
            return false;  // CUBIN incompatible (e.g., driver update), will recompile
        }

        // 5. Resolve kernel handles
        for (size_t i = 0; i < cached_names.size(); i++) {
            CUkernel k;
            CUresult r = cuLibraryGetKernel(&k, library, cached_names[i].c_str());
            if (r != CUDA_SUCCESS) {
                kernels.clear();
                cuLibraryUnload(library);
                delete[] cached_code;
                return false;
            }
            kernels.push_back(k);
        }

        // Success — store state
        code = cached_code;
        codeSize = size;
        lowered_names = cached_names;

        std::cerr << "[CUJITKernel] Cache HIT: loaded " << kernel_names.size()
                  << " kernel(s) from " << cubin_path << std::endl;
        return true;
    }

    // Save CUBIN + lowered names to disk for future runs.
    void save_to_cache(const std::string& cubin_path, const std::string& names_path) {
        std::string dir = cubin_path.substr(0, cubin_path.find_last_of('/'));
        if (!dir.empty()) {
            mkdir_recursive(dir);
        }

        std::ofstream cubin_file(cubin_path, std::ios::binary);
        if (cubin_file.is_open()) {
            cubin_file.write(code, static_cast<std::streamsize>(codeSize));
            cubin_file.close();
        }

        std::ofstream names_file(names_path);
        if (names_file.is_open()) {
            for (const auto& name : lowered_names) {
                names_file << name << "\n";
            }
            names_file.close();
        }

        std::cerr << "[CUJITKernel] Cache SAVE: wrote " << kernel_names.size()
                  << " kernel(s) to " << cubin_path << std::endl;
    }

    // ===== END NEW =====

public:
    string kernel_plaintext;
    CUJITKernel(string plaintext) :
        kernel_plaintext(plaintext) {

        int num_supported_archs;
        NVRTC_SAFE_CALL(
        nvrtcGetNumSupportedArchs(&num_supported_archs));

        supported_archs.resize(num_supported_archs);
        NVRTC_SAFE_CALL(
        nvrtcGetSupportedArchs(supported_archs.data()));

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
        DeviceProp dp(0);
        cu_major = dp.major;
        cu_minor = dp.minor;

        if(compiled) {
            throw std::logic_error("JIT object has already been compiled!");
        }

        if(kernel_names_i.size() != template_param_list.size()) {
            throw std::logic_error("Kernel names and template parameters must have the same size!");
        }

        int device_arch = cu_major * 10 + cu_minor;
        if (std::find(supported_archs.begin(), supported_archs.end(), device_arch) == supported_archs.end()){
            int nvrtc_version_major, nvrtc_version_minor;
            NVRTC_SAFE_CALL(
            nvrtcVersion(&nvrtc_version_major, &nvrtc_version_minor));

            throw std::runtime_error("NVRTC version "
                + std::to_string(nvrtc_version_major)
                + "."
                + std::to_string(nvrtc_version_minor)
                + " does not support device architecture "
                + std::to_string(device_arch)
            );
        }

        for(unsigned int kernel = 0; kernel < kernel_names_i.size(); kernel++) {
            string kernel_name = kernel_names_i[kernel];
            vector<int> &template_params = template_param_list[kernel];

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

        // ===== NEW: Try loading from cache before NVRTC compilation =====
        {
            std::string cache_key = compute_cache_key(sm);
            std::string cache_dir = get_cache_dir();
            std::string cubin_cache = cache_dir + "/" + cache_key + ".cubin";
            std::string names_cache = cache_dir + "/" + cache_key + ".names";

            if (try_load_from_cache(cubin_cache, names_cache)) {
                compiled = true;
                return;  // Skip NVRTC compilation entirely!
            }

            std::cerr << "[CUJITKernel] Cache MISS for sm_"
                      << cu_major << cu_minor
                      << ", compiling with NVRTC..." << std::endl;
        }
        // ===== END NEW =====

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

        nvrtcResult compileResult = nvrtcCompileProgram(prog,
                                                        static_cast<int>(opts.size()),
                                                        opts.data());

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
        // Step 3: Get CUBIN, initialize device, context, and module

        NVRTC_SAFE_CALL(nvrtcGetCUBINSize(prog, &codeSize));  // ===== CHANGED: use member =====
        code = new char[codeSize];
        NVRTC_SAFE_CALL(nvrtcGetCUBIN(prog, code));

        CUDA_SAFE_CALL(cuInit(0));
        CUDA_SAFE_CALL(cuLibraryLoadData(&library, code, 0, 0, 0, 0, 0, 0));

        for (size_t i = 0; i < kernel_names.size(); i++) {
            const char *name;

            NVRTC_SAFE_CALL(nvrtcGetLoweredName(
                            prog,
                            kernel_names[i].c_str(),
                            &name
                            ));

            lowered_names.push_back(std::string(name));  // ===== NEW: store lowered name =====

            kernels.emplace_back();
            CUDA_SAFE_CALL(cuLibraryGetKernel(&(kernels[i]), library, name));
        }

        // ===== NEW: Save to cache for future runs =====
        {
            std::string cache_key = compute_cache_key(sm);
            std::string cache_dir = get_cache_dir();
            std::string cubin_cache = cache_dir + "/" + cache_key + ".cubin";
            std::string names_cache = cache_dir + "/" + cache_key + ".names";
            save_to_cache(cubin_cache, names_cache);
        }
        // ===== END NEW =====
    }

    void set_max_smem(int kernel_id, uint32_t max_smem_bytes) {
        if(!compiled)
            throw std::logic_error("JIT object has not been compiled!");
        if(kernel_id >= (int)kernels.size())
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
        if(kernel_id >= (int)kernels.size())
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
            cuLaunchKernel( (CUfunction)(kernels[kernel_id]),
                            config.num_blocks, 1, 1,
                            config.num_threads, 1, 1,
                            config.smem, config.hStream,
                            args, NULL)
        );
    }

    ~CUJITKernel() {
        if(compiled) {
            auto result = cuLibraryUnload(library);
            if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
                std::cout << "Failed to unload CUDA library, error code: " << ((int) result) << std::endl;
            }

            delete[] code;
        }
        NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    }
};

KernelLaunchConfig with_stream(const KernelLaunchConfig& config, Stream stream) {
    KernelLaunchConfig new_config = config;
    new_config.hStream = stream;
    return new_config;
}