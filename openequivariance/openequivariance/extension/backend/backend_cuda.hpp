#pragma once

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

// ===== Caching & threading includes =====
#include <fstream>
#include <cstdint>
#include <cstdio>       // ← 修复: std::snprintf, std::rename, std::remove
#include <cinttypes>    // ← 修复: PRIx64 跨平台格式化
#include <cstring>
#include <cstdlib>
#include <thread>
#include <sstream>

#ifdef _WIN32
    #include <process.h>
    #include <direct.h>
    #define GET_PID _getpid
    #define MKDIR(path) _mkdir(path)
#else
    #include <unistd.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #define GET_PID getpid
    #define MKDIR(path) ::mkdir(path, 0755)
#endif
// ========================================

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
 * CUBIN Caching for NVRTC JIT Compilation
 * ========================================
 * - Cache location: $OEQ_CACHE_PATH > $HOME/.cache/openequivariance > /tmp/oeq_jit_cache
 * - Disable: OEQ_DISABLE_CACHE=1
 * - Clear:   rm -rf ~/.cache/openequivariance/
 * - Cache key: FNV-1a hash of (NVRTC version + compile opts + source + kernel names + arch)
 * - Format: .names file stores "original_name\tlowered_name" per line for collision detection
 * - Writes use temp file + rename() for crash safety
 */

class __attribute__((visibility("default"))) CUJITKernel {
private:
    nvrtcProgram prog;

    bool compiled = false;
    char* code = nullptr;
    size_t codeSize = 0;
    int cu_major, cu_minor;

    CUlibrary library;

    vector<int> supported_archs;

    vector<string> kernel_names;
    vector<CUkernel> kernels;
    vector<string> lowered_names;

    // =================== Cache & Context Helpers ===================

    // Ensure a CUDA context is bound on the current thread.
    // Critical for multi-threaded usage where worker threads
    // don't automatically have a CUDA context.
    void ensure_cuda_context() {
        CUcontext pctx = NULL;
        CUDA_SAFE_CALL(cuCtxGetCurrent(&pctx));
        if (pctx == NULL) {
            int device_id;
            CUDA_ERRCHK(cudaGetDevice(&device_id));
            CUdevice dev;
            CUDA_SAFE_CALL(cuDeviceGet(&dev, device_id));
            CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&pctx, dev));
            CUDA_SAFE_CALL(cuCtxSetCurrent(pctx));
        }
    }

    // Check if caching is enabled (default: yes).
    static bool cache_enabled() {
        static int result = -1;
        if (result < 0) {
            const char* env = std::getenv("OEQ_DISABLE_CACHE");
            result = (env && (std::strcmp(env, "1") == 0 ||
                              std::strcmp(env, "true") == 0)) ? 0 : 1;
        }
        return result != 0;
    }

    // Get cache directory. Priority:
    //   $OEQ_CACHE_PATH > $HOME/.cache/openequivariance > /tmp/oeq_jit_cache
    static std::string get_cache_dir() {
        const char* env = std::getenv("OEQ_CACHE_PATH");
        if (env && std::strlen(env) > 0)
            return std::string(env);
        const char* home = std::getenv("HOME");
        if (home && std::strlen(home) > 0)
            return std::string(home) + "/.cache/openequivariance";
        return "/tmp/oeq_jit_cache";
    }

    // FNV-1a 64-bit hash → 16-char hex string.
    static std::string fnv1a_hash_hex(const std::string& input) {
        uint64_t hash = 14695981039346656037ULL;
        for (unsigned char c : input) {
            hash ^= static_cast<uint64_t>(c);
            hash *= 1099511628211ULL;
        }
        char buf[17];
        std::snprintf(buf, sizeof(buf), "%016" PRIx64, hash);
        return std::string(buf);
    }

    // Build a deterministic cache key from ALL compilation inputs.
    std::string compute_cache_key(const std::vector<const char*>& compile_opts) {
        int nvrtc_major, nvrtc_minor;
        nvrtcVersion(&nvrtc_major, &nvrtc_minor);

        std::string combined;
        combined.reserve(kernel_plaintext.size() + 512);
        combined += "cache_v2\n";
        combined += "nvrtc=" + std::to_string(nvrtc_major) + "."
                             + std::to_string(nvrtc_minor) + "\n";
        for (const char* opt : compile_opts) {
            combined += "opt=";
            combined += opt;
            combined += "\n";
        }
        combined += "src_len=" + std::to_string(kernel_plaintext.size()) + "\n";
        combined += kernel_plaintext;
        combined += "\n";
        for (const auto& name : kernel_names) {
            combined += "kern=" + name + "\n";
        }

        return "sm" + std::to_string(cu_major) + std::to_string(cu_minor)
               + "_" + fnv1a_hash_hex(combined);
    }

    // mkdir -p equivalent.
    static void mkdir_recursive(const std::string& path) {
        std::string current;
        for (size_t i = 0; i < path.size(); i++) {
            current += path[i];
            if (path[i] == '/' && current.size() > 1)
                MKDIR(current.c_str());
        }
        if (!current.empty() && current != "/")
            MKDIR(current.c_str());
    }

    // Try to load cached CUBIN + lowered names from disk.
    // Returns true on success (kernels vector populated), false on miss/error.
    bool try_load_from_cache(const std::string& cubin_path,
                             const std::string& names_path) {
        std::ifstream cubin_file(cubin_path, std::ios::binary | std::ios::ate);
        if (!cubin_file.is_open()) return false;

        std::ifstream names_file(names_path);
        if (!names_file.is_open()) return false;

        // Parse names file: each line is "original_name\tlowered_name"
        std::vector<std::string> cached_orig;
        std::vector<std::string> cached_lowered;
        {
            std::string line;
            while (std::getline(names_file, line)) {
                if (line.empty()) continue;
                auto tab = line.find('\t');
                if (tab == std::string::npos) return false;  // corrupted
                cached_orig.push_back(line.substr(0, tab));
                cached_lowered.push_back(line.substr(tab + 1));
            }
        }
        names_file.close();

        if (cached_orig.size() != kernel_names.size()) return false;

        // Verify original kernel names match (guards against hash collisions)
        for (size_t i = 0; i < kernel_names.size(); i++) {
            if (cached_orig[i] != kernel_names[i]) return false;
        }

        // Read CUBIN binary
        auto size = static_cast<size_t>(cubin_file.tellg());
        if (size == 0) return false;
        cubin_file.seekg(0, std::ios::beg);

        char* buf = new char[size];
        cubin_file.read(buf, static_cast<std::streamsize>(size));
        bool read_ok = cubin_file.good();
        cubin_file.close();
        if (!read_ok) { delete[] buf; return false; }

        // Load CUBIN into CUDA
        CUDA_SAFE_CALL(cuInit(0));
        ensure_cuda_context();

        CUresult load_result = cuLibraryLoadData(&library, buf, 0, 0, 0, 0, 0, 0);
        if (load_result != CUDA_SUCCESS) {
            // CUBIN incompatible (driver update etc.) — will recompile
            delete[] buf;
            return false;
        }

        // Resolve kernel handles
        for (size_t i = 0; i < cached_lowered.size(); i++) {
            CUkernel k;
            CUresult r = cuLibraryGetKernel(&k, library, cached_lowered[i].c_str());
            if (r != CUDA_SUCCESS) {
                kernels.clear();
                cuLibraryUnload(library);
                delete[] buf;
                return false;
            }
            kernels.push_back(k);
        }

        // Success
        code = buf;
        codeSize = size;
        lowered_names = std::move(cached_lowered);
        return true;
    }

    // Save CUBIN + lowered names to disk via atomic temp+rename.
    void save_to_cache(const std::string& cubin_path,
                       const std::string& names_path) {
        std::string dir = cubin_path.substr(0, cubin_path.find_last_of('/'));
        if (!dir.empty()) mkdir_recursive(dir);

        // Unique temp suffix: PID + thread ID
        std::stringstream ss;
        ss << GET_PID() << "_" << std::this_thread::get_id();
        std::string tmp_suffix = ".tmp." + ss.str();

        std::string tmp_cubin = cubin_path + tmp_suffix;
        std::string tmp_names = names_path + tmp_suffix;

        // Write CUBIN
        {
            std::ofstream f(tmp_cubin, std::ios::binary);
            if (!f.is_open()) return;
            f.write(code, static_cast<std::streamsize>(codeSize));
            f.close();
            if (f.fail()) { std::remove(tmp_cubin.c_str()); return; }
        }

        // Write names: "original_name\tlowered_name" per line
        {
            std::ofstream f(tmp_names);
            if (!f.is_open()) { std::remove(tmp_cubin.c_str()); return; }
            for (size_t i = 0; i < kernel_names.size(); i++) {
                f << kernel_names[i] << "\t" << lowered_names[i] << "\n";
            }
            f.close();
            if (f.fail()) {
                std::remove(tmp_cubin.c_str());
                std::remove(tmp_names.c_str());
                return;
            }
        }

        // Atomic rename
        if (std::rename(tmp_cubin.c_str(), cubin_path.c_str()) != 0) {
            std::remove(tmp_cubin.c_str());
            std::remove(tmp_names.c_str());
            return;
        }
        if (std::rename(tmp_names.c_str(), names_path.c_str()) != 0) {
            std::remove(tmp_names.c_str());
            // cubin already renamed; next load will fail on names → recompile. Safe.
        }
    }

    // =================== End Cache Helpers ===================

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
        nvrtcCreateProgram( &prog,
                            kernel_plaintext.c_str(),
                            "kernel.cu",
                            0,
                            NULL,
                            NULL));
    }

    void compile(string kernel_name, const vector<int> template_params, int opt_level=3) {
        vector<string> kernel_names_local = {kernel_name};
        vector<vector<int>> template_param_list = {template_params};
        compile(kernel_names_local, template_param_list, opt_level);
    }

    void compile(vector<string> kernel_names_i, vector<vector<int>> template_param_list, int opt_level=3) {
        DeviceProp dp(0);
        cu_major = dp.major;
        cu_minor = dp.minor;

        if (compiled) {
            throw std::logic_error("JIT object has already been compiled!");
        }

        if (kernel_names_i.size() != template_param_list.size()) {
            throw std::logic_error("Kernel names and template parameters must have the same size!");
        }

        int device_arch = cu_major * 10 + cu_minor;
        if (std::find(supported_archs.begin(), supported_archs.end(), device_arch) == supported_archs.end()) {
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

        // Step 1: Generate kernel names from template parameters
        for (unsigned int kernel = 0; kernel < kernel_names_i.size(); kernel++) {
            string kernel_name = kernel_names_i[kernel];
            vector<int> &template_params = template_param_list[kernel];

            if (template_params.size() == 0) {
                kernel_names.push_back(kernel_name);
            } else {
                std::string result = kernel_name + "<";
                for (unsigned int i = 0; i < template_params.size(); i++) {
                    result += std::to_string(template_params[i]);
                    if (i != template_params.size() - 1)
                        result += ",";
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

        // ===== Try loading from cache =====
        if (cache_enabled()) {
            std::string cache_key = compute_cache_key(opts);
            std::string cache_dir = get_cache_dir();
            std::string cubin_cache = cache_dir + "/" + cache_key + ".cubin";
            std::string names_cache = cache_dir + "/" + cache_key + ".names";

            if (try_load_from_cache(cubin_cache, names_cache)) {
                compiled = true;
                return;
            }
        }
        // ===== End cache check =====

        // =========================================================
        // Step 2: Add name expressions, compile
        for (size_t i = 0; i < kernel_names.size(); ++i)
            NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernel_names[i].c_str()));

        nvrtcResult compileResult = nvrtcCompileProgram(prog,
                                                        static_cast<int>(opts.size()),
                                                        opts.data());

        size_t logSize;
        NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
        char *log = new char[logSize];
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));

        if (compileResult != NVRTC_SUCCESS) {
            std::string log_str(log);
            delete[] log;
            throw std::logic_error("NVRTC Fail, log: " + log_str);
        }
        delete[] log;
        compiled = true;

        // =========================================================
        // Step 3: Get CUBIN, load module

        NVRTC_SAFE_CALL(nvrtcGetCUBINSize(prog, &codeSize));
        code = new char[codeSize];
        NVRTC_SAFE_CALL(nvrtcGetCUBIN(prog, code));

        CUDA_SAFE_CALL(cuInit(0));
        ensure_cuda_context();
        CUDA_SAFE_CALL(cuLibraryLoadData(&library, code, 0, 0, 0, 0, 0, 0));

        for (size_t i = 0; i < kernel_names.size(); i++) {
            const char *name;
            NVRTC_SAFE_CALL(nvrtcGetLoweredName(
                            prog,
                            kernel_names[i].c_str(),
                            &name));

            lowered_names.push_back(std::string(name));
            kernels.emplace_back();
            CUDA_SAFE_CALL(cuLibraryGetKernel(&(kernels[i]), library, name));
        }

        // ===== Save to cache =====
        if (cache_enabled()) {
            std::string cache_key = compute_cache_key(opts);
            std::string cache_dir = get_cache_dir();
            save_to_cache(cache_dir + "/" + cache_key + ".cubin",
                          cache_dir + "/" + cache_key + ".names");
        }
        // ===== End save =====
    }

    void set_max_smem(int kernel_id, uint32_t max_smem_bytes) {
        if (!compiled)
            throw std::logic_error("JIT object has not been compiled!");
        if (kernel_id >= (int)kernels.size())
            throw std::logic_error("Kernel index out of range!");

        int device_count;
        CUDA_SAFE_CALL(cuDeviceGetCount(&device_count));

        for (int i = 0; i < device_count; i++) {
            DeviceProp dp(i);
            if (dp.major == cu_major && dp.minor == cu_minor) {
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
        if (kernel_id >= (int)kernels.size())
            throw std::logic_error("Kernel index out of range!");

        CUcontext pctx = NULL;
        CUDA_SAFE_CALL(cuCtxGetCurrent(&pctx));

        if (pctx == NULL) {
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
        if (compiled) {
            auto result = cuLibraryUnload(library);
            if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
                std::cout << "Failed to unload CUDA library, error code: "
                          << ((int) result) << std::endl;
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