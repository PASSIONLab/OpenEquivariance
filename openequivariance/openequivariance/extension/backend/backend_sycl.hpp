#pragma once

#include <cstdint>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>

using namespace std;
namespace syclex = sycl::ext::oneapi::experimental;

/*
* SYCL streams are queues. Unlike CUDA / HIP, a sycl::queue is a
* reference-counted handle rather than an opaque pointer, so the "stream" type
* is a pointer to the queue owned by the caller (PyTorch).
*/
using Stream = sycl::queue *;

// Defined by the translation unit that binds this header to the framework
// (PyTorch / JAX). Returns the queue of the framework's current stream.
Stream get_current_stream();

// Returns the queue the kernels should be submitted to. A null stream means
// "no queue was supplied", in which case we fall back to the framework's
// current stream, and only then to a process-wide default queue.
inline sycl::queue &resolve_queue(Stream stream) {
    if (stream != nullptr) {
        return *stream;
    }
    if (Stream current = get_current_stream()) {
        return *current;
    }
    static sycl::queue default_queue{sycl::gpu_selector_v};
    return default_queue;
}

class SYCL_Allocator {
public:
    static void* gpu_alloc (size_t size) {
        sycl::queue &q = resolve_queue(nullptr);
        void* ptr = sycl::malloc_device(size, q);
        if (ptr == nullptr) {
            throw std::runtime_error("SYCL device allocation failed!");
        }
        return ptr;
    }

    static void gpu_free (void* ptr) {
        sycl::queue &q = resolve_queue(nullptr);
        sycl::free(ptr, q);
    }

    static void copy_host_to_device (void* host, void* device, size_t size) {
        sycl::queue &q = resolve_queue(nullptr);
        q.memcpy(device, host, size).wait();
    }

    static void copy_device_to_host (void* host, void* device, size_t size) {
        sycl::queue &q = resolve_queue(nullptr);
        q.memcpy(host, device, size).wait();
    }
};

/*
* SYCL has no direct equivalent of cudaEvent elapsed time that works without
* enabling profiling on the queue, so the timer brackets a wall-clock interval
* around a queue synchronization.
*/
class GPUTimer {
    std::chrono::time_point<std::chrono::steady_clock> start_time;

public:
    GPUTimer() = default;

    void start() {
        sycl::queue &q = resolve_queue(nullptr);
        q.wait();
        start_time = std::chrono::steady_clock::now();
    }

    float stop_clock_get_elapsed() {
        sycl::queue &q = resolve_queue(nullptr);
        q.wait();
        auto stop_time = std::chrono::steady_clock::now();
        std::chrono::duration<float, std::milli> elapsed = stop_time - start_time;
        return elapsed.count();
    }

    void clear_L2_cache() {
        size_t element_count = 25000000;
        sycl::queue &q = resolve_queue(nullptr);

        int* ptr = (int*) sycl::malloc_device(element_count * sizeof(int), q);
        q.memset(ptr, 42, element_count * sizeof(int)).wait();
        sycl::free(ptr, q);
        q.wait();
    }

    ~GPUTimer() = default;
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
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (devices.empty()) {
            throw std::runtime_error("No SYCL GPU devices found!");
        }
        if (device_id < 0 || static_cast<size_t>(device_id) >= devices.size()) {
            device_id = 0;
        }
        sycl::device dev = devices[device_id];

        name = dev.get_info<sycl::info::device::name>();
        multiprocessorCount =
            static_cast<int>(dev.get_info<sycl::info::device::max_compute_units>());

        // A SYCL sub-group is the analogue of a CUDA warp / HIP wavefront. Pick
        // the largest supported size that the kernel generator can target.
        auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
        warpsize = 32;
        if (!sg_sizes.empty()) {
            if (std::find(sg_sizes.begin(), sg_sizes.end(), size_t(32)) != sg_sizes.end()) {
                warpsize = 32;
            } else {
                warpsize = static_cast<int>(
                    *std::max_element(sg_sizes.begin(), sg_sizes.end()));
            }
        }

        maxSharedMemPerBlock =
            static_cast<int>(dev.get_info<sycl::info::device::local_mem_size>());
        maxSharedMemoryPerMultiprocessor = maxSharedMemPerBlock;

        // SYCL exposes no compute-capability equivalent. These fields exist
        // only for parity with the CUDA backend and are unused on SYCL.
        major = 0;
        minor = 0;
    }
};

class __attribute__((visibility("default"))) KernelLaunchConfig {
public:
    uint32_t num_blocks = 0;
    uint32_t num_threads = 0;
    uint32_t warp_size = 32;
    uint32_t smem = 0;
    Stream hStream = nullptr;

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
* Runtime compilation uses the SYCL kernel_compiler extension with
* source_language::sycl, documented at
* https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler_sycl.asciidoc
*
* The generated kernels are free functions marked with nd_range_kernel, so they
* are launched with raw (untyped) arguments exactly like cuLaunchKernel takes a
* void* array.
*/
class __attribute__((visibility("default"))) SYCLJITKernel {
private:
    bool compiled = false;

    vector<string> kernel_names;
    vector<sycl::kernel> kernels;
    std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> bundle;

public:
    string kernel_plaintext;

    SYCLJITKernel(string plaintext) :
        kernel_plaintext(plaintext) { }

    void compile(string kernel_name, const vector<int> template_params, int opt_level=3) {
        vector<string> kernel_names_i = {kernel_name};
        vector<vector<int>> template_param_list = {template_params};
        compile(kernel_names_i, template_param_list, opt_level);
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

        // Build against the context the kernels will actually run in, so the
        // resulting bundle is valid for every device that context spans.
        sycl::queue &q = resolve_queue(nullptr);
        sycl::context build_context = q.get_context();

        if(!q.get_device().ext_oneapi_can_build(syclex::source_language::sycl)) {
            throw std::runtime_error(
                "The SYCL device does not support runtime compilation of SYCL source.");
        }

        std::string opt_arg = "-O" + std::to_string(opt_level);
        std::vector<std::string> build_opts = {opt_arg, "-ffast-math"};

        std::string log;
        try {
            auto source_bundle = syclex::create_kernel_bundle_from_source(
                build_context,
                syclex::source_language::sycl,
                kernel_plaintext);

            auto exe_bundle = syclex::build(
                source_bundle,
                syclex::properties{
                    syclex::build_options{build_opts},
                    syclex::save_log{&log}});

            bundle = std::make_unique<
                sycl::kernel_bundle<sycl::bundle_state::executable>>(
                    std::move(exe_bundle));
        } catch (const sycl::exception &e) {
            throw std::logic_error("SYCL runtime compilation failed: "
                + std::string(e.what()) + "\nlog: " + log);
        }

        compiled = true;

        for (size_t i = 0; i < kernel_names.size(); i++) {
            kernels.push_back(bundle->ext_oneapi_get_kernel(kernel_names[i]));
        }
    }

    void set_max_smem(int kernel_id, uint32_t max_smem_bytes) {
        // Shared (local) memory is declared statically inside the generated
        // kernel via work_group_static, so there is no opt-in to perform here.
        // Validate the request against the device limit so an oversubscription
        // fails with a clear message instead of at launch.
        if(!compiled)
            throw std::logic_error("JIT object has not been compiled!");
        if(static_cast<size_t>(kernel_id) >= kernels.size())
            throw std::logic_error("Kernel index out of range!");

        sycl::queue &q = resolve_queue(nullptr);
        size_t local_mem = q.get_device().get_info<sycl::info::device::local_mem_size>();
        if(static_cast<size_t>(max_smem_bytes) > local_mem) {
            throw std::runtime_error("Requested shared memory ("
                + std::to_string(max_smem_bytes)
                + " bytes) exceeds the device local memory size ("
                + std::to_string(local_mem) + " bytes).");
        }
    }

    void execute(int kernel_id, void* args[], const size_t arg_sizes[],
                 size_t num_args, KernelLaunchConfig config) {
        if(!compiled)
            throw std::logic_error("JIT object has not been compiled!");
        if(static_cast<size_t>(kernel_id) >= kernels.size())
            throw std::logic_error("Kernel index out of range!");

        sycl::queue &q = resolve_queue(config.hStream);

        std::vector<syclex::raw_kernel_arg> raw_args;
        raw_args.reserve(num_args);
        for (size_t i = 0; i < num_args; i++) {
            raw_args.emplace_back(args[i], arg_sizes[i]);
        }

        sycl::nd_range<1> range{
            sycl::range<1>(static_cast<size_t>(config.num_blocks) *
                           static_cast<size_t>(config.num_threads)),
            sycl::range<1>(static_cast<size_t>(config.num_threads))};

        sycl::kernel &k = kernels[kernel_id];

        syclex::submit(q, [&](sycl::handler &cgh) {
            for (size_t i = 0; i < raw_args.size(); i++) {
                cgh.set_arg(static_cast<int>(i), raw_args[i]);
            }
            cgh.parallel_for(range, k);
        });
    }

    ~SYCLJITKernel() = default;
};

inline KernelLaunchConfig with_stream(const KernelLaunchConfig& config, Stream stream) {
    KernelLaunchConfig new_config = config;
    new_config.hStream = stream;
    return new_config;
}
