{#
Compatibility shim that lets the CUDA/HIP-flavored kernel templates compile as
SYCL free-function kernels under runtime compilation. Included only when
targeting the SYCL backend; CUDA and HIP see none of this.

The generated source is compiled by the SYCL kernel_compiler extension, so it
must be self-contained: everything the kernel body relies on is declared here.
#}
#include <type_traits>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>

namespace syclex = sycl::ext::oneapi::experimental;
namespace twi = sycl::ext::oneapi::this_work_item;

// A CUDA __global__ kernel becomes a SYCL free-function nd_range kernel. The
// sub-group size is fixed to the warp size the schedule was generated against,
// which is what makes the warp-level code below well-defined.
#define OEQ_SUBGROUP_SIZE {{ warp_size }}
#define __global__ extern "C" SYCL_EXTERNAL                                    \
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<1>))            \
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::sub_group_size<OEQ_SUBGROUP_SIZE>))

#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__ __restrict

// Occupancy hints have no runtime-compilation equivalent; the work-group size
// is supplied at launch instead.
#define __launch_bounds__(...)

// ---------------------------------------------------------------------------
// Thread / block indexing
// ---------------------------------------------------------------------------
// All generated kernels are launched as 1D nd_ranges, so only .x is meaningful.
struct OeqIndex1D {
    size_t x;
    operator size_t() const { return x; }
};

static inline OeqIndex1D oeq_thread_idx() { return {twi::get_nd_item<1>().get_local_id(0)}; }
static inline OeqIndex1D oeq_block_idx()  { return {twi::get_nd_item<1>().get_group(0)}; }
static inline OeqIndex1D oeq_block_dim()  { return {twi::get_nd_item<1>().get_local_range(0)}; }
static inline OeqIndex1D oeq_grid_dim()   { return {twi::get_nd_item<1>().get_group_range(0)}; }

#define threadIdx oeq_thread_idx()
#define blockIdx  oeq_block_idx()
#define blockDim  oeq_block_dim()
#define gridDim   oeq_grid_dim()

// ---------------------------------------------------------------------------
// Synchronization
// ---------------------------------------------------------------------------
static inline void oeq_syncwarp() {
    sycl::group_barrier(twi::get_sub_group());
}

static inline void oeq_syncthreads() {
    sycl::group_barrier(twi::get_nd_item<1>().get_group());
}

#define __syncthreads() oeq_syncthreads()
#define __threadfence_block() oeq_syncwarp()

// ---------------------------------------------------------------------------
// Warp-level primitives
// ---------------------------------------------------------------------------
template<typename T>
static inline T oeq_shfl_down(T val, int offset) {
    return sycl::shift_group_left(twi::get_sub_group(), val, offset);
}

// ---------------------------------------------------------------------------
// Atomics
// ---------------------------------------------------------------------------
template<typename T>
static inline T oeq_atomic_add(T* address, T val) {
    sycl::atomic_ref<T,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ref(*address);
    return ref.fetch_add(val);
}

// ---------------------------------------------------------------------------
// min / max
// ---------------------------------------------------------------------------
// CUDA provides these as device builtins over mixed integer types. Templating
// on both operands keeps the mixed-width call sites in the templates working.
template<typename A, typename B>
static inline auto oeq_min(A a, B b) -> typename std::common_type<A, B>::type {
    using C = typename std::common_type<A, B>::type;
    return static_cast<C>(a) < static_cast<C>(b) ? static_cast<C>(a) : static_cast<C>(b);
}

template<typename A, typename B>
static inline auto oeq_max(A a, B b) -> typename std::common_type<A, B>::type {
    using C = typename std::common_type<A, B>::type;
    return static_cast<C>(a) > static_cast<C>(b) ? static_cast<C>(a) : static_cast<C>(b);
}

#define min oeq_min
#define max oeq_max

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------
// CUDA's `extern __shared__ char s[]` sizes the allocation at launch. SYCL
// runtime compilation has no dynamic-local-memory equivalent for free-function
// kernels, so each kernel declares a function-scope work_group_static buffer
// sized to the shared memory its own schedule requires.
#define OEQ_DECLARE_SMEM(BYTES)                                                \
    static syclex::work_group_static<char[BYTES]> oeq_smem_buf;                \
    char* s = &oeq_smem_buf[0];
