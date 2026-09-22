#include <type_traits>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>

namespace syclex = sycl::ext::oneapi::experimental;
namespace twi = sycl::ext::oneapi::this_work_item;

#define SYCL_SUBGROUP_SIZE {{ warp_size }}
#define __global__ extern "C" SYCL_EXTERNAL                                    \
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<1>))            \
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::sub_group_size<SYCL_SUBGROUP_SIZE>))

#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__ __restrict

#define __launch_bounds__(...)

struct SyclIndex1D {
    size_t x;
    operator size_t() const { return x; }
};

static inline SyclIndex1D _sycl_thread_idx() { return {twi::get_nd_item<1>().get_local_id(0)}; }
static inline SyclIndex1D _sycl_block_idx()  { return {twi::get_nd_item<1>().get_group(0)}; }
static inline SyclIndex1D _sycl_block_dim()  { return {twi::get_nd_item<1>().get_local_range(0)}; }
static inline SyclIndex1D _sycl_grid_dim()   { return {twi::get_nd_item<1>().get_group_range(0)}; }

#define threadIdx _sycl_thread_idx()
#define blockIdx  _sycl_block_idx()
#define blockDim  _sycl_block_dim()
#define gridDim   _sycl_grid_dim()

static inline void _sycl_syncwarp() {
    sycl::group_barrier(twi::get_sub_group());
}

static inline void _sycl_syncthreads() {
    sycl::group_barrier(twi::get_nd_item<1>().get_group());
}

#define __syncthreads() _sycl_syncthreads()

template<typename T>
static inline T _sycl_shfl_down(T val, int offset) {
    return sycl::shift_group_left(twi::get_sub_group(), val, offset);
}

template<typename T>
static inline T _sycl_atomic_add(T* address, T val) {
    sycl::atomic_ref<T,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ref(*address);
    return ref.fetch_add(val);
}

template<typename A, typename B>
static inline auto _sycl_min(A a, B b) -> typename std::common_type<A, B>::type {
    using C = typename std::common_type<A, B>::type;
    return static_cast<C>(a) < static_cast<C>(b) ? static_cast<C>(a) : static_cast<C>(b);
}

template<typename A, typename B>
static inline auto _sycl_max(A a, B b) -> typename std::common_type<A, B>::type {
    using C = typename std::common_type<A, B>::type;
    return static_cast<C>(a) > static_cast<C>(b) ? static_cast<C>(a) : static_cast<C>(b);
}

#define min _sycl_min
#define max _sycl_max

// SYCL runtime compilation has no dynamic-local-memory equivalent for
// free-function kernels, so each kernel declares a function-scope buffer sized
// to the shared memory its own schedule requires.
#define SYCL_DECLARE_SMEM(BYTES)                                                \
    static syclex::work_group_static<char[BYTES]> _sycl_smem_buf;                \
    char* s = &_sycl_smem_buf[0];
