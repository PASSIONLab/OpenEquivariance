#pragma once

#include <array>
#include <cstddef>

/*
* Packs kernel arguments as a list of (pointer, size) pairs.
*
* The CUDA and HIP driver APIs only need the pointer array, but SYCL launches
* free-function kernels with raw (untyped) arguments and therefore also needs
* the size of each argument. Collecting both here keeps a single call shape in
* the backend-independent code.
*
* As with the raw `void*[]` form this replaces, the caller must keep the
* referenced objects alive until the launch has been enqueued.
*/
template <size_t N>
struct KernelArgs {
    std::array<void *, N> ptrs;
    std::array<size_t, N> sizes;

    void **data() { return ptrs.data(); }
    const size_t *arg_sizes() const { return sizes.data(); }
    static constexpr size_t count() { return N; }
};

template <typename... Ts>
inline KernelArgs<sizeof...(Ts)> make_kernel_args(Ts &...args) {
    return KernelArgs<sizeof...(Ts)>{
        {static_cast<void *>(&args)...},
        {sizeof(Ts)...}};
}
