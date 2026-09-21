#pragma once

#include <array>
#include <cstddef>

template <size_t N>
class KernelArgs {
    std::array<void *, N> ptrs_;
    std::array<size_t, N> sizes_;

public:
    template <typename... Ts>
    explicit KernelArgs(Ts &...args)
        : ptrs_{static_cast<void *>(&args)...}, sizes_{sizeof(Ts)...} {
        static_assert(sizeof...(Ts) == N, "argument count mismatch");
    }

    void **data() { return ptrs_.data(); }
    const size_t *arg_sizes() const { return sizes_.data(); }
    static constexpr size_t count() { return N; }
};

template <typename... Ts>
KernelArgs(Ts &...) -> KernelArgs<sizeof...(Ts)>;
