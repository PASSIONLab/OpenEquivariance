#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

template <size_t N>
class KernelArgs {
    std::array<void *, N> ptrs_;
    std::array<size_t, N> sizes_;

    template <typename... Ts>
    static constexpr bool is_arg_pack =
        sizeof...(Ts) == N &&
        !(sizeof...(Ts) == 1 &&
          (std::is_same_v<std::remove_cv_t<Ts>, KernelArgs> && ...));

public:
    template <typename... Ts, typename = std::enable_if_t<is_arg_pack<Ts...>>>
    explicit constexpr KernelArgs(Ts &...args) noexcept
        : ptrs_{const_cast<void *>(
              static_cast<const volatile void *>(&args))...},
          sizes_{sizeof(Ts)...} {}

    constexpr void **data() noexcept { return ptrs_.data(); }
    constexpr const size_t *arg_sizes() const noexcept { return sizes_.data(); }
    static constexpr size_t count() noexcept { return N; }
};

template <typename... Ts>
KernelArgs(Ts &...) -> KernelArgs<sizeof...(Ts)>;
