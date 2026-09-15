#pragma once

#include <cstdint>
#include <string>
#include <vector>

template<typename JIT_IMPL>
class __attribute__ ((visibility ("default"))) JITFactorizedProjectedImpl {
public:
    JIT_IMPL jit;

    static std::vector<std::string> kernel_entry_points() {
        return {
            "oeq_projected_forward", "oeq_projected_forward_jvp",
            "oeq_projected_backward",
            "oeq_projected_backward_jvp"};
    }

    static std::vector<std::vector<int>> kernel_template_parameters() {
        return std::vector<std::vector<int>>(kernel_entry_points().size());
    }

    JITFactorizedProjectedImpl(
            std::string source, int64_t num_threads,
            int64_t logical_cohort_width, int64_t shared_memory_bytes) :
        jit(std::move(source)),
        num_threads_(num_threads),
        logical_cohort_width_(logical_cohort_width),
        shared_memory_bytes_(shared_memory_bytes) {
        jit.compile(kernel_entry_points(), kernel_template_parameters());
    }

    void forward(
            int64_t node_count, int64_t edge_count, int64_t channels,
            void* x, void* sh, void* weights, void* senders, void* row_ptr,
            void* out, Stream stream) {
        void* args[] = {
            &node_count, &edge_count, &x, &sh, &weights, &senders, &row_ptr, &out};
        execute(0, node_count * channels, args, stream);
    }

    void forward_jvp(
            int64_t node_count, int64_t edge_count, int64_t channels,
            void* x, void* sh, void* weights, void* senders, void* row_ptr,
            void* tx, void* tsh, void* tweights, void* out, Stream stream) {
        void* args[] = {
            &node_count, &edge_count, &x, &sh, &weights, &senders, &row_ptr,
            &tx, &tsh, &tweights, &out};
        execute(1, node_count * channels, args, stream);
    }

    void backward(
            int64_t node_count, int64_t edge_count,
            void* x, void* sh, void* weights, void* senders, void* receivers,
            void* dout, void* dx, void* dsh, void* dweights, Stream stream) {
        void* args[] = {
            &node_count, &edge_count, &x, &sh, &weights, &senders, &receivers,
            &dout, &dx, &dsh, &dweights};
        execute(2, edge_count * logical_cohort_width_, args, stream);
    }

    void backward_jvp(
            int64_t node_count, int64_t edge_count,
            void* x, void* sh, void* weights, void* senders, void* receivers,
            void* dout, void* tx, void* tsh, void* tweights, void* tdout,
            void* tdx, void* tdsh, void* tdweights, Stream stream) {
        void* args[] = {
            &node_count, &edge_count, &x, &sh, &weights, &senders, &receivers, &dout,
            &tx, &tsh, &tweights, &tdout, &tdx, &tdsh, &tdweights};
        execute(3, edge_count * logical_cohort_width_, args, stream);
    }

private:
    int64_t num_threads_;
    int64_t logical_cohort_width_;
    int64_t shared_memory_bytes_;

    void execute(int kernel_index, int64_t work_items, void* args[], Stream stream) {
        if (work_items == 0)
            return;
        const int64_t blocks =
            (work_items + num_threads_ - 1) / num_threads_;
        jit.execute(
            kernel_index, args,
            with_stream(
                KernelLaunchConfig(blocks, num_threads_, shared_memory_bytes_),
                stream));
    }
};
