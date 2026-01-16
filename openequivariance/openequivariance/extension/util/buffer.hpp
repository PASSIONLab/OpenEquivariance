#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;

template<typename ALLOC_T>
class PyDeviceBuffer {
public:
    char* host_ptr;
    char* device_ptr;
    size_t size;

    PyDeviceBuffer(uint64_t size) {
        this->size = size;
        device_ptr = static_cast<char*>(ALLOC_T::gpu_alloc(size));
        host_ptr = nullptr;
    }

    PyDeviceBuffer(py::buffer host_data) {
        const py::buffer_info &info = host_data.request();
        host_ptr = static_cast<char*>(info.ptr);
        size = 1;
        for(int64_t i = 0; i < info.ndim; i++) {
            size *= info.shape[i];
        }
        size *= info.itemsize;

        device_ptr = static_cast<char*>(ALLOC_T::gpu_alloc(size));
        ALLOC_T::copy_host_to_device(host_ptr, device_ptr, size);
    }

    ~PyDeviceBuffer() {
        ALLOC_T::gpu_free(static_cast<void*>(device_ptr));
    }

    void copy_to_host() {
        ALLOC_T::copy_device_to_host(host_ptr, device_ptr, size);
    }

    uint64_t data_ptr() {
        return reinterpret_cast<uint64_t>(device_ptr);
    }
};