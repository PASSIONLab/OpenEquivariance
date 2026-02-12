#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <unordered_map>
#include <stdexcept>

#ifdef CUDA_BACKEND
    #include "backend_cuda.hpp"
    #include "group_mm_cuda.hpp"
    using JITKernel = CUJITKernel;
    using GPU_Allocator = CUDA_Allocator;

    template<typename T>
    using GroupMM = GroupMMCUDA<T>; 
#endif

#ifdef HIP_BACKEND
    #include "backend_hip.hpp"
    #include "group_mm_hip.hpp"
    using JITKernel = HIPJITKernel;
    using GPU_Allocator = HIP_Allocator;

    template<typename T>
    using GroupMM = GroupMMHIP<T>; 
#endif

#include "tensorproducts.hpp"
#include "convolution.hpp"

using namespace std;
namespace py=pybind11;

PYBIND11_MODULE(generic_module, m) {
    py::class_<GroupMM<float>>(m, "GroupMM_F32")
        .def(py::init<int, int>())
        .def("group_gemm", &GroupMM<float>::group_gemm_intptr);
    py::class_<GroupMM<double>>(m, "GroupMM_F64")
        .def(py::init<int, int>())
        .def("group_gemm", &GroupMM<double>::group_gemm_intptr);

    py::class_<DeviceProp>(m, "DeviceProp")
        .def(py::init<int>())
        .def_readonly("name", &DeviceProp::name)
        .def_readonly("warpsize", &DeviceProp::warpsize)
        .def_readonly("major", &DeviceProp::major)
        .def_readonly("minor", &DeviceProp::minor)
        .def_readonly("multiprocessorCount", &DeviceProp::multiprocessorCount)
        .def_readonly("maxSharedMemPerBlock", &DeviceProp::maxSharedMemPerBlock); 

    py::class_<GPUTimer>(m, "GPUTimer")
        .def(py::init<>())
        .def("start", &GPUTimer::start)
        .def("stop_clock_get_elapsed", &GPUTimer::stop_clock_get_elapsed)
        .def("clear_L2_cache", &GPUTimer::clear_L2_cache);
}