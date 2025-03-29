#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef CUDA
#include "backend_cuda.hpp"
#include "group_mm_cuda.hpp"
using JITKernel = CUJITKernel;
using Allocator = CUDA_Allocator;
using GroupMM_F32 = GroupMMCUDA<float>; 
using GroupMM_F64 = GroupMMCUDA<double>; 
#endif

#ifdef HIP
#include "backend_hip.hpp"
using JITKernel = HIPJITKernel;
using Allocator = HIP_Allocator;
using GroupMM_F32 = GroupMMHIP<float>; 
using GroupMM_F64 = GroupMMHIP<double>; 
#endif

#include "buffer.hpp"
#include "tensorproducts.hpp"
#include "convolution.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(kernel_wrapper, m) {
    //=========== Batch tensor products =========
    py::class_<GenericTensorProductImpl>(m, "GenericTensorProductImpl")
        .def("exec_tensor_product", &GenericTensorProductImpl::exec_tensor_product_device_rawptrs)
        .def("backward", &GenericTensorProductImpl::backward_device_rawptrs);
    py::class_<JITTPImpl<JITKernel>, GenericTensorProductImpl>(m, "JITTPImpl")
        .def(py::init<std::string, KernelLaunchConfig&, KernelLaunchConfig&>());

    //============= Convolutions ===============
    py::class_<ConvolutionImpl>(m, "ConvolutionImpl")
        .def("exec_conv_rawptrs", &ConvolutionImpl::exec_conv_rawptrs)
        .def("backward_rawptrs", &ConvolutionImpl::backward_rawptrs);
    py::class_<JITConvImpl<JITKernel>, ConvolutionImpl>(m, "JITConvImpl")
        .def(py::init<std::string, KernelLaunchConfig&, KernelLaunchConfig&>());

    //============= Utilities ===============
    py::class_<KernelLaunchConfig>(m, "KernelLaunchConfig")
        .def(py::init<>())
        .def_readwrite("num_blocks", &KernelLaunchConfig::num_blocks)
        .def_readwrite("warp_size", &KernelLaunchConfig::warp_size)
        .def_readwrite("num_threads", &KernelLaunchConfig::num_threads)
        .def_readwrite("smem", &KernelLaunchConfig::smem);

    py::class_<DeviceProp>(m, "DeviceProp")
        .def(py::init<int>())
        .def_readonly("name", &DeviceProp::name)
        .def_readonly("warpsize", &DeviceProp::warpsize)
        .def_readonly("major", &DeviceProp::major)
        .def_readonly("minor", &DeviceProp::minor)
        .def_readonly("multiprocessorCount", &DeviceProp::multiprocessorCount)
        .def_readonly("maxSharedMemPerBlock", &DeviceProp::maxSharedMemPerBlock); 

    py::class_<PyDeviceBuffer<Allocator>>(m, "DeviceBuffer")
        .def(py::init<uint64_t>())
        .def(py::init<py::buffer>())
        .def("copy_to_host", &PyDeviceBuffer<Allocator>::copy_to_host)
        .def("data_ptr", &PyDeviceBuffer<Allocator>::data_ptr);

    py::class_<GPUTimer>(m, "GPUTimer")
        .def(py::init<>())
        .def("start", &GPUTimer::start)
        .def("stop_clock_get_elapsed", &GPUTimer::stop_clock_get_elapsed)
        .def("clear_L2_cache", &GPUTimer::clear_L2_cache);

    py::class_<GroupMM_F32>(m, "GroupMM_F32")
        .def(py::init<int>())
        .def("group_gemm", &GroupMM_F32::group_gemm_intptr);
}