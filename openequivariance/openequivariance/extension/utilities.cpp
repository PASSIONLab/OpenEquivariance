#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"

#ifdef CUDA_BACKEND
#include "backend/backend_cuda.hpp"
#endif

#ifdef HIP_BACKEND
#include "backend/backend_hip.hpp"
#endif

namespace nb = nanobind;

NB_MODULE(OEQ_UTILITIES_MODULE_NAME, m) {
    if (nb::handle existing = nb::type<DeviceProp>()) {
        m.attr("DeviceProp") = existing;
    } else {
        nb::class_<DeviceProp>(m, "DeviceProp")
            .def(nb::init<int>())
            .def_ro("name", &DeviceProp::name)
            .def_ro("warpsize", &DeviceProp::warpsize)
            .def_ro("major", &DeviceProp::major)
            .def_ro("minor", &DeviceProp::minor)
            .def_ro("multiprocessorCount", &DeviceProp::multiprocessorCount)
            .def_ro("maxSharedMemPerBlock", &DeviceProp::maxSharedMemPerBlock);
    }

    if (nb::handle existing = nb::type<GPUTimer>()) {
        m.attr("GPUTimer") = existing;
    } else {
        nb::class_<GPUTimer>(m, "GPUTimer")
            .def(nb::init<>())
            .def("start", &GPUTimer::start)
            .def("stop_clock_get_elapsed", &GPUTimer::stop_clock_get_elapsed)
            .def("clear_L2_cache", &GPUTimer::clear_L2_cache);
    }
}
