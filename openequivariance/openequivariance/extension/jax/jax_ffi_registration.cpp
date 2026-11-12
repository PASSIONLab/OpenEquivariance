#include "nanobind/nanobind.h"

#include "ffi_handler_table.h"

#ifdef CUDA_BACKEND
#include "backend/backend_cuda.hpp"
#endif

#ifdef HIP_BACKEND
#include "backend/backend_hip.hpp"
#endif

#include "nb_utilities.hpp"

namespace nb = nanobind;

namespace {

bool IsHip() {
#ifdef HIP_BACKEND
    return true;
#else
    return false;
#endif
}

void AddStage(nb::dict& stages, const char* name, void* handler) {
    if (handler != nullptr) {
        stages[name] = nb::capsule(handler);
    }
}

}  // namespace

NB_MODULE(OEQ_JAX_MODULE_NAME, m) {
    m.def("registrations", []() {
        nb::dict registrations;
        const auto* handlers = oeq_ffi_handler_table();
        for (uint32_t index = 0; index < handlers->handler_count; ++index) {
            const auto& handler = handlers->handlers[index];
            nb::dict stages;
            AddStage(stages, "instantiate", handler.instantiate);
            AddStage(stages, "prepare", handler.prepare);
            AddStage(stages, "initialize", handler.initialize);
            AddStage(stages, "execute", handler.execute);
            registrations[handler.name] = stages;
        }
        return registrations;
    });
    m.def("is_hip", &IsHip);

    oeq::bind_utilities(m);
}
