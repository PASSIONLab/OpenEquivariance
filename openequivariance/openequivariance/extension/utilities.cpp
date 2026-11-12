#ifdef CUDA_BACKEND
#include "backend/backend_cuda.hpp"
#endif

#ifdef HIP_BACKEND
#include "backend/backend_hip.hpp"
#endif

#include "nb_utilities.hpp"

NB_MODULE(OEQ_UTILITIES_MODULE_NAME, m) {
    oeq::bind_utilities(m);
}
