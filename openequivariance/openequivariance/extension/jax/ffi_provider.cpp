#include <cstddef>
#include <cstring>
#include <iostream>

#include "ffi_handler_table.h"
#include "xla/ffi/api/c_api.h"

extern "C" OEQ_FFI_EXPORT int RegisterFFi(
        const XLA_FFI_Api* api,
        const char* platform_name) {
    constexpr std::size_t kHandlerRegistrationSize =
        offsetof(XLA_FFI_Api, XLA_FFI_Handler_Register) +
        sizeof(api->XLA_FFI_Handler_Register);
    if (api == nullptr || platform_name == nullptr ||
        api->struct_size < kHandlerRegistrationSize ||
        api->XLA_FFI_Handler_Register == nullptr ||
        api->XLA_FFI_Error_GetMessage == nullptr ||
        api->XLA_FFI_Error_Destroy == nullptr) {
        return 1;
    }

    const OeqFfiHandlerTable* handlers = oeq_ffi_handler_table();
    if (handlers == nullptr || handlers->abi_version != OEQ_FFI_ABI_VERSION) {
        return 1;
    }
    for (uint32_t index = 0; index < handlers->handler_count; ++index) {
        const OeqFfiHandler& handler = handlers->handlers[index];
        if (handler.execute == nullptr) {
            std::cerr << "OpenEquivariance FFI target " << handler.name
                      << " has no execute handler" << std::endl;
            return 1;
        }

        XLA_FFI_Handler_Register_Args args{};
        args.struct_size = XLA_FFI_Handler_Register_Args_STRUCT_SIZE;
        args.name = {handler.name, std::strlen(handler.name)};
        args.platform = {platform_name, std::strlen(platform_name)};
        args.bundle = {
            reinterpret_cast<XLA_FFI_Handler*>(handler.instantiate),
            reinterpret_cast<XLA_FFI_Handler*>(handler.prepare),
            reinterpret_cast<XLA_FFI_Handler*>(handler.initialize),
            reinterpret_cast<XLA_FFI_Handler*>(handler.execute),
        };
        args.traits = static_cast<XLA_FFI_Handler_Traits>(handler.traits);

        XLA_FFI_Error* error = api->XLA_FFI_Handler_Register(&args);
        if (error == nullptr) {
            std::cout << "[OpenEquivariance] Registered FFI target "
                      << handler.name << " for " << platform_name << std::endl;
            continue;
        }

        XLA_FFI_Error_GetMessage_Args message_args{};
        message_args.struct_size = XLA_FFI_Error_GetMessage_Args_STRUCT_SIZE;
        message_args.error = error;
        api->XLA_FFI_Error_GetMessage(&message_args);
        std::cerr << "OpenEquivariance failed to register FFI target "
                  << handler.name << ": "
                  << (message_args.message == nullptr
                          ? "unknown XLA FFI error"
                          : message_args.message)
                  << std::endl;

        XLA_FFI_Error_Destroy_Args destroy_args{};
        destroy_args.struct_size = XLA_FFI_Error_Destroy_Args_STRUCT_SIZE;
        destroy_args.error = error;
        api->XLA_FFI_Error_Destroy(&destroy_args);
        return 1;
    }
    return 0;
}
