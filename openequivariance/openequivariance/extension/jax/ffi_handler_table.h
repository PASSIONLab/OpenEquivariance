#pragma once

#include <stdint.h>

#if defined(_WIN32)
#define OEQ_FFI_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define OEQ_FFI_EXPORT __attribute__((visibility("default")))
#else
#define OEQ_FFI_EXPORT
#endif

// This is a stable, Python-free view of XLA FFI handlers. Consumers must
// reject a table with an unknown ABI version or entry layout.
#ifdef __cplusplus
extern "C" {
#endif

#define OEQ_FFI_ABI_VERSION 2

#define OEQ_FFI_TRAIT_COMMAND_BUFFER_COMPATIBLE (1u << 0)

typedef struct OeqFfiHandler {
    const char* name;
    void* instantiate;
    void* prepare;
    void* initialize;
    void* execute;
    uint32_t traits;
} OeqFfiHandler;

typedef struct OeqFfiHandlerTable {
    uint32_t abi_version;
    uint32_t handler_count;
    const OeqFfiHandler* handlers;
} OeqFfiHandlerTable;

OEQ_FFI_EXPORT const OeqFfiHandlerTable* oeq_ffi_handler_table();

#ifdef __cplusplus
}
#endif
