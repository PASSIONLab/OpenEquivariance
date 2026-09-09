# OpenEquivariance JAX Extension

The JAX extension module for OpenEquivariance.

## XLA FFI provider

The `libjcn_ffi_openequivariance.so` Bazel target builds a Python-independent
FFI provider against the XLA and CUDA repositories selected by the caller's
Bazel configuration:

```text
bazel build @openequivariance_src//openequivariance_extjax:libjcn_ffi_openequivariance.so
```

The handler implementation owns one ABI-versioned table with slots for every
XLA stage (`instantiate`, `prepare`, `initialize`, and `execute`) plus each
target's traits. The nanobind extension reads that table and publishes only
stages that have a handler. The shared library exports `RegisterFFi(const
XLA_FFI_Api*, const char*)`; a runtime loader obtains the API from its PJRT
plugin and calls this entry point. The provider uses
`XLA_FFI_Handler_Register` to register every target as a complete handler
bundle, passing every non-null stage and the traits from the table. JAX
registration uses the same table.
