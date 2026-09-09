"""Names of JAX FFI handlers provided by the native extension."""

TENSOR_PRODUCT_TARGETS = (
    "tp_forward",
    "tp_backward",
    "tp_double_backward",
)

CONVOLUTION_TARGETS = (
    "conv_forward",
    "conv_backward",
    "conv_double_backward",
)

FFI_TARGETS = TENSOR_PRODUCT_TARGETS + CONVOLUTION_TARGETS
