"""Names of JAX FFI kernel targets."""

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

FACTORIZED_TARGET = "factorized_projected"
FFI_TARGETS = TENSOR_PRODUCT_TARGETS + CONVOLUTION_TARGETS + (FACTORIZED_TARGET,)
