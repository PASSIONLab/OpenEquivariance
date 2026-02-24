## Latest Changes

### v0.6.1 (2025-02-23) 
OpenEquivariance v0.6.2 brings long-needed improvements to the
PyTorch frontend. We strongly encourage all users to upgrade
to PyTorch 2.10 and OEQ v0.6.2.

**Added**:
- OpenEquivariance triggers a build of the CUDA extension module
  at `pip` install time and will use this precompiled extension if
  the user has PyTorch >=2.10 installed. If PyTorch <2.10 is installed,
  the JIT-compiled extension is used instead.
- PyTorch ABI support for C++ backend, using new features in PyTorch
  2.10 to support stable, forward-compatible ahead-of-time 
  extensions.
- Dropped support for TorchBind classes and a new kernel cache in its
  place, which greatly improves flexibility for automatic mixed precision
  and AOTI compilation. An inference test in C++ is included. 
- `openequivariance_extjax` has a version number that synchronizes with
  the main `openequivariance` package; ensure the two packages stay in sync.

**Fixed**:
- `torch.to()` is now called when either `TensorProduct`
  or `TensorProductConv` is a submodule of another PyTorch 
  module. 


### v0.5.4 (2025-02-01) 
Improvements to JAX frontend.

**Added**:
- Jacobian Vector Products (JVP) 
  for both `TensorProduct` and `TensorProductConv` via custom primitives, in addition to VJP.
- Arbitrary higher-order derivatives in JAX.
- JAX JIT support; in particular, support for
  Phonon Fine Tuning in [Nequix](https://github.com/atomicarchitects/nequix).

**Fixed**:
- Zero'd all output buffers in the backwards and double-backwards implementations of convolution
before calling kernels. 

### v0.5.1-0.5.3 (2025-02-01) 
Minor bugfixes related to packaging and JAX. 

### v0.5.0 (2025-12-25) 
JAX support is now available in 
OpenEquivariance for BOTH NVIDIA and
AMD GPUs! See the 
[documentation](https://passionlab.github.io/OpenEquivariance/)
and README.md for instructions on installation
and usage. 

Minor changes:
- Defer error reporting when CUDA is not available
  to the first library usage in code, not library load. 

### v0.4.1 (2025-09-04)
Minor update, fixes a bug loading JIT-compiled modules
with PyTorch 2.9.

### v0.4.0 (2025-08-14)
This release adds a benchmark against
FlashTP, exposes weight reordering functions
for e3nn compatibility, adds input validation,
and provides rudimentary support for PyTorch
automatic mixed precision (AMP). Our fused,
JIT-compiled kernels exhibit up to 2x speedup
over FlashTP!

**Added**:
1. Both `TensorProduct` and `TensorProductConv`
now have the methods `reoder_weights_from_e3nn`
and `reorder_weights_to_e3nn`. These convert
the buffer of trainable weights from / to e3nn's
canonical ordering. See the API page for usage
details.
2. If you have FlashTP installed, see our 
documentation ("Tests and Benchmarks" page) 
to benchmark FlashTP against OpenEquivariance. 
3. Tensor product inputs with incorrect sizes or 
datatypes now trigger clear errors in advance of
execution.
4. OpenEquivariance now has some support for
automatic mixed precision (AMP), but only if 
`TensorProduct` / `TensorProductConv` objects 
are constructed with `float32` precision for
both `irrep_dtype` and `weight_dtype`.

**Fixed / Enhanced**:
1. Added additional fake functions to remove 
warnings from TorchBind.
2. Removed bloat from benchmarking code. 

### v0.3.0 (2025-06-22)
This release includes bugfixes and new opaque operations that
compose with `torch.compile` 
for PT2.4-2.7. These will be unnecessary for PT2.8+. 

**Added**:
1. Opaque variants of major operations 
   via PyTorch `custom_op` declarations. These
   functions cannot be traced through and fail
   for JITScript / AOTI. They are shims that
   enable composition with `torch.compile`
   pre-PT2.8.
2. `torch.load`/`torch.save` functionality
   that, without `torch.compile`, is portable
   across GPU architectures.
3. `.to()` support to move `TensorProduct`
   and `TensorProductConv` between devices or
   change datatypes.

**Fixed**:
1. Gracefully records an error if `libpython.so`
   is not linked against C++ extension.
2. Resolves Kahan summation / various other bugs
   for HIP at O3 compiler-optimization level. 
3. Removes multiple contexts spawning for GPU 0
   when multiple devices are used.
4. Zero-initialized gradient buffers to prevent
   backward pass garbage accumulation. 

### v0.2.0 (2025-06-09) 

Our first stable release, **v0.2.0**, introduces several new features. Highlights include:

1. Full HIP support for all kernels.
2. Support for `torch.compile`, JITScript and export, preliminary support for AOTI.
3. Faster double backward performance for training.
4. Ability to install versioned releases from PyPI.
5. Support for CUDA streams and multiple devices.
6. An extensive test suite and newly released [documentation](https://passionlab.github.io/OpenEquivariance/).

If you successfully run OpenEquivariance on a GPU model not listed [here](https://passionlab.github.io/OpenEquivariance/tests_and_benchmarks/), let us know! We can add your name to the list.

---

**Known issues:**

- Kahan summation is broken on HIP – fix planned.
- FX + Export + Compile has trouble with PyTorch dynamo; fix planned.
- AOTI broken on PT <2.8; you need the nightly build due to incomplete support for TorchBind in prior versions.

### v0.1.0 (2025-01-23) 
Initial Github release with preprint. 
