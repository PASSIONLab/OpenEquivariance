## Latest Changes

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
