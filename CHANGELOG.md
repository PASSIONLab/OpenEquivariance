## Latest Changes

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
