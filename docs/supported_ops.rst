Supported Operations
====================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

.. list-table:: 
   :widths: 50 25 25
   :header-rows: 1

   * - Operation 
     - CUDA 
     - HIP 
   * - UVU 
     - ✅
     - ✅
   * - UVW 
     - ✅
     - ✅
   * - UVU + Convolution
     - ✅
     - ✅
   * - UVW + Convolution
     - ✅
     - ✅
   * - Symmetric Tensor Product 
     - ✅ (beta)
     - ✅ (beta)

e3nn supports a variety of connection modes for CG tensor products. We support 
two that are commonly used in equivariant graph neural networks:
"uvu" and "uvw". Our JIT compiled kernels should handle:

1. Pure "uvu" tensor products, which are most efficient when the input with higher
   multiplicities is the first argument. Our results are identical to e3nn when irreps in
   the second input have multiplicity 1, and otherwise identical up to a reordering
   of the input weights.

2. Pure "uvw" tensor products, which are currently more efficient when the input with
   higher multiplicities is the first argument. Our results are identical to e3nn up to a reordering
   of the input weights. 

Our code includes correctness checks, but the configuration space is large. If you notice
a bug, let us know in a GitHub issue. We'll try our best to correct it or document the problem here.

Unsupported Tensor Product Configurations 
-----------------------------------------

We do not (yet) support:

- Mixing different instruction types in the same tensor product. 
- Instruction types besides "uvu" and "uvw".
- Non-trainable instructions: all of your instructions must have weights associated. 

If you have a use case for any of the unsupported features above, let us know.


Torch Save / Load 
---------------------------------------------------
OpenEquivariance's ``TensorProduct`` / ``TensorProductConv`` modules 
can be saved via ``torch.save`` and restored via ``torch.load``.
You must call ``import openequivariance`` before attempting to load, i.e.

.. code-block::

    import torch
    import openequivariance
    module = torch.load("my_module_with_tp.pt")

If you do NOT use ``torch.compile`` or ``torch.export``, these modules 
can be loaded on a platform with a distinct GPU architecture from the saving
platform. In this case, kernels are recompiled dynamically. After compilation, 
a module may only be used on a platform with GPU architecture identical 
to the machine that saved it. 


Compilation with JITScript, Export, and AOTInductor
---------------------------------------------------

OpenEquivariance supports model compilation with
``torch.compile``, JITScript, ``torch.export``, and AOTInductor. 
Demo the C++ model exports with

.. code-block:: bash

    pytest tests/export_test.py 


NOTE: the AOTInductor test (and possibly export) fail 
unless you are using a Nightly
build of PyTorch past 4/10/2025 due to incomplete support for 
TorchBind in earlier versions.

Multiple Devices and Streams 
----------------------------
OpenEquivariance compiles kernels based on the compute capability of the
first visible GPU. On heterogeneous systems, our kernels
will only execute correctly on devices that share the compute capability 
of this first device.

Both CUDA and HIP backends support GPU streams.
See PyTorch usage details `here <https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams>`_.

Symmetric Contraction (Beta)
----------------------------

We have recently added beta support for symmetric
contraction acceleration. This primitive: 

- Is specific to MACE
- Requires e3nn as a dependency. 
- Currently has no support for compile / export

As a result, we do not expose it in the package
toplevel. You can use our implementation by running

.. code-block::

    from openequivariance.implementations.symmetric_contraction import SymmetricContraction as OEQSymmetricContraction