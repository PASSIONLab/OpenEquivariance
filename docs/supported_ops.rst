Supported Operations
==============================

.. list-table::

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

+----------------------------+----------+------------+
| Operation                 | CUDA     | HIP        |
+============================+==========+============+
| UVU                      | ✅        | ✅         |
+----------------------------+----------+------------+
| UVW                      | ✅        | ✅         |
+----------------------------+----------+------------+
| UVU + Convolution        | ✅        | ✅         |
+----------------------------+----------+------------+
| UVW + Convolution        | ✅        | ✅         |
+----------------------------+----------+------------+
| Symmetric Tensor Product | ✅ (beta) | ✅ (beta)  |
+----------------------------+----------+------------+

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
--------------------

We do not (yet) support:

- Mixing different instruction types in the same tensor product. 
- Instruction types besides "uvu" and "uvw".
- Non-trainable instructions: all of your instructions must have weights associated. 

If you have a use case for any of the unsupported features above, let us know.


Multiple Devices and Streams 
--------------------
OpenEquivariance compiles kernels based on the compute capability of the
first visible GPU. On heterogeneous systems, our kernels
will only execute correctly on devices that share the compute capability 
of this first device.

We are working on support for CUDA streams!


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