OpenEquivariance API
==============================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

OpenEquivariance exposes two key classes: :py:class:`openequivariance.TensorProduct`, which replaces
``o3.TensorProduct`` from e3nn, and :py:class:`openequivariance.TensorProductConv`, which fuses
the CG tensor product with a subsequent graph convolution. Initializing either class triggers
JIT compilation of a custom kernel, which can take a few seconds. 

.. autoclass:: openequivariance.TensorProduct
    :members:
    :undoc-members:
    :exclude-members: name

.. autoclass:: openequivariance.TensorProductConv
    :members:
    :undoc-members:
    :exclude-members: name

