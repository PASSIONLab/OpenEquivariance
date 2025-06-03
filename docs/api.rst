OpenEquivariance API
==============================

OpenEquivariance exposes two key classes: :py:class:`openequivariance.TensorProduct`, which replaces
``o3.TensorProduct`` from e3nn, and :py:class:`openequivariance.TensorProductConv`, which fuses
the CG tensor product with a subsequent graph convolution. Initializing either class triggers
JIT compilation of a custom kernel; this takes up to a few seconds. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. autoclass:: openequivariance.TensorProduct
    :members:
    :undoc-members:
    :exclude-members: name

..
    .. automodule:: openequivariance
        :members:
        :undoc-members:
