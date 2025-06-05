OpenEquivariance API
==============================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

OpenEquivariance exposes two key classes: :py:class:`openequivariance.TensorProduct`, which replaces
``o3.TensorProduct`` from e3nn, and :py:class:`openequivariance.TensorProductConv`, which fuses
the CG tensor product with a subsequent graph convolution. Initializing either class triggers
JIT compilation of a custom kernel, which can take a few seconds. 

Both classes require a configuration object specified 
by :py:class:`openequivariance.TPProblem`, which has a constructor
almost identical to ``o3.TensorProduct``. 
We recommend reading the `e3nn documentation <https://docs.e3nn.org/en/latest/>`_ before
trying our code. OpenEquivariance cannot accelerate all tensor products; see 
:doc:`this page </supported_ops>` for a list of supported configurations.

.. autoclass:: openequivariance.TensorProduct
    :members:
    :undoc-members:
    :exclude-members: name

.. autoclass:: openequivariance.TensorProductConv
    :members:
    :undoc-members:
    :exclude-members: name

.. autoclass:: openequivariance.TPProblem
    :members:
    :undoc-members:

.. autofunction:: openequivariance.torch_to_oeq_dtype

.. autofunction:: openequivariance.torch_ext_so_path

API Identical to e3nn
---------------------

These remaining API members are identical to the corresponding
objects in ``e3nn.o3``. You can freely mix these objects from
both packages. 

.. autoclass:: openequivariance.Irreps
    :members:
    :undoc-members: