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

PyTorch API
------------------------

.. autoclass:: openequivariance.TensorProduct
    :members: forward, reorder_weights_from_e3nn, reorder_weights_to_e3nn, to
    :undoc-members:
    :exclude-members: name

.. autoclass:: openequivariance.TensorProductConv
    :members: forward, reorder_weights_from_e3nn, reorder_weights_to_e3nn, to
    :undoc-members:
    :exclude-members: name

.. autofunction:: openequivariance._torch_to_oeq_dtype

.. autofunction:: openequivariance._torch_ext_so_path

JAX API
------------------------
The JAX API consists of ``TensorProduct`` and ``TensorProductConv``
classes that behave identically to their PyTorch counterparts. These classes
do not conform exactly to the e3nn-jax API, but perform the same computation.

If you plan to use ``oeq.jax`` without PyTorch installed, 
you need to set ``OEQ_NOTORCH=1`` in your local environment (within Python,
``os.environ["OEQ_NOTORCH"] = 1``). For the moment, we require this to avoid 
breaking the PyTorch version of OpenEquivariance.


.. autoclass:: openequivariance.jax.TensorProduct
    :members: forward, reorder_weights_from_e3nn, reorder_weights_to_e3nn
    :undoc-members:
    :exclude-members:

.. autoclass:: openequivariance.jax.TensorProductConv
    :members: forward, reorder_weights_from_e3nn, reorder_weights_to_e3nn
    :undoc-members:
    :exclude-members: 

Common API
---------------------

.. autoclass:: openequivariance.TPProblem
    :members:
    :undoc-members:

API Identical to e3nn
---------------------

These remaining API members are identical to the corresponding
objects in ``e3nn.o3``. You can freely mix these objects from
both packages. 

.. autoclass:: openequivariance.Irreps
    :members:
    :undoc-members: