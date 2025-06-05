Running MACE and Nequip
==============================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

MACE
----

We have modified MACE to use our accelerated kernels instead
of the standard e3nn backend. Here are the steps to replicate
our MACE benchmark:

1. Install ``oeq`` and our modified version of MACE via

   .. code-block:: bash

      pip uninstall mace-torch
      pip install git+https://github.com/vbharadwaj-bk/mace_oeq_integration.git@oeq_experimental

2. Download the ``carbon.xyz`` data file, available at 
   `<https://portal.nersc.gov/project/m1982/equivariant_nn_graphs/>`_.

   This graph has 158K edges. With the original e3nn backend, you would need a GPU with 80GB
   of memory to run the experiments. ``oeq`` provides a memory-efficient equivariant convolution,
   so we expect the test to succeed.

3. Benchmark OpenEquivariance:

   .. code-block:: bash

      python tests/mace_driver.py carbon.xyz -o outputs/mace_tests -i oeq

4. If you have a GPU with 80GB of memory *or* supply a smaller molecular graph
   as the input file, you can run the full benchmark that includes ``e3nn`` and ``cue``:

   .. code-block:: bash

      python tests/mace_driver.py carbon.xyz -o outputs/mace_tests -i e3nn cue oeq

Nequip
------
See the 
`official Nequip documentation <https://nequip.readthedocs.io/en/latest/guide/accelerations/openequivariance.html>`_
to use OpenEquivariance with Nequip.
