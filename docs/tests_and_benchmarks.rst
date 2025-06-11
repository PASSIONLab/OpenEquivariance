Tests and Benchmarks
==============================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

OpenEquivariance is equipped with a comprehensive suite of tests
and benchmarking utilities. You'll need some additional dependencies to run 
these; we provide instructions below. 

We recommend you clone our repository and use an editable install to run tests
and benchmarks. 

You can still test our code with a non-editable install; just 
download the test folder and install the non-editable package and the dependencies with:

.. code-block:: bash

    pip install openequivariance[dev,bench]

Correctness
------------------------------
To set up the editable install and run the entire testsuite, use: 

.. code-block:: bash

    git clone https://github.com/PASSIONLab/OpenEquivariance 
    cd OpenEquivariance
    pip install -e .[dev] 
    pytest 

Browse the ``tests`` directory to run specific components. 


Replicating our Benchmarks
------------------------------
We conducted our benchmarks on an NVIDIA A100-SXM-80GB GPU at Lawrence Berkeley National Laboratory. 
Your results may differ a different GPU. The following invocations run the experiments
and generate plots from our paper.

.. code-block:: bash

    git clone https://github.com/PASSIONLab/OpenEquivariance 
    cd OpenEquivariance
    pip install -e .[bench] 
    python tests/benchmark.py -o outputs/uvu uvu --plot
    python tests/benchmark.py -o outputs/uvw uvw --plot
    python tests/benchmark.py -o outputs/roofline roofline --plot
    python tests/benchmark.py -o outputs/conv conv --plot --data data/molecular_structures
    python tests/benchmark.py -o outputs/kahan_conv kahan_conv --data data/molecular_structures/

If your GPU has limited memory, try the ``--limited-memory`` flag 
to disable some expensive tests and / or reduce the batch size with ``-b``. 
Run ``python tests/benchmark.py --help`` for a full list of flags.

For example, here's a set of invocations for an NVIDIA A5000 GPU:

.. code-block:: bash

    python tests/benchmark.py -o outputs/uvu uvu --limited-memory --plot
    python tests/benchmark.py -o outputs/uvw uvw -b 25000 --plot
    python tests/benchmark.py -o outputs/roofline roofline --plot
    python tests/benchmark.py -o outputs/conv conv --data data/molecular_structures --limited-memory

For GPUs besides the NVIDIA A100, the roofline slope / peak will be incorrect.
The plots for the convolution fusion experiments also require a GPU 
with a minimum of 40GB of memory. 

List of GPUs Tested
--------------------------------
OpenEquivariance has been tested successfully the following GPUs. Submit a pull 
request if you'd like to add your own!

- NVIDIA V100 (V. Bharadwaj, LBNL Einsteinium, June 2025)
- NVIDIA A100-SXM-40GB and A100-SXM-80GB (A. Glover, NERSC Perlmutter, June 2025)
- NVIDIA A5000 (V. Bharadwaj, UCB SLICE, June 2025)
- NVIDIA H100 (L. Larsen, P1 DTU HPC, June 2025)
- AMD MI250x (V. Bharadwaj, OLCF Frontier, June 2025)
- AMD MI300x (V. Bharadwaj, AMD Cloud, February 2025)