Tests and Benchmarks
==============================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

OpenEquivariance is equipped with a comprehensive suite of tests
and benchmarking utilities. You'll need some additional dependencies to run 
these; we provide instructions below. 

We recommend you clone our repository and use an editable install to run tests
and benchmarks. You can still test our code with a non-editable install; just 
download the test folder and install only the dependencies with:

.. code-block:: bash

    pip install "https://github.com/PASSIONLab/OpenEquivariance[dev]" --only-deps
    pip install "https://github.com/PASSIONLab/OpenEquivariance[bench]" --only-deps

Correctness Tests
------------------------------
To set up the editable install and run the entire testsuite, use: 

.. code-block:: bash

    git clone https://github.com/PASSIONLab/OpenEquivariance 
    pip install -e .[dev] 
    cd OpenEquivariance
    pytest 

Browse the ``tests`` directory to run specific components. 


Replicating our Benchmarks
------------------------------
We conducted our benchmarks on an NVIDIA A100-SXM-80GB GPU at Lawrence Berkeley National Laboratory. 
Your results may differ a different GPU. The following invocations run the experiments
and generate plots from our paper.

.. code-block:: bash

    git clone https://github.com/PASSIONLab/OpenEquivariance 
    pip install -e .[bench] 
    cd OpenEquivariance
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

List of GPUs Tested
--------------------------------
OpenEquivariance has been tested successfully the following GPUs. Submit a pull 
request if you'd like to add your own!

- NVIDIA A100-SXM-40GB and A100-SXM-80GB (A. Glover, NERSC Perlmutter)
- NVIDIA A5000 (V. Bharadwaj, UCB SLICE)
- AMD MI250x (V. Bharadwaj, OLCF Frontier)
- AMD MI300x (V. Bharadwaj, AMD Cloud)