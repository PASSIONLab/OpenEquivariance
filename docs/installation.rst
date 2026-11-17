Installation
==============================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

You need the following to install OpenEquivariance:

- A Linux system equipped with an NVIDIA / AMD / Intel graphics card.
- Either PyTorch >= 2.10, or JAX>0.5.0 with CUDA, RocM, or XPU support.
- To build from source: GCC 9+ and the CUDA / HIP toolkit, or the
  oneAPI DPC++ compiler (``icpx``) for Intel GPUs. The command
  ``c++ --version`` should return >= 9.0; see below for details on
  setting an alternate compiler.

.. note::

   On Intel GPUs the kernels are generated as SYCL and compiled at runtime
   through the oneAPI kernel compiler, so ``icpx`` must be on your ``PATH``
   (PyTorch locates the SYCL toolchain from it). Tensors passed to the
   kernels live on the ``xpu`` device rather than ``cuda``. The precompiled
   extension and the JAX frontend both remain CUDA/HIP only.

.. tab:: CUDA or HIP Wheels

    Installation is one easy command, followed by import verification: 

    .. code-block:: bash 

        pip install openequivariance
        python -c "import openequivariance"

    OpenEquivariance bundles a precompiled library files for two recent
    CUDA versions and a HIP version. If you're building for an architecture
    we don't support or need a different CUDA version, see the source 
    distribution installation instructions. 

.. tab:: CUDA or HIP Source Distribution 

    Run 

    .. code-block:: bash

        pip install --no-binary openequivariance openequivariance

    Optionally, set the environment variables ``CC`` and ``CXX`` before the pip install
    to build with a different compiler:

    .. code-block:: bash

        export CC=/path/to/your/gcc
        export CXX=/path/to/your/g++
        pip install --no-binary openequivariance openequivariance


.. tab:: SYCL 

    On Intel GPUs, the extension is still compiled on first import,
    which can take a couple of minutes; subsequent imports use the
    cached build. Install via 

    .. code-block:: bash

        pip install openequivariance[sycl]
        export CC=/path/to/your/cc
        export CXX=/path/to/your/c++
        python -c "import openequivariance"


.. tab:: Developer Install 

    This mode enables fast recompilation for
    incremental builds. Build isolation is disabled
    in this mode:

    .. code-block:: bash

        git clone https://github.com/PASSIONLab/OpenEquivariance
        cd OpenEquivariance
        pip install $(python -c "import tomllib; print(*tomllib.load(open('openequivariance/pyproject.toml', 'rb'))['build-system']['requires'])")
        pip install --no-build-isolation -Ceditable.rebuild=true -e ./openequivariance

.. tab:: Nightly Source Distribution

    .. code-block:: bash 

        pip install --force-reinstall "git+https://github.com/PASSIONLab/OpenEquivariance#subdirectory=openequivariance"

    To install from a specific branch on our repo:

    .. code-block:: bash

        pip install --force-reinstall "git+https://github.com/PASSIONLab/OpenEquivariance@my-test-branch#subdirectory=openequivariance"



If you're using JAX, set the environment variable 
``OEQ_NOTORCH=1`` to avoid a PyTorch import: 

.. code-block:: bash

    export OEQ_NOTORCH=1
    python -c "import openequivariance.jax"


Configurations on Major Platforms 
---------------------------------
OpenEquivariance has been tested on both supercomputers and lab clusters.
Here are some tested environment configuration files. If you use OpenEquivariance
on a major cluster, send us a pull request to add your configuration! 


.. tab:: ALCF Aurora (Intel Data Center GPU Max)

    Aurora provides a PyTorch build with XPU support in its ``frameworks``
    module; no separate PyTorch install is needed.

    .. code-block:: bash
        :caption: env.sh (last updated August 2026)

        module restore
        module load frameworks

        export CC=icx
        export CXX=icpx