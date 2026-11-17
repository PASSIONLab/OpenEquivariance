Installation
==============================

.. admonition:: Deprecation of ``openequivariance_extjax``
   :class: tip

   Installing ``openequivariance_extjax`` is no longer
   required or supported, since the regular
   ``openequivariance`` package contains the
   relevant binaries. 
   

.. toctree::
   :maxdepth: 1
   :caption: Contents:

You need the following to install OpenEquivariance:

- A Linux system equipped with an NVIDIA / AMD / Intel graphics card.
- Either PyTorch >= 2.10, or JAX >= 0.8.1 with CUDA, ROCm, or XPU support.

.. tab:: CUDA or HIP Wheels

    Installation is one easy command for all PyTorch, JAX, CUDA, and
    HIP platforms: 

    .. code-block:: bash 

        pip install openequivariance

    OpenEquivariance bundles a precompiled library files for two recent
    CUDA versions and a HIP version. If you're building for an architecture
    we don't support or need a different CUDA version, see the source 
    distribution installation instructions. 


    If you're using JAX, set the environment variable 
    ``OEQ_NOTORCH=1`` to avoid a PyTorch import: 

    .. code-block:: bash

        export OEQ_NOTORCH=1
        python -c "import openequivariance.jax"

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

    You'll need the oneAPI DPC++ compiler (``icpx``) for Intel GPUs. 
    Kernels are generated as SYCL and compiled at runtime
    through the oneAPI kernel compiler, so ``icpx`` must be on your ``PATH``
    (PyTorch locates the SYCL toolchain from it). Tensors passed to the
    kernels are resident on the ``xpu`` device rather than ``cuda``. 

    Our SYCL extension is still compiled on first import,
    which can take a couple of minutes; subsequent imports use the
    cached build. Install via 

    .. code-block:: bash

        pip install "openequivariance[sycl]"
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

.. tab:: Nightly

    .. code-block:: bash 

        pip install --force-reinstall "git+https://github.com/PASSIONLab/OpenEquivariance#subdirectory=openequivariance"

    To install from a specific branch on our repo:

    .. code-block:: bash

        pip install --force-reinstall "git+https://github.com/PASSIONLab/OpenEquivariance@my-test-branch#subdirectory=openequivariance"


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