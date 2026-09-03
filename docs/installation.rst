Installation
==============================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

You need the following to install OpenEquivariance:

- A Linux system equipped with an NVIDIA / AMD / Intel graphics card.
- Either PyTorch >= 2.4 (>= 2.8 for AOTI and export, >= 2.7 for Intel
  GPUs), or JAX>0.5.0 with CUDA, RocM, or XPU support. 
- GCC 9+ and the CUDA / HIP toolkit, or the oneAPI DPC++ compiler
  (``icpx``) for Intel GPUs. The command
  ``c++ --version`` should return >= 9.0; see below for details on 
  setting an alternate compiler.

.. note::

   On Intel GPUs the kernels are generated as SYCL and compiled at runtime
   through the oneAPI kernel compiler, so ``icpx`` must be on your ``PATH``
   (PyTorch locates the SYCL toolchain from it). Tensors passed to the
   kernels live on the ``xpu`` device rather than ``cuda``. PyTorch 2.7 is
   the minimum: 2.6 introduces the XPU device and the SYCL support in
   ``torch.utils.cpp_extension``, and 2.7 adds
   ``torch.library.register_autocast``. The precompiled extension and the
   JAX frontend both remain CUDA/HIP only.

.. tab:: PyTorch

    Installation is one easy command, followed by import verification: 

    .. code-block:: bash 

        pip install openequivariance
        python -c "import openequivariance"

    The second line triggers a build of the C++ extension we use to compile
    kernels, which can take a couple of minutes. Subsequent imports are
    much faster since this extension is cached.

    To support ``torch.compile``, ``torch.export``, and
    JITScript, OpenEquivariance needs to compile a C++ extension
    tightly integrated with PyTorch. If you see a warning that
    this extension could not be compiled, first check:

    .. code-block:: bash 

        c++ --version 
        
    To build the extension with an alternate compiler, set the 
    ``CC`` and ``CXX``
    environment variable and retry the import:

    .. code-block:: bash

        export CC=/path/to/your/gcc
        export CXX=/path/to/your/g++
        python -c "import openequivariance"


    These configuration steps are required only ONCE after 
    installation (or upgrade) with pip. 


.. tab:: JAX NVIDIA GPUs 

    First ensure the appropriate JAX Python
    package is installed in your environment. Then
    run the following two commands stricly in order:

    .. code-block:: bash 

        pip install openequivariance[jax]
        pip install openequivariance_extjax --no-build-isolation

.. tab:: JAX AMD GPUs 

    Ensure that JAX is installed correctly with RocM support
    before running, in order,

    .. code-block:: bash 

        pip install openequivariance[jax]
        JAX_HIP=1 pip install openequivariance_extjax --no-build-isolation


.. tab:: Nightly (PT) 

    .. code-block:: bash 

        pip install "git+https://github.com/PASSIONLab/OpenEquivariance#subdirectory=openequivariance"


.. tab:: Nightly (JAX)

    .. code-block:: bash 

        pip install "git+https://github.com/PASSIONLab/OpenEquivariance#subdirectory=openequivariance[jax]"
        pip install "git+https://github.com/PASSIONLab/OpenEquivariance#subdirectory=openequivariance_extjax --no-build-isolation"

        # Use the command below for JAX+AMD
        # JAX_HIP=1 pip install "git+https://github.com/PASSIONLab/OpenEquivariance#subdirectory=openequivariance_extjax --no-build-isolation"


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


.. tab:: NERSC Perlmutter (NVIDIA A100) 

    .. code-block:: bash
        :caption: env.sh (last updated June 2025)

        module load gcc 
        module load conda

        # Deactivate any base environments
        for i in $(seq ${CONDA_SHLVL}); do 
            conda deactivate
        done

        conda activate <your-conda-env>


.. tab:: OLCF Frontier (AMD MI250x)

    You need to install a HIP-enabled verison of PyTorch to use our package. 
    Follow the steps `here <https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html>`_.


    .. code-block:: bash
        :caption: env.sh (last updated June 2025) 

        module load PrgEnv-gnu/8.6.0
        module load miniforge3/23.11.0-0
        module load rocm/6.4.0
        module load craype-accel-amd-gfx90a

        for i in $(seq ${CONDA_SHLVL}); do
            conda deactivate
        done

        conda activate <your-conda-env>
        export CC=cc
        export CXX=CC


.. tab:: ALCF Aurora (Intel Data Center GPU Max)

    Aurora provides a PyTorch build with XPU support in its ``frameworks``
    module; no separate PyTorch install is needed.

    .. code-block:: bash
        :caption: env.sh (last updated August 2026)

        module load oneapi/release
        module load frameworks

        export CC=icx
        export CXX=icpx