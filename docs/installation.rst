Installation
==============================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

You need the following to install OpenEquivariance:

- A Linux system equipped with an NVIDIA / AMD graphics card.
- PyTorch >= 2.4 (>= 2.8 for AOTI and export).
- GCC 9+ and the CUDA / HIP toolkit. The command
  ``c++ --version`` should return >= 9.0; see below for details on 
  setting an alternate compiler.

Installation is one easy command, followed by import verification: 

.. code-block:: bash 

    pip install openequivariance
    python -c "import openequivariance"

The second line triggers a build of the C++ extension we use to compile
kernels, which can take a couple of minutes. Subsequent imports are
much faster since this extension is cached. 


Compiling the Integrated PyTorch Extension
------------------------------------------
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

    export CCC=/path/to/your/gcc
    export CXX=/path/to/your/g++
    python -c "import openequivariance"

These configuration steps are required only ONCE after 
installation (or upgrade) with pip. 

Configurations on Major Platforms 
---------------------------------
OpenEquivariance has been tested on both supercomputers and lab clusters.
Here are some tested environment configuration files. If use OpenEquivariance
on a widely-used platform, send us a pull request to add your configuration! 

NERSC Perlmutter (NVIDIA A100)
""""""""""""""""""""""""""""""

.. code-block:: bash
    :caption: env.sh (last updated June 2025)

    module load gcc 
    module load conda

    # Deactivate any base environments
    for i in $(seq ${CONDA_SHLVL}); do 
        conda deactivate
    done

    conda activate <your-conda-env>


OLCF Frontier (AMD MI250x)
""""""""""""""""""""""""""
You need to install a HIP-enabled verison of PyTorch to use our package. 
To do this, follow the steps `here <https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html>`_.


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