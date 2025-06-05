.. OpenEquivariance documentation master file, created by
   sphinx-quickstart on Tue Jun  3 00:20:54 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpenEquivariance
==============================

`OpenEquivariance <https://github.com/PASSIONLab/OpenEquivariance>`_ is a CUDA and 
HIP kernel generator for the Clebsch-Gordon
tensor product, a key kernel in equivariant graph neural networks. We offer
an identical interface to e3nn and produce the same results 
(up to numerical roundoff). Our package exhibits up to an order of magnitude
speedup over e3nn and competitive performance with NVIDIA's cuEquivariance. 

Here, you can find our API reference, installation instructions, 
and troubleshooting guide. We support for both NVIDIA and AMD GPUs through
our PyTorch interface, including support for JITScript compilation accessible
from C++.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   api
   supported_ops
   tests_and_benchmarks
   models 
