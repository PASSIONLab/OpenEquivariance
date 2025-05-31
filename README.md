# OpenEquivariance
[![OEQ CUDA C++ Extension Build Verification](https://github.com/PASSIONLab/OpenEquivariance/actions/workflows/verify_extension_build.yml/badge.svg?event=push)](https://github.com/PASSIONLab/OpenEquivariance/actions/workflows/verify_extension_build.yml)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

[[Examples]](#show-me-some-examples) [[Installation]](#installation)
[[Supported Tensor Products]](#tensor-products-we-accelerate)
[[Citation and Acknowledgements]](#citation-and-acknowledgements)

OpenEquivariance is a CUDA and HIP kernel generator for the Clebsch-Gordon tensor product, 
a key kernel in rotation-equivariant deep neural networks. 
It implements some of the tensor products 
that [e3nn](https://e3nn.org/) supports 
commonly found in graph neural networks 
(e.g. [Nequip](https://github.com/mir-group/nequip) or
[MACE](https://github.com/ACEsuit/mace)). To get 
started, ensure that you have GCC 9+ on your system 
and install our package via

```bash
pip install git+https://github.com/PASSIONLab/OpenEquivariance
```

We provide up to an order of magnitude acceleration over e3nn perform on par with the latest
version of [NVIDIA cuEquivariance](https://github.com/NVIDIA/cuEquivariance),
which has a closed-source kernel package. 
We also offer fused equivariant graph 
convolutions that can reduce 
computation and memory consumption significantly. 

We currently support NVIDIA GPUs and just added beta support on AMD GPUs for
all tensor products! See [the coverage table](#tensor-products-we-accelerate) for more 
details.

📣 📣 OpenEquivariance was accepted to the 2025 SIAM Conference on Applied and 
Computational Discrete Algorithms (Proceedings Track)! Catch the talk in 
Montréal and check out the [camera-ready copy on Arxiv](https://arxiv.org/abs/2501.13986) (available May 12, 2025).

## Show me some examples
Here's a CG tensor product implemented by e3nn: 

```python
import torch
import e3nn.o3 as o3

gen = torch.Generator(device='cuda')

batch_size = 1000
X_ir, Y_ir, Z_ir = o3.Irreps("1x2e"), o3.Irreps("1x3e"), o3.Irreps("1x2e") 
X = torch.rand(batch_size, X_ir.dim, device='cuda', generator=gen)
Y = torch.rand(batch_size, Y_ir.dim, device='cuda', generator=gen)

instructions=[(0, 0, 0, "uvu", True)]

tp_e3nn = o3.TensorProduct(X_ir, Y_ir, Z_ir, instructions,
        shared_weights=False, internal_weights=False).to('cuda')
W = torch.rand(batch_size, tp_e3nn.weight_numel, device='cuda', generator=gen)

Z = tp_e3nn(X, Y, W)
print(torch.norm(Z))
```

And here's the same tensor product using openequivariance. We require that your
tensors are stored on a CUDA device for this to work: 

```python
import openequivariance as oeq

problem = oeq.TPProblem(X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False)
tp_fast = oeq.TensorProduct(problem, torch_op=True)

Z = tp_fast(X, Y, W) # Reuse X, Y, W from earlier
print(torch.norm(Z))
```

Our interface for `oeq.TPProblem` is almost a strict superset of 
`o3.TensorProduct` (two key differences: we 
impose `internal_weights=False` and add support for multiple datatypes). 
You can pass e3nn `Irreps` instances directly or 
use `oeq.Irreps`, which is identical. 

We recommend reading the [e3nn documentation and API reference](https://docs.e3nn.org/en/latest/) first, then using our kernels 
as drop-in replacements. We support most "uvu" and "uvw" tensor products; 
see [this section](#tensor-products-we-accelerate) for an up-to-date list of supported configurations. 

**Important**: For many configurations, our kernels return results identical to
e3nn up to floating point roundoff (this includes all "uvu" problems with
multiplicity 1 for all irreps in the second input). For other configurations 
(e.g. any "uvw" connection modes), we return identical 
results up to a well-defined reordering of the weights relative to e3nn. 

If you're executing tensor products as part of a message passing graph
neural network, we offer fused kernels that save both memory and compute time: 

```python
from torch_geometric import EdgeIndex

node_ct, nonzero_ct = 3, 4

# Receiver, sender indices for message passing GNN
edge_index = EdgeIndex(
                [[0, 1, 1, 2],  # Receiver 
                 [1, 0, 2, 1]], # Sender 
                device='cuda',
                dtype=torch.long)

X = torch.rand(node_ct, X_ir.dim, device='cuda', generator=gen)
Y = torch.rand(nonzero_ct, Y_ir.dim, device='cuda', generator=gen)
W = torch.rand(nonzero_ct, problem.weight_numel, device='cuda', generator=gen)

tp_conv = oeq.TensorProductConv(problem, torch_op=True, deterministic=False) # Reuse problem from earlier
Z = tp_conv.forward(X, Y, W, edge_index[0], edge_index[1]) # Z has shape [node_ct, z_ir.dim]
print(torch.norm(Z))
```

If you can guarantee `EdgeIndex` is sorted by receiver index and supply the transpose
permutation, we can provide even greater speedup (and deterministic results) 
by avoiding atomics: 

```python
_, sender_perm = edge_index.sort_by("col")            # Sort by sender index 
edge_index, receiver_perm = edge_index.sort_by("row") # Sort by receiver index

# Now we can use the faster deterministic algorithm
tp_conv = oeq.TensorProductConv(problem, torch_op=True, deterministic=True) 
Z = tp_conv.forward(X, Y[receiver_perm], W[receiver_perm], edge_index[0], edge_index[1], sender_perm) 
print(torch.norm(Z))
```
**Note**: you don't need Pytorch geometric to use our kernels. When
`deterministic=False`, the `sender` and `receiver` indices can have
arbitrary order.

**New:** If you're working in FP32 precision and want
higher accuracy during graph convolution, we offer a Kahan 
summation variant of our deterministic algorithm:

```python
tp_conv_kahan = oeq.TensorProductConv(problem, torch_op=True, deterministic=True, kahan=True) 
Z = tp_conv_kahan.forward(X, Y[receiver_perm], W[receiver_perm], edge_index[0], edge_index[1], sender_perm) 
print(torch.norm(Z))
```

## Installation 
We currently support Linux systems only. 
Before installation and the first library import, 
ensure that the command 
`c++ --version` returns GCC 9+; if not, set the
`CC` and `CXX` environment variables to point to
valid compilers. On NERSC Perlmutter,
`module load gcc` will set up your environment
correctly. 

To install, run
```bash
pip install git+https://github.com/PASSIONLab/OpenEquivariance
```
After installation, the very first library
import will trigger a build of a C++ extension we use,
which takes longer than usual.
All subsequent imports will not retrigger compilation.

## Replicating our benchmarks 
To run our benchmark suite, you'll also need the following packages: 
- `e3nn`, 
- `cuEquivariance`
- `cuEquivariance-torch` 
- `cuEquivariance-ops-torch-cu11` OR `cuEquivariance-ops-torch-cu12` 
- `matplotlib` (to reproduce our figures) 

You can get all the necessary dependencies via our optional dependencies `[bench]`

```bash
pip install "git+https://github.com/PASSIONLab/OpenEquivariance[bench]"
```

We conducted our benchmarks on an NVIDIA A100-SXM-80GB GPU at
Lawrence Berkeley National Laboratory. Your results may differ 
a different GPU.

The file `tests/benchmark.py` can reproduce the figures in 
our paper an A100-SXM4-80GB GPU. 
Run it with the following invocations: 
```bash
python tests/benchmark.py -o outputs/uvu uvu --plot
python tests/benchmark.py -o outputs/uvw uvw --plot
python tests/benchmark.py -o outputs/roofline roofline --plot
python tests/benchmark.py -o outputs/conv conv --plot --data data/molecular_structures
python tests/benchmark.py -o outputs/kahan_conv kahan_conv --data data/molecular_structures/
```

If your GPU has limited memory, you might want to try
the `--limited-memory` flag to disable some expensive
tests and / or reduce the batch size with `-b`. Run
`python tests/benchmark.py --help` for a full list of flags.

Here's a set
of invocations for an A5000 GPU:

```bash
python tests/benchmark.py -o outputs/uvu uvu --limited-memory --plot
python tests/benchmark.py -o outputs/uvw uvw -b 25000 --plot
python tests/benchmark.py -o outputs/roofline roofline --plot
python tests/benchmark.py -o outputs/conv conv --data data/molecular_structures --limited-memory
```
Note that for GPUs besides the one we used in our 
testing, the roofline slope / peak will be incorrect, and your results
may differ from the ones we've reported. The plots for the convolution fusion
experiments also require a GPU with a minimum of 40GB of memory. 

## Testing Correctness
See the `dev` dependencies in `pyproject.toml`; you'll need `e3nn`,
`pytest`, `torch_geometric`, and `pytest-check` installed. You can test batch 
tensor products and fused convolution tensor products as follows:
```bash
pytest tests/batch_test.py 
pytest tests/conv_test.py 
```
Browse the file to select specific tests.

## Compilation with JITScript, Export, and AOTInductor 
OpenEquivariance supports model compilation with
`torch.compile`, JITScript, `torch.export`, and AOTInductor. 
Demo the C++ model exports with
```bash
pytest tests/export_test.py 
```
NOTE: the AOTInductor test (and possibly export) fail 
unless you are using a Nightly
build of PyTorch past 4/10/2025 due to incomplete support for 
TorchBind in earlier versions.

## Running MACE
We have modified MACE to use our accelerated kernels instead
of the standard e3nn backend. Here are the steps to replicate
our MACE benchmark:

1. Install `oeq` and our modified version of MACE:
```bash
pip uninstall mace-torch
pip install git+https://github.com/PASSIONLab/OpenEquivariance
pip install git+https://github.com/vbharadwaj-bk/mace_oeq
```

2. Download the `carbon.xyz` data file, available at <https://portal.nersc.gov/project/m1982/equivariant_nn_graphs/>. 
   This graph has 158K edges. With the original e3nn backend, you would need a GPU with 80GB
   of memory to run the experiments. `oeq` provides a memory-efficient equivariant convolution, so we expect
   the test to succeed.

3. Benchmark OpenEquivariance: 
```bash
python tests/mace_driver.py carbon.xyz -o outputs/mace_tests -i oeq
```

4. If you have a GPU with 80GB of memory OR supply a smaller molecular graph
   as the input file, you can run the full benchmark that includes `e3nn` and `cue`: 
```bash
python tests/mace_driver.py carbon.xyz -o outputs/mace_tests -i e3nn cue oeq
```

## Tensor products we accelerate

| Operation                | CUDA     | HIP |
|--------------------------|----------|-----|
| UVU                      | ✅        | ✅    |
| UVW                      | ✅        | ✅    |
| UVU + Convolution        | ✅        | ✅    |
| UVW + Convolution        | ✅        | ✅    |
| Symmetric Tensor Product | ✅ (beta) | ✅ (beta)  |

e3nn supports a variety of connection modes for CG tensor products. We support 
two that are commonly used in equivariant graph neural networks:
"uvu" and "uvw". Our JIT compiled kernels should handle:

1. Pure "uvu" tensor products, which are most efficient when the input with higher
multiplicities is the first argument. Our results are identical to e3nn when irreps in
the second input have multiplicity 1, and otherwise identical up to a reordering
of the input weights.

2. Pure "uvw" tensor products, which are currently more efficient when the input with
higher multiplicities is the first argument. Our results are identical to e3nn up to a reordering
of the input weights. 

Our code includes correctness checks, but the configuration space is large. If you notice
a bug, let us know in a Github issue. We'll try our best to correct it or document the problem here.

We do not (yet) support:

- Mixing different instruction types in the same tensor product. 
- Instruction types besides "uvu" and "uvw".
- Non-trainable instructions: all of your instructions must have weights associated. 

If you have a use case for any of the unsupported features above, let us know.

## Multidevice / Stream Support
To use OpenEquivariance on multiple GPUs of a single
compute node, we currently require that all GPUs 
share the same compute capability. This is because
our kernels are compiled based on the shared memory
capacity of the numerically first visible GPU card. 
On heterogeneous systems, you can still 
use OpenEquivariance on all GPUs that match the
compute capability of the first visible device.

We are working on support for CUDA streams!

## Citation and Acknowledgements
If you find this code useful, please cite our paper:

```bibtex
@inbook{openequivariance,
author={Vivek Bharadwaj and Austin Glover and Aydin Buluc and James Demmel},
title={An Efficient Sparse Kernel Generator for O(3)-Equivariant Deep Networks}, 
booktitle = {SIAM Conference on Applied and Computational Discrete Algorithms (ACDA25)},
chapter = {},
url={https://arxiv.org/abs/2501.13986}
}
```

Our codebase includes a lightweight clone of 
[e3nn](https://e3nn.org/)'s frontend interface (in particular, the 
`TensorProduct` and `Irreps` classes). We removed references to Pytorch
and separated the implementation from the problem description (for future
frontend support outside of torch). We also extracted the Wigner 3j tensor generating code from QuTiP. Thank you to the current
developers and maintainers! 

## Copyright

Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved. 

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
