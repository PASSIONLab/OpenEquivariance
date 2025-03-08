# OpenEquivariance

[[Examples]](#show-me-some-examples) [[Installation]](#installation)
[[Supported Tensor Products]](#tensor-products-we-accelerate)
[[Citation and Acknowledgements]](#citation-and-acknowledgements)

OpenEquivariance is a kernel generator for the Clebsch-Gordon tensor product, 
a key kernel in rotation-equivariant deep neural networks. 
It implements some of the tensor products 
that [e3nn](https://e3nn.org/) supports that are
commonly found in graph neural networks 
(e.g. [Nequip](https://github.com/mir-group/nequip) or
[MACE](https://github.com/ACEsuit/mace)). To get started, install our package via

```bash
pip install git+https://github.com/PASSIONLab/OpenEquivariance
```

We provide up to an order of magnitude acceleration over e3nn
and up to ~2x speedup over 
[NVIDIA cuEquivariance](https://github.com/NVIDIA/cuEquivariance),
which has a closed-source kernel package. We also offer fused
equivariant graph convolutions that can reduce 
computation and memory consumption significantly. 

We currently support NVIDIA GPUs and offer a PyTorch frontend.

**Warning**: This is an early release, bug reports are welcome.

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
neural network, we offer fused kernels that save both memory and compute time (only supported
for "uvu" at the moment, "uvw" support coming soon): 

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

## Installation 
We currently support Linux systems only.
We highly recommend that you use
`conda` or `mamba` to set up a Python environment for installation.

### Install via pip
After activating an environment of your choice, run
```bash
pip install git+https://github.com/PASSIONLab/OpenEquivariance
```
After installation, the very first library
import will trigger a build of a C++ extension we use.
All subsequent imports will not retrigger compilation.

If you encounter problems with installation, let us
know by filing a bug and try a development build (see
below). After installation, you should be able 
to run the example above.

### Build to replicate our benchmarks 
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

The file `test/benchmark.py` can reproduce the figures in 
our paper an A100-SXM4-80GB GPU. 
Run it with the following invocations: 
```bash
python test/benchmark.py -o outputs/uvu uvu --plot
python test/benchmark.py -o outputs/uvu uvw --plot
python test/benchmark.py -o outputs/roofline roofline --plot
python test/benchmark.py -o outputs/conv conv --plot --data data/molecular_structures
```

If your GPU has limited memory, you might want to try
the `--limited-memory` flag to disable some expensive
tests and / or reduce the batch size with `-b`. Run
`python test/benchmark.py --help` for a full list of flags.

Here's a set
of invocations for an A5000 GPU:

```bash
python test/benchmark.py -o outputs/uvu uvu --limited-memory --plot
python test/benchmark.py -o outputs/uvw uvw -b 25000 --plot
python test/benchmark.py -o outputs/roofline roofline --plot
python test/benchmark.py -o outputs/conv conv --data data/molecular_structures --limited-memory
```
Note that for GPUs besides the one we used in our 
testing, the roofline slope / peak will be incorrect, and your results
may differ from the ones we've reported. The plots for the convolution fusion
experiments also require a GPU with a minimum of 40GB of memory. 

### Running MACE
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
python test/mace_driver.py carbon.xyz -o outputs/mace_tests -i oeq
```

4. If you have a GPU with 80GB of memory OR supply a smaller molecular graph
   as the input file, you can run the full benchmark that includes `e3nn` and `cue`: 
```bash
python test/mace_driver.py carbon.xyz -o outputs/mace_tests -i e3nn cue oeq
```

## Tensor products we accelerate 
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

## Citation and Acknowledgements
If you find this code useful, please cite our paper:

```bibtex
@misc{openequivariance,
      title={An Efficient Sparse Kernel Generator for O(3)-Equivariant Deep Networks}, 
      author={Vivek Bharadwaj and Austin Glover and Aydin Buluc and James Demmel},
      year={2025},
      eprint={2501.13986},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.13986}, 
}
```

Our codebase includes a lightweight clone of 
[e3nn](https://e3nn.org/)'s frontend interface (in particular, the 
`TensorProduct` and `Irreps` classes). We removed references to Pytorch
and separated the implementation from the problem description (for future
frontend support outside of torch). Thank you to the current
developers and maintainers! 

## Copyright

Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved. 

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
