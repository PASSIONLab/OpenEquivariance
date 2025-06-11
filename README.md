# OpenEquivariance
[![OEQ CUDA C++ Extension Build Verification](https://github.com/PASSIONLab/OpenEquivariance/actions/workflows/verify_extension_build.yml/badge.svg?event=push)](https://github.com/PASSIONLab/OpenEquivariance/actions/workflows/verify_extension_build.yml)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

[[Examples]](#show-me-some-examples) 
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
pip install openequivariance
```

We provide up to an order of magnitude acceleration over e3nn perform on par with the latest
version of [NVIDIA cuEquivariance](https://github.com/NVIDIA/cuEquivariance),
which has a closed-source kernel package. 
We also offer fused equivariant graph 
convolutions that can reduce 
computation and memory consumption significantly. 

For detailed instructions on tests, benchmarks, MACE / Nequip, and our API,
check out the [documentation](https://passionlab.github.io/OpenEquivariance).

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

## Citation and Acknowledgements
If you find this code useful, please cite our paper:

```bibtex
@inbook{openequivariance,
author={Vivek Bharadwaj and Austin Glover and Aydin Buluc and James Demmel},
title={An Efficient Sparse Kernel Generator for O(3)-Equivariant Deep Networks}, 
booktitle = {SIAM Conference on Applied and Computational Discrete Algorithms (ACDA25)},
chapter = {},
url={https://arxiv.org/abs/2501.13986},
publisher={Society for Industrial and Applied Mathematics},
year={2025}
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