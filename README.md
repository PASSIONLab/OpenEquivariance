# ESPMM

[[Examples]](#show-me-some-examples) [[Installation]](#installation)
[[Supported Tensor Products]](#tensor-products-we-accelerate)
[[Citation and Acknowledgements]](#acknowledgements)

This repository a kernel generator for the Clebsch-Gordon tensor product, 
a key kernel in equivariant deep neural networks. It implements
a subset of the functionality of [e3nn](https://e3nn.org/)
for its common use cases in equivariant graph neural networks
(e.g. [Nequip](https://github.com/mir-group/nequip) or
[MACE](https://github.com/ACEsuit/mace)). 

We provide up to an order of magnitude acceleration over e3nn
and up to ~2x speedup over 
[NVIDIA cuEquivariance](https://github.com/NVIDIA/cuEquivariance),
which has a closed-source kernel package. We also offer fused
equivariant graph convolutions that can reduce memory consumption 
significantly. 

We currently support NVIDIA GPUs and offer a torch frontend.
HIP support for AMD is planned! 

## Show me some examples
Here's a CG tensor product implemented by e3nn: 

```python
import torch
import e3nn.o3 as o3

batch_size = 1000
X_ir, Y_ir, Z_ir = o3.Irreps("128x5e"), o3.Irreps("128x3e"), o3.Irreps("128x5e") 
X, Y = torch.rand(batch_size, x_ir.dim, device='cuda'), torch.rand(batch_size, y_ir.dim, device='cuda')
W = torch.rand(tp.weight_numel, device='cuda')

tp_e3nn = o3.TensorProduct(X_ir, Y_ir, Z_ir,
        shared_weights=False, internal_weights=False)

Z = tp_e3nn(X, Y, W)
print(torch.norm(Z))
```

And here's the same tensor product using fast_tp. We require that your
tensors are stord on a CUDA device for this to work: 

```python
import fast_tp as ftp

problem = ftp.TPProblem(X_ir, Y_ir, Z_ir, shared_weights=False, internal_weights=False)
tp_fast = ftp.LoopUnrollTP(problem, torch_op=True)

Z = tp_fast(X, Y, W) # Reuse X, Y, W from earlier
print(torch.norm(Z))
```

Our interface for `ftp.TPProblem` is almost a strict superset of 
`o3.TensorProduct` (two key differences: we 
impose `internal_weights=False` and add support for multiple datatypes). 
You can pass e3nn `Irreps` instances directly or 
use `ftp.Irreps`, which is identical. We recommend 
reading the [e3nn documentation and API reference](https://docs.e3nn.org/en/latest/) first, then using our kernels 
as drop-in replacements. We support most "uvu" and "uvw" tensor products; 
see [this section](#tensor-products-we-support) for an up-to-date list of supported configurations. 

**Important**: For many configurations, our kernels return results identical to
e3nn up to floating point roundoff (in particular, all "uvu" problems with
multiplicity 1 for all irreps in the second input). For other configurations 
(e.g. any "uvw" connection modes), we return identical 
results up to a well-defined reordering of the weights relative to e3nn. 

If you're performing tensor products as part of a message passing graph
neural network, we offer fused kernels that save both memory and compute time: 

```python
from torch.geometric import EdgeIndex

node_ct, edge_ct = 3, 4

# Sender, receiver indices for message passing GNN
edge_index = EdgeIndex(
                [[0, 1, 2, 1],
                 [1, 0, 1, 2]],
                device='cuda')

X, Y = torch.rand(node_ct, x_ir.dim, device='cuda'), torch.rand(edge_ct, y_ir.dim, device='cuda')
W = torch.rand(tp.weight_numel, device='cuda')

tp_conv = ftp.LoopUnrollConv(problem, torch_op=True, deterministic=False) # Reuse problem from earlier
Z = tp_conv.forward(X, Y, W, edge_index[0], edge_index[1]) # Z has shape [node_ct, z_ir.dim] 
```

If you can guarantee `EdgeIndex` is sorted by row and supply the transpose
permutation, 


## Installation 
We provide several options to build our package and replicate
the benchmarks in our preprint. Right now, we only support
source builds, but we provide scripts to streamline installation.

We highly recommend that you use
`conda` or `mamba` to set up a Python environment for installation.

### Build via install script and pip (fastest) 
The steps below assume that you're using a bash shell and have a C / C++ 
compiler that CMake can find. If not, you can install [gxx](https://anaconda.org/conda-forge/gxx/) from `conda-forge`. 

1. **Setup**: Create an environment (or activate an existing one) with 
  our core dependencies: 
    ```bash
    conda create -c conda-forge --name my_env python=3.11 pybind11 cmake nvidia::cuda-toolkit
    conda activate my_env 
    ``` 

2. **Install**: Build our package and install via `pip`: 
    ```bash
    git clone https://github.com/vbharadwaj-bk/equivariant_spmm/tree/release 
    cd equivariant_spmm
    sh dev_build.sh 
    pip install . # Use pip install -e . for an editable install 
    ``` 

3. **Test**: You're ready to go!

You don't have to install NVIDIA's CUDA toolkit or CMake if they exist on your
platform, but you're responsible for setting LD_LIBRARY_PATH so that libraries
are findable at runtime. Installing the CUDA toolkit via `conda` takes care of this for
you. 

### Build via conda or mambabuild
You can can also build our package via `conda-build` or
`conda mambabuild`. This can be much slower, but may help if you
encounter problems with the workflow above.

1. **Setup**: Create a new conda environment, or activate an existing one.
    You must install either `boa` or `conda-build`; we 
    use `boa` for its speed. 
    ```bash
    conda create --name my_env python=3.11 conda-forge::boa mamba
    conda activate my_env 
    ``` 

2. **Install**: Clone, build, and install in three steps:
    ```bash
    git clone https://github.com/vbharadwaj-bk/equivariant_spmm.git
    conda mambabuild ./equivariant_spmm 
    mamba install --use-local fast_tp 
    ```

    Use `build` and `conda` in place of `mambabuild` and `mamba`, 
    respectively, if you installed `conda-build` in Step 1.

### Build to replicate our benchmarks 
Follow either build process above. You'll also need the following packages: 
- `e3nn`, 
- `cuEquivariance`
- `cuEquivariance-torch` 
- `cuEquivariance-ops-torch-cu11` OR `cuEquivariance-ops-torch-cu12` 
- `matplotlib` (to reproduce our figures) 

We conducted our benchmarks on an NVIDIA A100-SXM-80GB GPU at
Lawrence Berkeley National Laboratory. Your results may differ 
a different GPU.

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

Our code include correctness checks, but the configuration space is large. If you notice
a bug, let us know in a Github issue. We'll try our best to correct it or document the problem here.

We do not yet support:

- Mixing different instruction types in the same tensor product. 
- Instruction types besides "uvu" and "uvw".

If you have a use case for any of the unsupported features above, let us know.

## Acknowledgements