# Examples on the front page of README

# ===============================
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
# ===============================

# ===============================
import fast_tp as ftp

problem = ftp.TPProblem(X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False)
tp_fast = ftp.LoopUnrollTP(problem, torch_op=True)

Z = tp_fast(X, Y, W) # Reuse X, Y, W from earlier
print(torch.norm(Z))
# ===============================


# ===============================
from torch_geometric import EdgeIndex

node_ct, nonzero_ct = 3, 4

# Sender, receiver indices for message passing GNN
edge_index = EdgeIndex(
                [[0, 1, 2, 1],  # Receiver 
                 [1, 0, 1, 2]], # Sender 
                device='cuda',
                dtype=torch.long)

X = torch.rand(node_ct, X_ir.dim, device='cuda', generator=gen)
Y = torch.rand(nonzero_ct, Y_ir.dim, device='cuda', generator=gen)
W = torch.rand(nonzero_ct, problem.weight_numel, device='cuda', generator=gen)

tp_conv = ftp.LoopUnrollConv(problem, torch_op=True, deterministic=False) # Reuse problem from earlier
Z = tp_conv.forward(X, Y, W, edge_index[0], edge_index[1]) # Z has shape [node_ct, z_ir.dim]
print(torch.norm(Z))
# ===============================

# ===============================
_, sender_perm = edge_index.sort_by("col")            # Sort by sender index 
edge_index, receiver_perm = edge_index.sort_by("row")     # Sort by receiver index


# Now we can use the faster deterministic algorithm
tp_conv = ftp.LoopUnrollConv(problem, torch_op=True, deterministic=True) 
Z = tp_conv.forward(X, Y, W[receiver_perm], edge_index[0], edge_index[1], sender_perm) # Z has shape [node_ct, z_ir.dim]
print(torch.norm(Z))
# ===============================