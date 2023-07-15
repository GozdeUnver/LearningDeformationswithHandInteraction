import torch
from torch.nn import Sequential as Seq, Linear, Parameter, ReLU
from torch_geometric.nn import MessagePassing, SAGEConv
from torch_geometric.utils import add_self_loops, degree


# Try BFS to find subgraph around the contact point
class MeshGNN(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.relu = ReLU()
        self.layers = Seq(
            SAGEConv(6, 3),
            # Try different layers. Use activation
        )
    
    def forward(self, x, edge_index):
        for _ in range(15):
            for layer in self.layers[:-1]:
                x = layer(x, edge_index)
            x = self.layers[-1](x, edge_index)
        
        return x

# class EdgeConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(aggr='max') #  "Max" aggregation.
#         self.mlp = Seq(Linear(2 * in_channels, out_channels),
#                        ReLU(),
#                        Linear(out_channels, out_channels))

#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]

#         return self.propagate(edge_index, x=x)

#     def message(self, x_i, x_j):
#         # x_i has shape [E, in_channels]
#         # x_j has shape [E, in_channels]

#         tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
#         return self.mlp(tmp)


# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(aggr='add')  # "Add" aggregation (Step 5).
#         self.lin = Linear(in_channels, out_channels, bias=False)
#         self.bias = Parameter(torch.empty(out_channels))

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         self.bias.data.zero_()

#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]

#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         # Step 2: Linearly transform node feature matrix.
#         x = self.lin(x)

#         # Step 3: Compute normalization.
#         row, col = edge_index
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         # Step 4-5: Start propagating messages.
#         out = self.propagate(edge_index, x=x, norm=norm)

#         # Step 6: Apply a final bias vector.
#         out += self.bias

#         return out

#     def message(self, x_j, norm):
#         # x_j has shape [E, out_channels]

#         # Step 4: Normalize node features.
#         return norm.view(-1, 1) * x_j
