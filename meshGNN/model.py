import torch
from torch.nn import Sequential as Seq, Linear, Parameter, ReLU
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree


class MeshGNN(torch.nn.Module):
    def __init__(self, hidden_dim=128, n_message_passings=4):
        super().__init__()
        self.n_message_passings = n_message_passings
        self.encoder = GCNConv(6, hidden_dim)
        self.hidden = GCNConv(hidden_dim, hidden_dim)
        self.decoder = GCNConv(hidden_dim, 3)
        self.relu = ReLU()
        
    def forward(self, x, edge_index):
        x = self.relu(self.encoder(x, edge_index))
        for _ in range(self.n_message_passings):
            x = self.relu(self.hidden(x, edge_index))
        return self.decoder(x, edge_index)
        
