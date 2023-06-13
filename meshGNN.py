import pymesh
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

# loading a single mesh_pair
loaded_mesh_pairs = []
undeformed_mesh = pymesh.meshio.load_mesh('data/undeformed_mesh_1.obj')
deformed_mesh = pymesh.meshio.load_mesh('data/deformed_mesh_1.obj')
correspondences = np.load('data/correspondences_1')
contact_point = np.load('data/contact_point_1')
max_displacement = np.load('data/max_disp_1')
loaded_mesh_pairs.append(zip(undeformed_mesh, deformed_mesh, correspondences, contact_point, max_displacement))
#need to: load all pairs of meshes with correspondences + contact point

#retrieve vertices/faces and edges -- create dataset
#remaining question -- embedding feature 'max displacement' in graph -- global feature?

class MeshPairDataset(Dataset):
    def __init__(self, mesh_pairs, transform=None, pre_transform=None):
        super(MeshPairDataset, self).__init__('.', transform, pre_transform)
        self.mesh_pairs = mesh_pairs

    def len(self):
        return len(self.mesh_pairs)

    def get(self, idx):
        undeformed_mesh, deformed_mesh, correspondences, contact_point, max_displacement = self.mesh_pairs[idx]
        x = torch.tensor(undeformed_mesh.vertices, dtype=torch.float)
        edges = []
        for face in undeformed_mesh.faces: #assumes triangle mesh
            edges.append([face[0], face[1]])
            edges.append([face[1], face[0]])
            edges.append([face[1], face[2]])
            edges.append([face[2], face[1]])
            edges.append([face[2], face[0]])
            edges.append([face[0], face[2]])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        #can append displacement as feature for only that node or for all nodes...
        #depending on message passing procedure may not affect all nodes if only applied to single node
        #but inputting multiple contact points becomes unviable if applied to all
        #here adding max_disp feature to all nodes while one-hot encoding the contact point
        #could maybe try to encode proximity or something instead of one-hot
        max_displacement_tensor = torch.full((undeformed_mesh.num_nodes, 1), max_displacement)
        contact_point_tensor = torch.zeros(undeformed_mesh.num_nodes, 1)
        contact_point_tensor[contact_point] = 1
        x = torch.cat([x, max_displacement_tensor, contact_point_tensor], dim=-1)

        y = torch.tensor(deformed_mesh.vertices, dtype=torch.float)
        correspondences_tensor = torch.tensor(correspondences, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y, correspondences=correspondences_tensor)
        data.num_nodes = len(undeformed_mesh.vertices)
        return data


dataset = MeshPairDataset(loaded_mesh_pairs)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

class MeshGCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(MeshGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_node_features)  #can alternatively define own message passing procedure

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First Message Passing Layer (ReLU activation)
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)

        return x


model = MeshGCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    for data in dataloader:
        model.train()
        optimizer.zero_grad()
        target = data.y
        correspondences = data.correspondences
        out = model(data)
        #use correspondences to calculate loss
        #means network will learn how to move vertices according to correspondence algorithm...
        pred_corr = out.x[correspondences[:, 0]]
        actual_corr = target[correspondences[:, 1]]
        loss = torch.nn.functional.mse_loss(pred_corr, actual_corr)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')


unseen_contact_point_w_disp = ...
new_data = ...

out = model(new_data)