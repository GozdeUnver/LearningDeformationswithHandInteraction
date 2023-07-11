import os

import torch

import numpy as np
import open3d as o3d

from torch_geometric.loader import DataLoader

from CustomDataset import CustomDataset
from model import MeshGNN

Vec3d = o3d.utility.Vector3dVector
Vec3i = o3d.utility.Vector3iVector


CONFIG_NAME = 'sageconv_multilayer'

outputs_folder_path = os.path.join('outputs', CONFIG_NAME)
models_folder_path = os.path.join(outputs_folder_path, 'models')
model_file_path = os.path.join(models_folder_path, 'model.t7')
meshes_folder_path = os.path.join(outputs_folder_path, 'predicted_meshes')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = torch.nn.L1Loss()


def create_train_data_loader():
    dataset = CustomDataset(preload=True, flavor='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader


def create_test_data_loader():
    dataset = CustomDataset(preload=True, flavor='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader


def prepare_graph(graph):
    nodes, edges = graph
    return (
        nodes.to(device).squeeze(0),
        edges.to(device).squeeze(0)
    )

def train(epochs = 1000, print_every = 10):
    model = MeshGNN().to(device)

    # parameters = list(gcn_conv.parameters()) + list(edge_conv.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_loss = torch.inf
    best_model = model.state_dict()

    dataloader = create_train_data_loader()

    for epoch in range(epochs):
        for input_graph, target_points in dataloader:
            input_nodes, input_edges = prepare_graph(input_graph)
            pred_points = model(input_nodes, input_edges)
            target_points = target_points.to(device).squeeze(0)

            optimizer.zero_grad()
            loss = loss_fn(pred_points, target_points)
            loss.backward()
            optimizer.step()
            
            if epoch % print_every == 0:
                print(f'Epoch {epoch}: Loss={loss.item()}')
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = model.state_dict()
                print('New Best! Loss=' + str(loss.item()))

    if not os.path.exists(models_folder_path):
        os.makedirs(models_folder_path)
    torch.save(best_model, model_file_path)


def test(save_targets=False):
    model = MeshGNN().to(device)
    state_dict = torch.load(model_file_path, map_location=device)
    model.load_state_dict(state_dict)
    dataloader = create_test_data_loader()
    if not os.path.exists(meshes_folder_path):
        os.makedirs(meshes_folder_path)

    for i, (input_triangles, input_graph, target_points) in enumerate(dataloader):
        input_nodes, input_edges = prepare_graph(input_graph)
        pred_points = model(input_nodes, input_edges)

        target_points = target_points.to(device).squeeze(0)
        loss = loss_fn(pred_points, target_points).item()
        print(f'Sample {i}: Loss={loss}')

        pred_mesh = o3d.geometry.TriangleMesh(
            Vec3d(pred_points.cpu().detach().numpy()),
            Vec3i(input_triangles.squeeze().detach().numpy())
        )
        mesh_file_path = os.path.join(meshes_folder_path, f'pred_{i}.obj')
        o3d.io.write_triangle_mesh(mesh_file_path, pred_mesh)
        if save_targets:
            pred_mesh.vertices = Vec3d(target_points.cpu().detach().numpy())
            mesh_file_path = os.path.join(meshes_folder_path, f'target_{i}.obj')
            o3d.io.write_triangle_mesh(mesh_file_path, pred_mesh)

if __name__ == '__main__':
    # train(epochs=3000)
    test(save_targets=True)
