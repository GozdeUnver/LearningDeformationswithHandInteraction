import os

import torch

import numpy as np
import open3d as o3d

from torch_geometric.loader import DataLoader

from CubeDataset import CubeDataset
from model import MeshGNN

Vec3d = o3d.utility.Vector3dVector
Vec3i = o3d.utility.Vector3iVector


CONFIG_NAME = 'sageconv_cube_2xsageconv'

outputs_folder_path = os.path.join('outputs', CONFIG_NAME)
models_folder_path = os.path.join(outputs_folder_path, 'models')
model_file_path = os.path.join(models_folder_path, 'model.t7')
meshes_folder_path = os.path.join(outputs_folder_path, 'predicted_meshes')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = torch.nn.L1Loss()


def create_train_data_loader():
    dataset = CubeDataset(preload=True, split='train')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader


def create_test_data_loader():
    dataset = CubeDataset(preload=True, split='test')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader


def prepare_graph(graph):
    nodes, edges = graph
    return (
        nodes.to(device).squeeze(0),
        edges.to(device).squeeze(0)
    )

def train(epochs = 1000, print_every = 10, tolerance=0):
    model = MeshGNN().to(device)

    # parameters = list(gcn_conv.parameters()) + list(edge_conv.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_loss = torch.inf
    best_model = model.state_dict()

    dataloader = create_train_data_loader()

    if not os.path.exists(models_folder_path):
        os.makedirs(models_folder_path)
    waiting = 0
    new_best = False
    for epoch in range(epochs):
        epoch_loss = 0.0
        count = 0
        for batch, (input_graph, target_points) in enumerate(dataloader):
            count += input_graph[0].shape[0]
            input_nodes, input_edges = prepare_graph(input_graph)
            pred_points = torch.stack([
                model(x, edge_index) \
                    for x, edge_index in zip(input_nodes, input_edges)
            ])
            target_points = target_points.to(device).squeeze(0)

            optimizer.zero_grad()
            loss = loss_fn(pred_points, target_points)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        waiting += 1
        epoch_loss /= count

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            torch.save(best_model, model_file_path)
            new_best = True
            waiting = 0
        if epoch % print_every == 0:
            print(f'Epoch {epoch}: Loss={epoch_loss}' +
                  ('\tNew best!' if new_best else ''))
            new_best = False
        
        if waiting >= tolerance > 0:
            break
            


def test(save_targets=False):
    model = MeshGNN().to(device)
    state_dict = torch.load(model_file_path, map_location=device)
    model.load_state_dict(state_dict)
    dataloader = create_test_data_loader()
    if not os.path.exists(meshes_folder_path):
        os.makedirs(meshes_folder_path)

    for i, (batch_input_faces, input_graph, batch_target_points) in enumerate(dataloader):
        input_nodes, input_edges = prepare_graph(input_graph)
        batch_pred_points = torch.stack([
            model(x, edge_index) \
                for x, edge_index in zip(input_nodes, input_edges)
        ])

        batch_target_points = batch_target_points.to(device).squeeze(0)
        loss = loss_fn(batch_pred_points, batch_target_points).item()
        print(f'Sample {i}: Loss={loss}')

        for j, (pred_points, input_faces, target_points) in enumerate(zip(batch_pred_points, batch_input_faces, batch_target_points)):
            pred_mesh = o3d.geometry.TriangleMesh(
                Vec3d(pred_points.cpu().detach().numpy()),
                Vec3i(input_faces.squeeze().cpu().detach().numpy())
            )
            mesh_file_path = os.path.join(meshes_folder_path, f'{i}_{j}_pred.obj')
            o3d.io.write_triangle_mesh(mesh_file_path, pred_mesh)
            if save_targets:
                pred_mesh.vertices = Vec3d(target_points.cpu().detach().numpy())
                mesh_file_path = os.path.join(meshes_folder_path, f'{i}_{j}_target.obj')
                o3d.io.write_triangle_mesh(mesh_file_path, pred_mesh)

if __name__ == '__main__':
    train(epochs=3000, tolerance=100)
    test(save_targets=True)
