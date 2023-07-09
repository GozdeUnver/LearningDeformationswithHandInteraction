## Learning Deformations with Hand Interaction Project
- Correspondences are found by using [pyFM](https://github.com/RobinMagnet/pyFM).

### Non-deformed-to-Deformed Corresponsences
This directory contains the mapping from the non-deformed mesh to the deformed mesh. 

`deformed_<num>_correspondence_<refinemet_method>.npy` stores the correspondence points of the deformed mesh-`<num>`. `<num>` is greater than 1.

`non_deformed_<num>_correspondence_<refinemet_method>.npy` stores the correspondence points of the non-deformed mesh-1. (This array stores the correspondences that mesh-1 makes with mesh-`<num>`)

### Landmarks
`<toy_name>_<non-deformed_mesh_id>_<num>_landmarks_deformed_and_non.txt` stores two columns of indices of correspondence. The first column belongs to the vertices of the `deformed`mesh and the second column belongs to the `non-deformed`mesh. `<non-deformed_mesh_id>`=1, `<num>`is greater than 1.

`<toy_name>_<non-deformed_mesh_id>_<num>_landmarks_non_and_deformed.txt` stores two columns of indices of correspondences. The first column belongs to the vertices of the `non-deformed` mesh and the second column belongs to the `deformed` mesh. `<non-deformed_mesh_id>`=1, `<num>`is greater than 1.

### Mesh decimation
In the data/dataset directory, the downsampled versions of meshes are stored. After decimation, all meshes have 70000 faces and the new meshes are stored with the name `push_toy_<num>_70000.obj` for BlueToy02 and `yellow_push_toy_<num>_70000.obj` for YellowToy01. The original meshes can still be found in the same directory. In the correspondence finding step, meshes with 70000 faces were used only.
