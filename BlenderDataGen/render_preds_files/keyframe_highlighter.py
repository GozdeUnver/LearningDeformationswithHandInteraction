import bpy


# Select mesh and vertex
mesh_name = "softbody"
vertex_index = 77 

# Create a vertex group for the vertex
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.object.mode_set(mode='OBJECT')
mesh = bpy.data.objects[mesh_name]
mesh.data.vertices[vertex_index].select = True
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.object.vertex_group_assign_new()
vertex_group_name = mesh.vertex_groups[-1].name  # Get the name of the new vertex group
bpy.ops.object.mode_set(mode='OBJECT')

# Add a new sphere
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(0, 0, 0))

# Get the newly created sphere
sphere = bpy.data.objects['Sphere']

# Rename the sphere
sphere.name = "highlight_vertex"

# Set its scale
sphere.scale = (1, 1, 1)  # x, y, z scale factors

# Set its location to the vertex's location
# You'll need to set the vertex's object to be the sphere's parent first, as described in the previous scripts.

# Create a new material
mat = bpy.data.materials.new(name="Highlight_Material")

# Set material color
mat.diffuse_color = (1, 0, 0, 1)  # RGBA, red in this case

# Assign it to the sphere
if len(sphere.data.materials) > 0:
    # If the sphere already has a material, just replace the first one
    sphere.data.materials[0] = mat
else:
    # If the sphere has no materials, append the new one
    sphere.data.materials.append(mat)
    
# Parent the sphere to the mesh
sphere.parent = mesh

# Add a Copy Location constraint to the sphere
constraint = sphere.constraints.new('COPY_LOCATION')
constraint.target = mesh
constraint.subtarget = vertex_group_name
