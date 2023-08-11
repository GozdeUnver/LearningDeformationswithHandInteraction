import bpy
from mathutils import Matrix, Vector
import os
import time
from mathutils.bvhtree import BVHTree
import bmesh

softbody = bpy.data.objects['Softbody']
collider = bpy.data.objects['Collider']
group_index = softbody.vertex_groups['DeformationGroup'].index
export_root = '/home/max/Desktop/deform_datasets/plane_dataset/' 
os.makedirs(export_root)
num_frames = 100
max_displacement = 0.5
    
def is_collider_in_softbody(deformed_softbody):
    # Set the collider geometry
    collider_length = 0.04
#    collider_normal = collider.data.polygons[33].normal
#    collider_vertex_local = collider.data.polygons[33].center - collider_normal*0.005
    collider_normal = collider.data.vertices[9682].normal
    collider_vertex_local = collider.data.vertices[9682].co - collider_normal*0.005
    
    #Transform to world coordinates
    collider_vertex_global = collider.matrix_world @ collider_vertex_local
    collider_normal_global = collider.matrix_world.to_3x3() @ collider_normal
    mat = deformed_softbody.matrix_world.inverted()

    # Create a bmesh object and ensure it is in local space
    bm = bmesh.new()
    bm.from_mesh(deformed_softbody.data)
    bm.transform(deformed_softbody.matrix_world)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    
    hit, normal, index, distance = bvh.ray_cast(collider_vertex_global, -collider_normal_global, collider_length)
    return hit is not None

def animate_with_collider_at_vertex(index):
    ### Transform collider ###

    # Make sure the object data is loaded
    bpy.context.view_layer.objects.active = collider
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Get the specific vertex coordinates and normal
    vertex = softbody.data.vertices[index]
    vertex_coord = vertex.co
    vertex_normal = vertex.normal  

    # Transform to world coordinates
    world_coord = softbody.matrix_world @ vertex_coord
    world_normal = softbody.matrix_world.to_3x3() @ vertex_normal
    world_normal.normalize()

    # Move the object to the vertex
    collider.location = world_coord

    # Calculate the rotation matrix
    rot_mat = Vector((0, 0, 1)).rotation_difference(world_normal).to_matrix().to_4x4()

    # Retrieve the original scale of the source object
    source_scale = collider.scale

    # Apply the rotation and preserve the original scale
    collider.matrix_world = rot_mat @ Matrix.Translation(world_coord)
    collider.scale = source_scale

    ### Set collider keypoints ###

    # Delete any existing keypoints
    collider.animation_data_clear()

    # Define a list of keypoints, each keypoint is a tuple (frame_number, location)
    keypoints = [
        (0, world_coord), 
        (num_frames, world_coord-world_normal*max_displacement), 
    ]

    # Set the location and create a keyframe for each keypoint
    for frame_number, location in keypoints:
        # Change the current frame
        bpy.context.scene.frame_set(frame_number)

        # Set the location of the object
        collider.location = location

        # Insert a keyframe for the object's location at the current frame
        collider.keyframe_insert(data_path="location", index=-1)  # index=-1 means all location components (x, y, z)

    # Change the interpolation type of the keyframe points
    for fcurve in collider.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'LINEAR'


    ### Bake the animation ###

    softbody_modifier = softbody.modifiers['Softbody']

    # Clear soft body physics bake cache
    if softbody_modifier.point_cache.is_baked:
        bpy.ops.ptcache.free_bake({'point_cache': softbody_modifier.point_cache})

    # Set the start and end frames
    softbody_modifier.point_cache.frame_start = 0
    softbody_modifier.point_cache.frame_end = num_frames

    # Set the object as the active object
    bpy.context.view_layer.objects.active = softbody

    # Select the object
    bpy.ops.object.select_all(action='DESELECT')
    softbody.select_set(True)

    # Bake the soft body simulation
    bpy.context.scene.frame_set(0)  # Go to the first frame

    # Create context override dictionary for the bake operation
    context_override = {'blend_data': bpy.context.blend_data, 'scene': bpy.context.scene, 'active_object': softbody, 'point_cache': softbody_modifier.point_cache, 'object': softbody}

    # Bake the physics
    bpy.ops.ptcache.bake(context_override, bake=True)

    ### Store the results ###

    # Specify the directory to which you want to export the STL files
    export_dir = export_root+str(vertex_coord)
    os.makedirs(export_dir)

    # Get the start and end frames of the animation
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end

    # Iterate through each frame
    for frame in range(start_frame, end_frame+1):
        # Set the current frame
        bpy.context.scene.frame_set(frame)
        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        # Select the object
        softbody.select_set(True)
        
        # Get the evaluated object
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_obj = softbody.evaluated_get(depsgraph)
        
        # Get the global position of the vertex
        vertex_global = evaluated_obj.matrix_world @ evaluated_obj.data.vertices[index].co
        collider_in_softbody = is_collider_in_softbody(evaluated_obj)
        if collider_in_softbody:
            break
        else:
            # Define the filename for the exported STL
            filename = f"{frame}.stl"
            filepath = os.path.join(export_dir, filename)

            # Export the current frame as STL
            bpy.ops.export_mesh.stl(filepath=filepath, use_selection=True)
            softbody.select_set(False)


### Generate and collect data for all vertices in the group ###

indeces = []
for vertex in softbody.data.vertices:
    for group in vertex.groups:
        if group.group == group_index:
            indeces.append(vertex.index)
indeces = [8]
for vert_i in indeces:
    animate_with_collider_at_vertex(vert_i)
    #time.sleep(10)
