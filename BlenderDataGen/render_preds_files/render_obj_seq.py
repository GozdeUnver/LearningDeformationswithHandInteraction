import bpy
import os
import re

pattern = re.compile(r'^\d+_\d+_.*')

# delete previous objects
for obj in bpy.data.objects:
    # Check if the object's name starts with d_d_
    if pattern.match(obj.name):
        bpy.data.objects.remove(obj)

# directory where the .stl files are
directory = "/home/max/Desktop/deform_datasets/prism_meshgnn/0/gt"
os.chdir(directory)

# read .obj file names from the directory
file_list = sorted(os.listdir(directory))
stl_list = [item for item in file_list if item.endswith('.obj')]

# loop through the .obj files and load them
for item in stl_list:
    frame_num = int(pattern.match(item).group(2))
    path_to_file = os.path.join(directory, item)
    bpy.ops.import_mesh.obj(filepath=path_to_file)

    # get the last imported object
    obj = bpy.context.selected_objects[0]

    # set object hide_render to True for all frames
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_render", frame=0)

    # unhide the object for the specific frame and the next one
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_render", frame=frame_num)
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_render", frame=frame_num + 1)
