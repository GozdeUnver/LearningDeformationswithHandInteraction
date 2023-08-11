import bpy
import os
import re

# delete previous objects
for obj in bpy.data.objects:
    # Check if the object's name is just a number
    if re.fullmatch(r'\d+', obj.name):
        bpy.data.objects.remove(obj)

# directory where the .stl files are
directory = "/home/max/Desktop/deform_datasets/prism_meshgnn/0/gt"
os.chdir(directory)

# read .stl file names from the directory
file_list = sorted(os.listdir(directory))
stl_list = [item for item in file_list if item.endswith('.stl')]

# loop through the .stl files and load them
for item in stl_list:
    frame_num = int(os.path.splitext(item)[0])  # get frame number from file name
    path_to_file = os.path.join(directory, item)
    bpy.ops.import_mesh.stl(filepath=path_to_file)

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