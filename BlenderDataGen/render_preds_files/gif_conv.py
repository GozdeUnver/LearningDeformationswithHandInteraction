import subprocess
import os
import glob

def create_palette(input_path, palette_path):
    palette_command = ['ffmpeg', '-i', input_path, '-vf', 'palettegen', '-y', palette_path]
    subprocess.call(palette_command)

def apply_palette(input_path, palette_path, output_path):
    apply_command = ['ffmpeg', '-i', input_path, '-i', palette_path, '-lavfi', 'paletteuse', '-y', output_path]
    subprocess.call(apply_command)

directory = '/home/max/Desktop/deform_datasets/videos/'
webm_files = glob.glob(os.path.join(directory, '*.webm'))

for file in webm_files:
    base_name = os.path.basename(file)
    name_without_extension = os.path.splitext(base_name)[0]
    palette_path = f"{name_without_extension}_palette.png"
    output_path = directory+f"{name_without_extension}.gif"

    create_palette(file, palette_path)
    apply_palette(file, palette_path, output_path)

    if os.path.exists(palette_path):  # remove palette file after conversion
        os.remove(palette_path)
