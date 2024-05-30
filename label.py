# meta info
mesh_folder = '/Users/knpob/Territory/Kolmo/code/foot-measure/data/stl'
mesh_type = '.stl'

output_folder = mesh_folder
# output_folder = 'output'
start = None
end = None
stride = 1

import os
from pathlib import Path
from measure import label

# get file list
files = os.listdir(mesh_folder)
files = [os.path.join(mesh_folder, f) for f in files if mesh_type in f]
files.sort()

if start == None:
    start = 0

if end == None:
    end = len(files)

for idx in range(start, end):
    # landmarks labelling
    file = files[idx]
    file_name = Path(file).stem

    df = label.label(
            mesh_folder = mesh_folder,
            start = idx,
            end = idx,
            stride = stride,
            point_names = [f'P{idx + 1}' for idx in range(12)],
            file_type = mesh_type,
            use_texture = False,
            export_folder = output_folder,
            export_name = file_name,
        )