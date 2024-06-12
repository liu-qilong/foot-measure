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
files = list(Path(mesh_folder).glob(f'*{mesh_type}'))
files.sort()

if start == None:
    start = 0

if end == None:
    end = len(files) - 1

for idx in range(start, end + 1):
    # landmarks labelling
    df = label.label(
            file = files[idx],
            point_names = [f'P{idx + 1}' for idx in range(12)],
            use_texture = False,
        )