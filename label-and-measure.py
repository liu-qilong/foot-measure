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
import pandas as pd
import pyvista as pv

from mesh4d.analyse import crave
from measure import label, frame, metric, visual

# get file list
files = os.listdir(mesh_folder)
files = [os.path.join(mesh_folder, f) for f in files if mesh_type in f]
files.sort()

if start == None:
    start = 0

if end == None:
    end = len(files) - 1

for idx in range(start, end + 1, stride):
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

    # measurement
    results = [
        {
            'file': 'description',
            'FL': 'foot length (mm)',
            'MBL': 'medial ball length (mm)',
            'LBL': 'lateral ball length (mm)',
            'ABW': 'anatomical ball width (mm)',
            'OBW': 'orthogonal ball width (mm)',
            'OHW': 'orthogonal heel width (mm)',
            'BH': 'ball heigh (mm)',
            'IH': 'instep height (mm)',
            'BA': 'ball angle (°)',
            'T1A': 'toe 1 angle (°)',
            'T5A': 'toe 5 angle (°)',
            'ABG': 'anatomical ball girth (mm)',
            'IG': 'instep girth (mm)',
        }
    ]

    # local frame
    mesh = crave.fix_pvmesh_disconnect(pv.read(file))
    axes_frame, origin = frame.estimate_foot_frame(mesh, file, df)
    mesh_clip = frame.foot_clip(mesh, df, file)
    mesh_local = frame.foot2local(mesh_clip, axes_frame, origin)
    df_local = frame.df2local(df, axes_frame, origin)

    # visual check
    visual.plot_axes(origin, axes_frame, mesh_clip)

    # metrics
    results.append(
        {
            'file': file,
            'FL': metric.fl(df_local, file),
            'MBL': metric.mbl(df_local, file),
            'LBL': metric.lbl(df_local, file),
            'ABW': metric.abw(df_local, file),
            'OBW': metric.obw(df_local, file),
            'OHW': metric.ohw(df_local, file),
            'BH': metric.bh(df_local, file),
            'IH': metric.ih(df_local, file),
            'BA': metric.ba(df_local, file),
            'T1A': metric.t1a(df_local, file),
            'T5A': metric.t5a(df_local, file),
            'ABG': metric.abg(df_local, file, mesh_local),
            'IG': metric.ig(df_local, file, mesh_local),
        }
    )

    df_auto = pd.DataFrame(results).set_index('file')
    df_auto.to_csv(os.path.join(output_folder, f'{file_name}.csv'))
    print(f'measurements exported to {Path(output_folder)/f"{file_name}.csv"}')

    # combine all results so far
    df_all = metric.combine_measurement_csv(output_folder)
    df_all.to_csv(os.path.join(output_folder, 'measurements.csv'))
    print(f'combined all measurements to {Path(output_folder)/"measurements.csv"}')
    print("-"*20)