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
files = list(Path(mesh_folder).glob(f'*{mesh_type}'))
files.sort()

if start == None:
    start = 0

if end == None:
    end = len(files) - 1

for idx in range(start, end + 1, stride):
    # landmarks labelling
    file = files[idx]
    df = label.label(
            file = file,
            point_names = [f'P{idx + 1}' for idx in range(12)],
            use_texture = False,
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
    mesh = crave.fix_pvmesh_disconnect(pv.read(file), df.values)
    axes_frame, origin = frame.estimate_foot_frame(mesh, df)
    mesh_clip = frame.foot_clip(mesh, df)
    mesh_local = frame.foot2local(mesh_clip, axes_frame, origin)
    df_local = frame.df2local(df, axes_frame, origin)

    # visual check
    visual.plot_axes(origin, axes_frame, mesh_clip)

    # metrics
    results.append(
        {
            'file': str(file),
            'FL': metric.fl(df_local),
            'MBL': metric.mbl(df_local),
            'LBL': metric.lbl(df_local),
            'ABW': metric.abw(df_local),
            'OBW': metric.obw(df_local),
            'OHW': metric.ohw(df_local),
            'BH': metric.bh(df_local),
            'IH': metric.ih(df_local),
            'BA': metric.ba(df_local),
            'T1A': metric.t1a(df_local),
            'T5A': metric.t5a(df_local),
            'ABG': metric.abg(df_local, mesh_local),
            'IG': metric.ig(df_local, mesh_local),
        }
    )
    
    file_name = file.stem
    df_auto = pd.DataFrame(results).set_index('file')
    df_auto.to_csv(os.path.join(output_folder, f'{file_name}.csv')) # <-- editing
    print(f'measurements exported to {Path(output_folder)/f"{file_name}.csv"}')

    # combine all results so far
    df_all = metric.combine_measurement_csv(output_folder)
    df_all.to_csv(os.path.join(output_folder, 'measurements.csv'))
    print(f'combined all measurements to {Path(output_folder)/"measurements.csv"}')
    print("-"*20)