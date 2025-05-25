import os
import argparse
from pathlib import Path
import pandas as pd
import pyvista as pv

from mesh4d.analyse import crave
from measure import label, frame, metric, visual

def run(
        mesh_folder = 'data/stl',
        output_folder = 'same_as_mesh_folder',
        # output_folder = 'output',
        start = None,
        end = None,
        stride = 1,
    ):
    if output_folder == 'same_as_mesh_folder':
        output_folder = mesh_folder

    # get file list
    files = list(Path(mesh_folder).glob(f'*.obj')) + list(Path(mesh_folder).glob(f'*.stl'))
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

        # local frame
        mesh = crave.fix_pvmesh_disconnect(pv.read(file), df.values)
        axes_frame, origin = frame.estimate_foot_frame(mesh, df)
        mesh_clip = frame.foot_clip(mesh, df)
        mesh_local = frame.foot2local(mesh_clip, axes_frame, origin)
        df_local = frame.df2local(df, axes_frame, origin)

        # visual check
        visual.plot_axes(origin, axes_frame, mesh_clip)

        # metrics
        results = {
            'file': ['description', str(file)],
            'FL': ['foot length (mm)', metric.fl(df_local)],
            'MBL': ['medial ball length (mm)', metric.mbl(df_local)],
            'LBL': ['lateral ball length (mm)', metric.lbl(df_local)],
            'ABW': ['anatomical ball width (mm)', metric.abw(df_local)],
            'OBW': ['orthogonal ball width (mm)', metric.obw(df_local)],
            'OHW': ['orthogonal heel width (mm)', metric.ohw(df_local)],
            'BH': ['ball heigh (mm)', metric.bh(df_local)],
            'IH': ['instep height (mm)', metric.ih(df_local)],
            'BA': ['ball angle (°)', metric.ba(df_local)],
            'T1A': ['toe 1 angle (°)', metric.t1a(df_local)],
            'T5A': ['toe 5 angle (°)', metric.t5a(df_local)],
            'ABG': ['anatomical ball girth (mm)', metric.abg(df_local, mesh_local)],
            'IG': ['instep girth (mm)', metric.ig(df_local, mesh_local)],
        }
        
        file_name = file.stem
        df_auto = pd.DataFrame(results).set_index('file')
        df_auto.to_csv(os.path.join(output_folder, f'{file_name}.csv')) # <-- editing
        print(f'measurements exported to {Path(output_folder)/f"{file_name}.csv"}')

        # combine all results so far
        df_all = metric.combine_measurement_csv(output_folder)
        df_all.to_csv(os.path.join(output_folder, 'measurements.csv'))
        print(f'combined all measurements to {Path(output_folder)/"measurements.csv"}')
        print("-"*20)

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh-folder', type=str, required=True, default='data/stl')
    parser.add_argument('--output-folder', type=str, required=True, default='same_as_mesh_folder')
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()

    # run the function
    run(
        mesh_folder = args.mesh_folder,
        output_folder = args.output_folder,
        start = args.start,
        end = args.end,
        stride = args.stride,
    )