import argparse
from pathlib import Path
import pandas as pd
import pyvista as pv

from mesh4d.analyse import crave
from measure import label, frame, metric, visual
from measure.metric import height, dist, dist_along_axis

# define extra metrics
def al(
        lmkr_local: pd.DataFrame,
    ):
    """arch length"""
    return dist(lmkr_local, 'MLA-AP', 'MLA-PP')

def ab(
        lmkr_local: pd.DataFrame,
    ):
    """arch breadth"""
    return dist(lmkr_local, 'LMB', 'MMB')

def ad(
        lmkr_local: pd.DataFrame,
    ):
    """arch depth"""
    return dist_along_axis(lmkr_local, 'P4', 'MMB', 'x')

def ah(
        lmkr_local: pd.DataFrame,
    ):
    """arch height"""
    return height(lmkr_local, 'AA')

def run(
        mesh_folder = 'data/olivia/Flat feet',
        output_folder = 'same_as_mesh_folder',
        extra_mark = 'extra',
        point_names = ['MLA-AP', 'MLA-PP', 'MMB', 'LMB', 'AA'],
        start = None,
        end = None,
        stride = 1,
    ):
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
        lmrk_extra = label.label(
                file = file,
                point_names = point_names,
                export_name = f'{file.stem}-{extra_mark}',
                use_texture = False,
            )
        
        lmrk = pd.concat([
            pd.read_pickle(file.with_suffix('.pkl')),
            lmrk_extra,
        ])

        # local frame
        mesh = crave.fix_pvmesh_disconnect(pv.read(file), lmrk.values)
        axes_frame, origin = frame.estimate_foot_frame(mesh, lmrk)
        mesh_clip = frame.foot_clip(mesh, lmrk)
        mesh_local = frame.foot2local(mesh_clip, axes_frame, origin)
        lmrk_local = frame.df2local(lmrk, axes_frame, origin)

        # visual check
        visual.plot_axes(origin, axes_frame, mesh_clip)

        # metrics
        results = {
            'file': ['description', str(file)],
            'AL': ['arch length (mm)', al(lmrk_local)],
            'AB': ['arch breadth (mm)', ab(lmrk_local)],
            'AD': ['arch depth (mm)', ad(lmrk_local)],
            'AH': ['arch height (mm)', ah(lmrk_local)],
        }
        
        file_name = file.stem
        df_auto = pd.DataFrame(results).set_index('file')
        output_path = Path(output_folder)/f"{file_name}-{extra_mark}.csv"
        df_auto.to_csv(output_path)
        print(f'measurements exported to {output_path}')

        # combine all results so far
        df_all = metric.combine_measurement_csv(
            folder=output_folder,
            match_str=extra_mark,
            exclude_str='measurement',
        )
        output_path = Path(output_folder)/f"measurements-{extra_mark}.csv"
        df_all.to_csv(output_path)
        print(f'combined all measurements to {output_path}')

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh-folder', type=str, required=True, default='data/olivia/Flat feet')
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