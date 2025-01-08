# meta info
# MESH_FOLDER = 'data/olivia/Healthy'
MESH_FOLDER = 'data/olivia/Flat feet'
MESH_TYPE = '.stl'
# MESH_TYPE = '.obj'
OUTPUT_FOLDER = MESH_FOLDER
# OUTPUT_FOLDER = 'output'
EXTRA_MARK = 'extra'
POINT_NAMES = ['MLA-AP', 'MLA-PP', 'MMB', 'LMB', 'AA']
START = None
END = None
STRIDE = 1

import os
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

# get file list
files = list(Path(MESH_FOLDER).glob(f'*{MESH_TYPE}'))
files.sort()

if START == None:
    START = 0

if END == None:
    END = len(files) - 1

for idx in range(START, END + 1, STRIDE):
    # landmarks labelling
    file = files[idx]
    lmrk_extra = label.label(
            file = file,
            point_names = POINT_NAMES,
            export_name = f'{file.stem}-{EXTRA_MARK}',
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
    output_path = Path(OUTPUT_FOLDER)/f"{file_name}-{EXTRA_MARK}.csv"
    df_auto.to_csv(output_path)
    print(f'measurements exported to {output_path}')

    # combine all results so far
    df_all = metric.combine_measurement_csv(
        folder=OUTPUT_FOLDER,
        match_str=EXTRA_MARK,
        exclude_str='measurement',
    )
    output_path = Path(OUTPUT_FOLDER)/f"measurements-{EXTRA_MARK}.csv"
    df_all.to_csv(output_path)
    print(f'combined all measurements to {output_path}')