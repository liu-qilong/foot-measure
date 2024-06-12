import os
from pathlib import Path
import numpy as np
import pandas as pd

from mesh4d.analyse import crave

def label(
        file: str,
        point_names: str = None,
        export_folder: str = None,
        export_name: str = None,
        **kwargs,
        ):
    """manual labeling of landmarks on a mesh sequence"""
    # default value
    if export_folder is None:
        export_folder = Path(file).parent
    
    if export_name is None:
        export_name = Path(file).stem

    # labelling
    print("-"*20)
    print(f"labelling mesh file: {file}")
    points = crave.mesh_pick_points_with_check(file, point_names, **kwargs)

    # arrange as dataframe
    val_ls = []
    id2xyz = {0: 'x', 1:'y', 2: 'z'}

    for index, val in np.ndenumerate(points):
        val_ls.append({'landmark': point_names[index[0]], 'coord': id2xyz[index[1]], 'value': val})

    df = pd.DataFrame(val_ls)
    
    try:
        df = df.pivot(index='landmark', columns='coord', values='value')
    except:
        print('WARNING: unable to pivot!')

    # export as pkl
    export_path = export_folder / export_name.with_suffix('.pkl')
    df.to_pickle(export_path)
    print(f'landmarks exported to {export_path}')

    return df

def get_landmark_ls(df: pd.DataFrame) -> pd.DataFrame:
    return df.index

def slice(df: pd.DataFrame, landmark_ls: list) -> pd.DataFrame:
    return df.loc[pd.IndexSlice[landmark_ls], :]

def coord(df: pd.DataFrame, landmark: str) -> np.array:
    return slice(df, landmark).values

def axis_coord(df, landmark, axis):
    axis2idx = {
        'x': 0,
        'y': 1,
        'z': 2
    }
        
    return coord(df, landmark)[axis2idx[axis]]

if __name__ == '__main__':
    df = label(
        file='data/stl/020_l.stl',
        point_names = [f'P{idx + 1}' for idx in range(12)],
        use_texture = False,
        # export_folder = 'output',
        export_name = 'test',
    )