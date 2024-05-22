import numpy as np
import pandas as pd

from mesh4d.analyse import crave

def label(
        mesh_folder: str,
        start: int = 0,
        end: int = 0,
        stride: int = 1,
        point_names: str = None,
        file_type: str = 'obj',
        use_texture: bool = False, 
        export_folder: str = 'output',
        export_name: str = 'label',
        ):
    """manual labeling of landmarks on a mesh sequence"""
    # labelling
    landmarks, files = crave.landmarks_labelling(
        mesh_folder = mesh_folder,
        start = start,
        end = end,
        stride = stride,
        point_names = point_names,
        file_type = file_type,
        use_texture = use_texture,
        export_folder = export_folder,
        export_name = export_name,
    )

    # arrange as dataframe
    val_ls = []
    id2xyz = {0: 'x', 1:'y', 2: 'z'}
    arr, names = landmarks.to_array()

    for index, val in np.ndenumerate(arr):
        val_ls.append({'file': files[index[0]], 'landmark': names[index[1]], 'coord': id2xyz[index[2]], 'value': val})

    df = pd.DataFrame(val_ls)
    
    try:
        df = df.pivot(index=('file', 'landmark'), columns='coord', values='value')
    except:
        print('WARNING: unable to pivot!')

    # export as pkl
    df.to_pickle(f'{export_folder}/{export_name}.pkl')
    print(f'landmarks exported to {export_folder}/{export_name}.pkl')

    return df

def get_file_ls(df: pd.DataFrame) -> pd.DataFrame:
    return df.index.get_level_values(0).drop_duplicates()

def get_landmark_ls(df: pd.DataFrame) -> pd.DataFrame:
    return df.index.get_level_values(1).drop_duplicates()

def slice(df: pd.DataFrame, file_ls: list, landmark_ls: list) -> pd.DataFrame:
    return df.loc[pd.IndexSlice[file_ls, landmark_ls], :]

def coord(df: pd.DataFrame, file: str, landmark: str) -> np.array:
    return slice(df, file, landmark).values

def axis_coord(df, file, landmark, axis):
    axis2idx = {
        'x': 0,
        'y': 1,
        'z': 2
    }
    return coord(df, file, landmark)[axis2idx[axis]]

if __name__ == '__main__':
    df = label(
        mesh_folder = '/Users/knpob/Territory/Kolmo/code/foot-measure/data',
        start = 0,
        end = 4,
        stride = 1,
        point_names = [f'P{idx + 1}' for idx in range(12)],
        file_type = 'obj',
        use_texture = False,
        export_folder = 'data',
        export_name = 'label',
    )