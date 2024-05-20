import numpy as np
import pandas as pd

from measure import label

# metric types
def dist_along_axis(
        df_local: pd.DataFrame,
        file: str,
        landmark1: str,
        landmark2: str,
        axis: str,
        is_abs: bool = True,
        ) -> float:
    dist = label.axis_coord(df_local, file, landmark1, axis) - label.axis_coord(df_local, file, landmark2, axis)

    if is_abs:
        return np.abs(dist)
    
    else:
        return dist

def dist(
        df_local: pd.DataFrame,
        file: str,
        landmark1: str,
        landmark2: str,
    ) -> float:
    return np.linalg.norm(
        label.coord(df_local, file, landmark1) - label.coord(df_local, file, landmark2)
        )

def height(
        df_local: pd.DataFrame,
        file: str,
        landmark: str,
    ) -> float:
    return np.abs(label.axis_coord(df_local, file, landmark, 'z'))

def angle(
        df_local: pd.DataFrame,
        file: str, 
        landmark_origin: str,
        landmark1: str,
        landmark2: str,):
    v1 = label.coord(df_local, file, landmark1) - label.coord(df_local, file, landmark_origin)
    v2 = label.coord(df_local, file, landmark2) - label.coord(df_local, file, landmark_origin)
    cos = v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)
    return np.arccos(cos) / np.pi * 180

# foot measurement metrics
def fl(
        df: pd.DataFrame,
        file: str,
    ):
    """foot length"""
    return dist_along_axis(df, file, 'P1', 'P10', 'x')

def mbl(
        df: pd.DataFrame,
        file: str,
    ):
    """medial ball length"""
    return dist_along_axis(df, file, 'P4', 'P10', 'x')

def lbl(
        df: pd.DataFrame,
        file: str,
    ):
    """lateral ball length"""
    return dist_along_axis(df, file, 'P5', 'P10', 'x')

def abw(
        df: pd.DataFrame,
        file: str,
    ):
    """anatomical ball width"""
    return dist(df, file, 'P4', 'P5')

def obw(
        df: pd.DataFrame,
        file: str,
    ):
    """orthogonal ball width"""
    return dist_along_axis(df, file, 'P4', 'P3', 'y')

def ohw(
        df: pd.DataFrame,
        file: str,
    ):
    """orthogonal heel width"""
    return dist_along_axis(df, file, 'P9', 'P8', 'y')

def bh(
        df: pd.DataFrame,
        file: str,
    ):
    """ball heigh"""
    return height(df, file, 'P6')

def ih(
        df: pd.DataFrame,
        file: str,
    ):
    """instep height"""
    return height(df, file, 'P7')

def ba(
        df: pd.DataFrame,
        file: str,
    ):
    """ball angle"""
    return angle(df, file, 'P4', 'P5', 'P8')

def t1a(
        df: pd.DataFrame,
        file: str,
    ):
    """toe 1 angle"""
    return angle(df, file, 'P4', 'P2', 'P8')

def t5a(
        df: pd.DataFrame,
        file: str,
    ):
    """toe 5 angle"""
    return angle(df, file, 'P5', 'P3', 'P9')

