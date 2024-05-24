import os
import numpy as np
import pandas as pd
import pyvista as pv

from mesh4d.analyse import measure, crave
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
        landmark2: str,
        acute_angle: bool = False,
    ) -> float:
    v1 = label.coord(df_local, file, landmark1) - label.coord(df_local, file, landmark_origin)
    v2 = label.coord(df_local, file, landmark2) - label.coord(df_local, file, landmark_origin)
    cos = v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)
    a =  np.arccos(cos) / np.pi * 180

    if a >= 90:
        return 180 - a
    else:
        return a

def circ_pass_landmark(
        df_local: pd.DataFrame,
        file: str,
        mesh_local: pv.core.pointset.PolyData,
        landmark: str,
        norm_axis: np.array,
        full_return: bool = False,
    ) -> float:
    # estimate circumference plane
    axis2norm = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1]),
    }

    norm, center = axis2norm[norm_axis], label.coord(df_local, file, landmark)

    # calculate circumference
    cir_ls = []
    boundary_ls = []

    for invert in [True, False]:
        mesh_clip = mesh_local.clip(norm, origin=center, invert=invert)
        boundary = crave.fix_pvmesh_disconnect(
            mesh_clip.extract_feature_edges(
                boundary_edges=True, 
                feature_edges=False, 
                manifold_edges=False,
                )
            )
        arc = boundary.compute_arc_length()
        cir_ls.append(sum(arc['arc_length']))
        boundary_ls.append(boundary)

    idx = np.argmin(cir_ls)
    boundary = boundary_ls[idx]
    cir = cir_ls[idx]

    if full_return:
        return cir, boundary
    else:
        return cir

def circ_pass_2landmarks(
        df_local: pd.DataFrame,
        file: str,
        mesh_local: pv.core.pointset.PolyData,
        landmark_ls: list,
        tangent_axis: str,
        full_return: bool = False,
    ) -> float:
    # estimate circumference plane
    axis2tangent = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1]),
    }

    points = label.slice(df_local, file, landmark_ls).values
    link = points[0] - points[1]
    norm, center = np.cross(link, axis2tangent[tangent_axis]), points[1]

    # calculate circumference
    cir_ls = []
    boundary_ls = []

    for invert in [True, False]:
        mesh_clip = mesh_local.clip(norm, origin=center, invert=invert)
        boundary = crave.fix_pvmesh_disconnect(
            mesh_clip.extract_feature_edges(
                boundary_edges=True, 
                feature_edges=False, 
                manifold_edges=False,
                )
            )
        arc = boundary.compute_arc_length()
        cir_ls.append(sum(arc['arc_length']))
        boundary_ls.append(boundary)

    idx = np.argmin(cir_ls)
    boundary = boundary_ls[idx]
    cir = cir_ls[idx]

    if full_return:
        return cir, boundary
    else:
        return cir

def circ_pass_landmarks(
        df_local: pd.DataFrame,
        file: str,
        mesh_local: pv.core.pointset.PolyData,
        landmark_ls: list,
        full_return: bool = False,
    ) -> float:
    """use with caution and check"""
    points = label.slice(df_local, file, landmark_ls).values
    path_points_ls = []

    for idx in range(len(points) - 1):
        id1 = mesh_local.find_closest_point(points[idx])
        id2 = mesh_local.find_closest_point(points[idx + 1])
        path = mesh_local.geodesic(id1, id2)
        path_points_ls.append(np.array(path.points))

    path_points = np.concatenate(path_points_ls)

    # estimate circumference plane
    norm, center = measure.estimate_plane_from_points(path_points)

    # calculate circumference
    cir_ls = []
    boundary_ls = []

    for invert in [True, False]:
        mesh_clip = mesh_local.clip(norm, origin=center, invert=invert)
        boundary = crave.fix_pvmesh_disconnect(
            mesh_clip.extract_feature_edges(
                boundary_edges=True, 
                feature_edges=False, 
                manifold_edges=False,
                )
            )
        arc = boundary.compute_arc_length()
        cir_ls.append(sum(arc['arc_length']))
        boundary_ls.append(boundary)

    idx = np.argmin(cir_ls)
    boundary = boundary_ls[idx]
    cir = cir_ls[idx]

    if full_return:
        return cir, boundary
    else:
        return cir

# foot measurement metrics
def fl(
        df_local: pd.DataFrame,
        file: str,
    ) -> float:
    """foot length"""
    return dist_along_axis(df_local, file, 'P1', 'P10', 'x')

def mbl(
        df_local: pd.DataFrame,
        file: str,
    ) -> float:
    """medial ball length"""
    return dist_along_axis(df_local, file, 'P4', 'P10', 'x')

def lbl(
        df_local: pd.DataFrame,
        file: str,
    ) -> float:
    """lateral ball length"""
    return dist_along_axis(df_local, file, 'P5', 'P10', 'x')

def abw(
        df_local: pd.DataFrame,
        file: str,
    ) -> float:
    """anatomical ball width"""
    return dist(df_local, file, 'P4', 'P5')

def obw(
        df_local: pd.DataFrame,
        file: str,
    ) -> float:
    """orthogonal ball width"""
    return dist_along_axis(df_local, file, 'P4', 'P3', 'y')

def ohw(
        df_local: pd.DataFrame,
        file: str,
    ) -> float:
    """orthogonal heel width"""
    return dist_along_axis(df_local, file, 'P9', 'P8', 'y')

def bh(
        df_local: pd.DataFrame,
        file: str,
    ) -> float:
    """ball heigh"""
    return height(df_local, file, 'P6')

def ih(
        df_local: pd.DataFrame,
        file: str,
    ) -> float:
    """instep height"""
    return height(df_local, file, 'P7')

def ba(
        df_local: pd.DataFrame,
        file: str,
    ) -> float:
    """ball angle"""
    return angle(df_local, file, 'P4', 'P5', 'P8')

def t1a(
        df_local: pd.DataFrame,
        file: str,
    ) -> float:
    """toe 1 angle"""
    return  angle(df_local, file, 'P4', 'P2', 'P8')

def t5a(
        df_local: pd.DataFrame,
        file: str,
    ) -> float:
    """toe 5 angle"""
    return angle(df_local, file, 'P5', 'P3', 'P9')

def abg(
        df_local: pd.DataFrame,
        file: str,
        mesh_local: pv.core.pointset.PolyData,
    ) -> float:
    """anatomical ball girth"""
    return circ_pass_2landmarks(df_local, file, mesh_local, ['P4', 'P5'], 'z')

def ig(
        df_local: pd.DataFrame,
        file: str,
        mesh_local: pv.core.pointset.PolyData,
    ) -> float:
    """instep girth"""
    return circ_pass_landmark(df_local, file, mesh_local, 'P6', 'x')

# result operations
def combine_measurement_csv(folder):
    files = os.listdir(folder)
    files = [os.path.join(folder, f) for f in files if '.csv' in f]
    files.sort()

    df_ls = []

    for file in files:
        df = pd.read_csv(file)
        df_ls.append(df)

    df = pd.concat(df_ls).set_index('file')
    return df[~df.index.duplicated(keep='first')]