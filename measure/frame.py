import numpy as np
import pandas as pd
import pyvista as pv
from sklearn.decomposition import PCA

from mesh4d.analyse import crave, measure
from measure import label

# pca operations
def pca_axes(vertices):
    pca = PCA()
    pca.fit(vertices)

    stds = np.sqrt(pca.explained_variance_)
    axes = pca.components_  # (n_axes, n_coords)
    mean = pca.mean_ # (n_coords)
    return mean, axes, stds

# coordinates operations
def coord_cart2homo(vertices):
    shape = list(vertices.shape)
    shape[1] += 1

    vertices_homo = np.ones(shape)
    vertices_homo[:, :3] = vertices
    return vertices_homo

def coord_homo2cart(vertices_homo):
    return vertices_homo[:, :3] / vertices_homo[:, [-1]]

def trans_mat2global(axes, origin):
    """e(i) -> a_i + t"""
    matrix = np.eye(4)
    matrix[:3, :3] = axes.T
    matrix[:3, 3] = origin
    
    return matrix

def trans_mat2local(axes, origin):
    """a_i + t -> e(i)"""
    matrix2golbal = trans_mat2global(axes, origin)
    return np.linalg.inv(matrix2golbal)

def transform(matrix, vertices):
    if len(vertices.shape) == 1:
        # input is only one point of shape (3,)
        vertices = np.array([vertices])
        one_point_mode = True
    else:
        # input is multiple points of shape (N, 3)
        one_point_mode = False

    vertices_homo = coord_cart2homo(vertices)
    vertices_transform = (matrix @ vertices_homo.T).T

    if one_point_mode:
        # output is only one point of shape (3,)
        return coord_homo2cart(vertices_transform)[0]
    else:
        # output is multiple points of shape (N, 3)
        return coord_homo2cart(vertices_transform)

# foot local frame
def plantar_clip(
    mesh: pv.core.pointset.PolyData,
    df: pd.DataFrame,
    file: str,
    clip_landmarks: list = ['P2', 'P3', 'P4', 'P5', 'P8', 'P9'],
    margin: float = 0,
    ) -> pv.core.pointset.PolyData:
    # estimate clipping plane
    df_contour = label.slice(df, [file], clip_landmarks)
    norm, center = measure.estimate_plane_from_points(df_contour.values)

    # estimate cos<norm, po-p6>
    pop6 = label.coord(df, file, 'P6') - center
    cos = pop6 @ norm / np.linalg.norm(pop6) / np.linalg.norm(norm)

    # when cos > 0, then p6 is on the norm side of the plan
    # to clip out the plantar area, which excludes p6, invert should be true
    # vice versa when cos > 0
    return crave.clip_mesh_with_plane(mesh, norm, center, margin, invert=(cos > 0))

def foot_clip(
        mesh: pv.core.pointset.PolyData,
        df: pd.DataFrame,
        file: str,
        clip_landmarks: list = ['P7', 'P11', 'P12'],
        margin: float = -10,
        ) -> pv.core.pointset.PolyData:
    # estimate clipping plane
    df_contour = label.slice(df, [file], clip_landmarks)
    norm, center = measure.estimate_plane_from_points(df_contour.values)

    # estimate cos<norm, po-p6>
    pop6 = label.coord(df, file, 'P6') - center
    cos = pop6 @ norm / np.linalg.norm(pop6) / np.linalg.norm(norm)

    # when cos > 0, then p6 is on the norm side of the plan
    # to clip out the foot area, which includes p6, invert should be false
    # vice versa when cos > 0
    return crave.clip_mesh_with_plane(mesh, norm, center, margin, invert=not(cos > 0))

def estimate_foot_frame(
        mesh: pv.core.pointset.PolyData,
        file: str,
        df: pd.DataFrame,
        clip_landmarks: list = ['P2', 'P3', 'P4', 'P5', 'P8', 'P9'],
        **kwargs
        ):
    def axis_flip_to_align_link(axis, start, end):
        link = label.coord(df, file, end) - label.coord(df, file, start)
        cos = (link @ axis) / np.linalg.norm(link) / np.linalg.norm(axis)
        return np.sign(cos) * axis, np.sign(cos)
    
    # use clipped foot bottom to estimate x-axis (frontal direction)
    mesh_clip = plantar_clip(mesh, df, file, clip_landmarks, **kwargs)
    origin, axes, _ = pca_axes(mesh_clip.points)
    x_axis, _ = axis_flip_to_align_link(axes[0], 'P10', 'P1')  # set x-axis as the 1st PC and align it to P10-P1 direction
    y_axis = axes[1] # set y-axis
    z_axis, sign = axis_flip_to_align_link(np.cross(x_axis, y_axis), 'P8', 'P11')  # set z-axis and align it to P8-P11 direction
    y_axis = sign * y_axis  # adjust y-axis according to weather z-axis is flipped
    axes_frame = np.array([x_axis, y_axis, z_axis])  # (axes, coord)

    # transform mesh to local frame
    mat2local = trans_mat2local(axes_frame, origin)
    mat2global = trans_mat2global(axes_frame, origin)
    mesh_local = mesh.transform(mat2local, inplace=False)

    # estimate ground (lowest) point under local frame
    min_idx = mesh_local.points[:, -1].argmin()
    ground = mesh_local.points[min_idx]

    # push origin to the ground level
    origin_local = transform(mat2local, origin)
    origin_local[2] = ground[2]
    origin = transform(mat2global, origin_local)
    origin = measure.nearest_points_from_plane(mesh, origin)
    
    return axes_frame, origin

def foot2local(mesh: pv.core.pointset.PolyData, axes_frame: np.array, origin: np.array) -> pv.core.pointset.PolyData:
    mat2local = trans_mat2local(axes_frame, origin)
    return mesh.transform(mat2local, inplace=False)

def df2local(df: pd.DataFrame, axes_frame: np.array, origin: np.array) -> pv.core.pointset.PolyData:
    arr_local = transform(
        trans_mat2local(axes_frame, origin),
        df.values
    )

    return pd.DataFrame(arr_local, columns=df.columns, index=df.index)