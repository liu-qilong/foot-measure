import numpy as np
import pandas as pd
import pyvista as pv
from sklearn.decomposition import PCA

from mesh4d.analyse import crave
from measure import label

# pca operations
def pca_axes(vertices):
    pca = PCA()
    pca.fit(vertices)

    stds = np.sqrt(pca.explained_variance_)
    axes = pca.components_  # (n_axes, n_coords)
    mean = pca.mean_ # (n_coords)
    return mean, axes, stds

def draw_pca(mean, axes, stds, mesh, is_export: bool = False, export_path: str = ''):
    scene = pv.Plotter()

    for i in range(3):
        start_point = mean
        end_point = mean + 3 * stds[i] * axes[i]
        scene.add_lines(np.array([start_point, end_point]), color='goldenrod')
        scene.add_point_labels(
            end_point, [f"PC{i + 1}@({axes[i][0]:.2f}, {axes[i][1]:.2f}, {axes[i][2]:.2f})var{stds[i]:.2f}"], 
            font_size=15, point_size=0, shape='rounded_rect',
            point_color='goldenrod', always_visible=True,
            )

    scene.add_mesh(mesh, opacity=0.5)
    scene.camera_position = 'zy'
    
    if is_export:
        scene.screenshot(export_path)
    else:
        scene.show()

def draw_axes(origin, axes, mesh, len=150, names=['X', 'Y', 'Z'], is_export: bool = False, export_path: str = ''):
    scene = pv.Plotter()

    for i in range(3):
        start_point = origin
        end_point = origin + len * axes[i]
        scene.add_lines(np.array([start_point, end_point]), color='goldenrod')
        scene.add_point_labels(
            end_point,
            [f"{names[i]}@({axes[i][0]:.2f}, {axes[i][1]:.2f}, {axes[i][2]:.2f})"], 
            font_size=15, point_size=0, shape='rounded_rect',
            point_color='goldenrod', always_visible=True,
            )
    
    scene.add_point_labels(
            origin,
            [f"O@({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})"],
            font_size=15, point_size=0, shape='rounded_rect',
            point_color='goldenrod', always_visible=True,
            )

    scene.add_mesh(mesh, opacity=0.5)
    scene.camera_position = 'zy'
    
    if is_export:
        scene.screenshot(export_path)
    else:
        scene.show()

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
def foot_clip(
        mesh: pv.core.pointset.PolyData,
        df: pd.DataFrame,
        file: str,
        clip_landmarks: list = ['P7', 'P11', 'P12'],
        margin: float = -10,
        invert: bool = True,
        ) -> pv.core.pointset.PolyData:
    df_contour = label.slice(df, [file], clip_landmarks)
    return crave.clip_mesh_with_contour(mesh, df_contour.values, clip_bound='', margin=margin, invert=invert)

def estimate_foot_frame(
        mesh: pv.core.pointset.PolyData,
        file: str,
        df: pd.DataFrame,
        clip_landmarks: list = ['P2', 'P3', 'P4', 'P5', 'P8', 'P9'],
        ):
    def axis_flip_to_align_link(axis, start, end):
        link = label.coord(df, file, end) - label.coord(df, file, start)
        cos = (link @ axis) / np.linalg.norm(link) / np.linalg.norm(axis)
        return np.sign(cos) * axis, np.sign(cos)
    
    # use clipped foot bottom to estimate x-axis (frontal direction)
    mesh_clip = foot_clip(mesh, df, file, clip_landmarks)
    origin, axes, _ = pca_axes(mesh_clip.points)
    x_axis, _ = axis_flip_to_align_link(axes[0], 'P10', 'P1')  # set x-axis (aligned to P10-P1 direction)

    # use whole foot with leg to estimate y-axis (sided direction) and z-axis (vertical direction)
    _, axes, _ = pca_axes(mesh.points)
    y_axis = axes[-1]  # set y-axis
    z_axis, sign = axis_flip_to_align_link(np.cross(x_axis, y_axis), 'P8', 'P11')  # set z-axis (aligned to P8-P11 direction)
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