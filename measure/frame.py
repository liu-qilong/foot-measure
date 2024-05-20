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
    vertices_homo = coord_cart2homo(vertices)
    vertices_transform = (matrix @ vertices_homo.T).T
    return coord_homo2cart(vertices_transform)

# foot local frame
def foot_clip(
        mesh: pv.core.pointset.PolyData,
        df: pd.DataFrame,
        file: str,
        clip_landmarks: list = ['P7', 'P10', 'P11', 'P12'],
        ) -> pv.core.pointset.PolyData:
    df_contour = label.slice(df, [file], clip_landmarks)
    return crave.clip_mesh_with_contour(mesh, df_contour.values, clip_bound='', margin=0, invert=True)

def estimate_foot_frame(
        mesh: pv.core.pointset.PolyData,
        file: str, df: pd.DataFrame,
        clip_landmarks: list = ['P7', 'P10', 'P11', 'P12'],
        ):
    axes_frame = np.zeros((3, 3))  # (axes, coord)
    
    # use whole foot with leg to estimate y-axis (sided direction)
    _, axes, _ = pca_axes(mesh.points)
    axes_frame[1] = axes[-1]  # set y-axis

    # use clipped foot to estimate x-axis (frontal direction) and z-axis (vertical direction)
    mesh_clip = foot_clip(mesh, df, file, clip_landmarks)
    mean, axes, _ = pca_axes(mesh_clip.points)
    axes_frame[0] = axes[0]  # set x-axis
    axes_frame[2] = np.cross(axes_frame[0], axes_frame[1])  # set z-axis

    # transform mesh to local frame
    mat2local = trans_mat2local(axes_frame, mean)
    mat2global = trans_mat2global(axes_frame, mean)
    mesh_local = mesh.transform(mat2local, inplace=False)

    # set origin as the ground (lowest) point of foot (in local frame)
    min_idx = mesh_local.points[:, -1].argmin()
    ground = mesh_local.points[min_idx]
    origin = transform(mat2global, np.array([ground]))[0]

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