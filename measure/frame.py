import numpy as np
import pandas as pd
import pyvista as pv
from sklearn.decomposition import PCA

from mesh4d.analyse import crave
from measure import label

def foot_clip(
        mesh: pv.core.pointset.PolyData,
        df: pd.DataFrame,
        file: str,
        clip_landmarks: list = ['P7', 'P10', 'P11', 'P12'],
        ) -> pv.core.pointset.PolyData:
    df_contour = label.slice(df, [file], clip_landmarks)
    return crave.clip_mesh_with_contour(mesh, df_contour.values, clip_bound='', margin=0, invert=True)

def pca_axes(vertices):
    pca = PCA()
    pca.fit(vertices)

    stds = np.sqrt(pca.explained_variance_)
    axes = pca.components_  # (n_axes, n_coords)
    mean = pca.mean_ # (n_coords)
    return mean, axes, stds

def draw_axes(mean, axes, stds, mesh, is_export: bool = False, export_path: str = ''):
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

def estimate_foot_frame(mesh: pv.core.pointset.PolyData, file: str, df: pd.DataFrame):
    axes_frame = np.zeros((3, 3))  # (axes, coord)
    
    # use whole foot with leg to estimate y-axis (sided direction)
    _, axes, _ = pca_axes(mesh.points)
    axes_frame[1] = axes[-1]  # set y-axis

    # use clipped foot to estimate x-axis (frontal direction) and z-axis (vertical direction) as well as local-frame's origin
    mesh_clip = foot_clip(mesh, df, file, ['P7', 'P11', 'P12'])
    origin, axes, _ = pca_axes(mesh_clip.points)
    axes_frame[0] = axes[0]  # set x-axis
    axes_frame[2] = np.cross(axes_frame[0], axes_frame[1])  # set z-axis

    return axes_frame, origin