import numpy as np
import pyvista as pv

from mesh4d.analyse import measure
from measure import label, metric

# frame related visuals
def plot_pca(mean, axes, stds, mesh, is_export: bool = False, export_path: str = ''):
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

def plot_axes(origin, axes, mesh, len=150, names=['X', 'Y', 'Z'], is_export: bool = False, export_path: str = ''):
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

# metric types related visuals
def plot_landmarks(scene, df_local, file, landamark_ls, font_size=15, **kwargs):
    points = label.slice(df_local, file, landamark_ls).values
    scene.add_points(points, render_points_as_spheres=True, point_size=10, **kwargs)

    offset = np.array([5, 5, 5])
    scene.add_point_labels(points + offset, landamark_ls, always_visible=True, font_size=font_size)

    return points

def plot_dist_along_axis(scene, df_local, file, landmark1, landmark2, axis, name='dist', unit='mm', font_size=15, **kwargs):
    measurement = metric.dist_along_axis(df_local, file, landmark1, landmark2, axis)
    landmark_points = plot_landmarks(scene, df_local, file, [landmark1, landmark2], font_size, **kwargs)

    # plot axis frame
    dx = metric.dist_along_axis(df_local, file, landmark1, landmark2, 'x', is_abs=False)
    dy = metric.dist_along_axis(df_local, file, landmark1, landmark2, 'y', is_abs=False)
    dz = metric.dist_along_axis(df_local, file, landmark1, landmark2, 'z', is_abs=False)

    points = np.vstack([
        landmark_points[1],
        landmark_points[1] + np.array([dx, 0, 0]),
        landmark_points[1] + np.array([dx, dy, 0]),
        landmark_points[0],
        ])
    pdata = pv.PolyData(points)
    pdata.lines = np.hstack([[2, 0, 1], [2, 1, 2], [2, 2, 3]])
    scene.add_mesh(pdata, line_width=2, **kwargs)

    # plot axis dist
    if axis == 'x':
        points = np.vstack([
            landmark_points[1],
            landmark_points[1] + np.array([dx, 0, 0]),
            ])
        scene.add_point_labels([landmark_points[1] + np.array([dx/2, 0, 0])], [f'{name} = {measurement:.2f} {unit}'], font_size=font_size, always_visible=True)
    
    elif axis == 'y':
        points = np.vstack([
            landmark_points[1] + np.array([dx, 0, 0]),
            landmark_points[1] + np.array([dx, dy, 0]),
            ])
        scene.add_point_labels([landmark_points[1] + np.array([dx, dy/2, 0])], [f'{name} = {measurement:.2f} {unit}'], font_size=font_size, always_visible=True)
        
    elif axis=='z':
        points = np.vstack([
            landmark_points[1] + np.array([dx, dy, 0]),
            landmark_points[0],
            ])
        scene.add_point_labels([landmark_points[1] + np.array([dx, dy, dz/2])], [f'{name} = {measurement:.2f} {unit}'], font_size=font_size, always_visible=True)
        
    pdata = pv.PolyData(points)
    pdata.lines = np.hstack([[2, 0, 1]])
    scene.add_mesh(pdata, line_width=5, **kwargs)

def plot_dist(scene, df_local, file, landmark1, landmark2, name='dist', unit='mm', font_size=15, **kwargs):
    measurement = metric.dist(df_local, file, landmark1, landmark2)
    
    points = plot_landmarks(scene, df_local, file, [landmark1, landmark2], font_size, **kwargs)
    lines = np.hstack([[2, 0, 1]])
    pdata = pv.PolyData(points)
    pdata.lines = lines

    scene.add_mesh(pdata, line_width=5, **kwargs)
    scene.add_point_labels([points.mean(axis=0)], [f'{name} = {measurement:.2f} {unit}'], font_size=font_size, always_visible=True)

def plot_angle(scene, df_local, file, landmark_origin, landmark1, landmark2, unit='°', name='angle', font_size=15, **kwargs):
    measurement = metric.angle(df_local, file, landmark_origin, landmark1, landmark2)

    points = plot_landmarks(scene, df_local, file, [landmark_origin, landmark1, landmark2], **kwargs)
    pdata = pv.PolyData(points)
    pdata.lines = np.hstack([[2, 0, 1], [2, 0, 2]])

    scene.add_mesh(pdata, line_width=3, **kwargs)
    scene.add_point_labels(points.mean(axis=0), [f'{name} = {measurement:.2f}{unit}'], font_size=font_size, always_visible=True)

def plot_height(scene, df_local, file, landmark, name='height', unit='mm', font_size=15, **kwargs):
    measurement = metric.height(df_local, file, landmark)

    point = plot_landmarks(scene, df_local, file, [landmark], **kwargs)[0]
    ground = np.copy(point)
    ground[2] = 0
    points = np.array([point, ground])
    
    lines = np.hstack([[2, 0, 1]])
    pdata = pv.PolyData(points)
    pdata.lines = lines

    scene.add_mesh(pdata, line_width=5, **kwargs)
    scene.add_point_labels([points.mean(axis=0)], [f'{name} = {measurement:.2f} {unit}'], font_size=font_size, always_visible=True)

def plot_circ_pass_landmark(scene, df_local, file, mesh_local, landmark, norm_axis, name='circ', unit='mm', font_size=15, **kwargs):
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
        boundary = mesh_clip.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        cir_ls.append(boundary.length)
        boundary_ls.append(boundary)

    idx = np.argmin(cir_ls)
    boundary = boundary_ls[idx]
    measurement = cir_ls[idx]

    # plot
    plot_landmarks(scene, df_local, file, [landmark], **kwargs)

    scene.add_mesh(boundary, line_width=5, **kwargs)
    scene.add_point_labels([boundary.points.mean(axis=0)], [f'{name} = {measurement:.2f} {unit}'], font_size=font_size, always_visible=True)

def plot_circ_pass_landmarks(scene, df_local, file, mesh_local, landmark_ls, name='circ', unit='mm', font_size=15, **kwargs):
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
        boundary = mesh_clip.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        boundary_ls.append(boundary)
        cir_ls.append(boundary.length)

    idx = np.argmin(cir_ls)
    boundary = boundary_ls[idx]
    measurement = cir_ls[idx]

    # plot
    plot_landmarks(scene, df_local, file, landmark_ls, **kwargs)

    scene.add_mesh(boundary, line_width=5, **kwargs)
    scene.add_point_labels([boundary.points.mean(axis=0)], [f'{name} = {measurement:.2f} {unit}'], font_size=font_size, always_visible=True)