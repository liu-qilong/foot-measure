import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from mesh4d import obj3d, utils, kps
from mesh4d.analyse import crave, measure

mesh_folder = '/Users/knpob/Territory/Kolmo/data/DynaFootLite/Fast'
mesh_fps = 40
start = 1
end = 1
stride = 1

output_folder = 'output'
output_name = 'test'

##############################
# landmarks labelling
##############################
landmarks, files = crave.landmarks_labelling(
    mesh_folder = mesh_folder,
    mesh_fps = mesh_fps,
    start = start,
    end = end,
    stride = stride,
    point_names = [f'P{idx + 1}' for idx in range(12)],
    use_texture = False,
    export_folder = output_folder,
    export_name = output_name,
)

utils.save_pkl_object(files, output_folder, f'{output_name}_files')

##############################
# foot pca
##############################
def pca_axes(vertices):
    pca = PCA()
    pca.fit(vertices)

    stds = np.sqrt(pca.explained_variance_)
    axes = pca.components_  # (n_axes, n_coords)
    mean = pca.mean_ # (n_coords)
    return mean, axes, stds

mesh_ls, _ = obj3d.load_mesh_series(
    folder=mesh_folder,
    start=start,
    stride=stride,
    end=end,
    load_texture=False,
)

contour = landmarks.extract(('P7', 'P10', 'P11', 'P12'))
mesh_clip_ls = crave.clip_with_contour(mesh_ls, start_time=0, contour=contour, clip_bound='', margin=0, invert=True)

foot_ls = obj3d.init_obj_series(
    mesh_clip_ls,
    obj_type=obj3d.Obj3d_Deform
    )

axes_ls = []

for idx in range(len(foot_ls)):
    vertices = landmarks.to_array()[0][idx][:10]
    mean, axes, stds = pca_axes(vertices)
    axes_ls.append({'mean': mean, 'axes': axes, 'stds': stds})

utils.save_pkl_object(axes_ls, output_folder, f'{output_name}_axes')

##############################
# convert landmarks to local frame
##############################
def coord_cartesian2homo(vertices):
    shape = list(vertices.shape)
    shape[1] += 1

    vertices_homo = np.ones(shape)
    vertices_homo[:, :3] = vertices
    return vertices_homo

def coord_homo2cartesian(vertices_homo):
    return vertices_homo[:, :3] / vertices_homo[:, [-1]]

def transformation_matrix2global(axes, origin):
    """e(i) -> a_i + t"""
    matrix = np.eye(4)
    matrix[:3, :3] = axes.T
    matrix[:3, 3] = origin
    
    return matrix

def transformation_matrix2local(axes, origin):
    """a_i + t -> e(i)"""
    matrix2golbal = transformation_matrix2global(axes, origin)
    return np.linalg.inv(matrix2golbal)

def transform(matrix, vertices):
    vertices_homo = coord_cartesian2homo(vertices)
    vertices_transform = (matrix @ vertices_homo.T).T
    return coord_homo2cartesian(vertices_transform)

arr, names = landmarks.to_array()
arr_transform = []

for idx in range(len(mesh_clip_ls)):
    matrix = transformation_matrix2local(axes_ls[idx]['axes'], axes_ls[idx]['mean'])
    vertices_transform = transform(matrix, arr[idx])
    arr_transform.append(vertices_transform)

arr_transform = np.array(arr_transform)
landmarks_transform = kps.MarkerSet()
landmarks_transform.load_from_array(arr_transform, start_time=0, fps=landmarks.fps, index=names)
utils.save_pkl_object(landmarks_transform, output_folder, f'{output_name}_transform')

##############################
# measurement
##############################
def coord_cartesian2homo(vertices):
    shape = list(vertices.shape)
    shape[1] += 1

    vertices_homo = np.ones(shape)
    vertices_homo[:, :3] = vertices
    return vertices_homo

def coord_homo2cartesian(vertices_homo):
    return vertices_homo[:, :3] / vertices_homo[:, [-1]]

def transformation_matrix2global(axes, origin):
    """e(i) -> a_i + t"""
    matrix = np.eye(4)
    matrix[:3, :3] = axes.T
    matrix[:3, 3] = origin
    
    return matrix

def transformation_matrix2local(axes, origin):
    """a_i + t -> e(i)"""
    matrix2golbal = transformation_matrix2global(axes, origin)
    return np.linalg.inv(matrix2golbal)

def transform(matrix, vertices):
    vertices_homo = coord_cartesian2homo(vertices)
    vertices_transform = (matrix @ vertices_homo.T).T
    return coord_homo2cartesian(vertices_transform)

# preprocessing
mesh_transform_ls = []

for idx in range(len(mesh_ls)):
    matrix = transformation_matrix2local(axes_ls[idx]['axes'], axes_ls[idx]['mean'])
    mesh_transform_ls.append(mesh_ls[idx].transform(matrix, inplace=False))

mesh_fix_ls = []

for mesh in mesh_transform_ls:
    clean = mesh.clean()
    bodies = clean.split_bodies()

    point_nums = [len(body.points) for body in bodies]
    max_index = point_nums.index(max(point_nums))
    mesh_fix_ls.append(bodies[max_index].extract_surface())

# add ground points to landmarks
ground_ls = []

for frame in range(len(mesh_fix_ls)):
    min_idx = mesh_fix_ls[frame].points[:, -1].argmin()
    ground = mesh_fix_ls[frame].points[min_idx]
    ground_ls.append(ground)

ground_array = np.array(ground_ls)

array, names = landmarks_transform.to_array()
landmarks_transform.load_from_array(
    array=np.concatenate([array, np.expand_dims(ground_array, axis=-2)], axis=-2),
    index=names + ['Ground'],
    )

# euclidean measurements
def get(landmarks, marker_name, axis):
    axis2idx = {
        'x': 0,
        'y': 1,
        'z': 2
    }
    return landmarks.markers[marker_name].coord[axis2idx[axis]]

def dist_along_axis(landmarks, marker1, marker2, axis, is_abs=True):
    dist = landmarks.get(marker1, axis) - landmarks.get(marker2, axis)

    if is_abs:
        return np.abs(dist)
    else:
        return dist

def dist(landmarks, marker1, marker2):
    return np.linalg.norm(landmarks.markers[marker1].coord - landmarks.markers[marker2].coord, axis=0)

def angle(landmarks, marker_origin, marker1, marker2):
    v1 = (landmarks.markers[marker1].coord - landmarks.markers[marker_origin].coord).T
    v2 = (landmarks.markers[marker2].coord - landmarks.markers[marker_origin].coord).T
    cos = np.abs(np.sum(v1 * v2, axis=-1)) / np.linalg.norm(v1, axis=-1) / np.linalg.norm(v2, axis=-1)
    return np.arccos(cos) / np.pi * 180

type(landmarks_transform).get = get
type(landmarks_transform).dist_along_axis = dist_along_axis
type(landmarks_transform).dist = dist
type(landmarks_transform).angle = angle

metric = {}

metric['FL'] = ['foot length (mm)']\
    + list(landmarks_transform.dist_along_axis('P1', 'P10', 'x'))
metric['MBL'] = ['medial ball length (mm)']\
    + list(landmarks_transform.dist_along_axis('P4', 'P10', 'x'))
metric['LBL'] = ['lateral ball length (mm)']\
    + list(landmarks_transform.dist_along_axis('P5', 'P10', 'x'))
metric['ABW'] = ['anatomical ball width (mm)']\
    + list(landmarks_transform.dist('P4', 'P5'))
metric['OBW'] = ['orthogonal ball width (mm)']\
    + list(landmarks_transform.dist_along_axis('P4', 'P3', 'y'))
metric['OHW'] = ['orthogonal heel width (mm)']\
    + list(landmarks_transform.dist_along_axis('P9', 'P8', 'y'))
metric['BH'] = ['ball heigh (mm)']\
    + list(landmarks_transform.dist_along_axis('P6', 'Ground', 'z'))
metric['IH'] = ['instep height (mm)']\
    + list(landmarks_transform.dist_along_axis('P7', 'Ground', 'z'))
metric['BA'] = ['ball angle (°)']\
    + list(landmarks_transform.angle('P4', 'P5', 'P8'))
metric['T1A'] = ['toe 1 angle (°)']\
    + list(180 - landmarks_transform.angle('P4', 'P2', 'P8'))
metric['T5A'] = ['toe 5 angle (°)']\
    + list(180 -landmarks_transform.angle('P5', 'P3', 'P9'))

df = pd.DataFrame(metric, index=['description'] + [files[idx] for idx in range(len(landmarks_transform.to_array()[0]))])

# manifold measurements
cir_ls = []

for frame_id in range(len(mesh_fix_ls)):
    p4 = landmarks_transform.markers['P4'].coord[:, frame_id]
    p5 = landmarks_transform.markers['P5'].coord[:, frame_id]
    id4 = mesh_fix_ls[frame_id].find_closest_point(p4)
    id5 = mesh_fix_ls[frame_id].find_closest_point(p5)

    # extract geodesic path
    path = mesh_fix_ls[frame_id].geodesic(id4, id5)

    # estimate circumference plane
    norm, center = measure.estimate_plane_from_points(path.points)
    mesh_clip = mesh_fix_ls[frame_id].clip(norm, origin=center, invert=True)

    # calculate circumference
    boundary = mesh_clip.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
    cir_ls.append(boundary.length)

df['ABG'] = ['anatomical ball girth'] + cir_ls
df.to_csv(f'{output_folder}/{output_name}_measurement.csv')