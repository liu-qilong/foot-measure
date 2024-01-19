from mesh4d.analyse import crave

crave.landmarks_labelling(
    mesh_folder = '/Users/knpob/Territory/Kolmo/data/DynaFootLite/Fast',
    mesh_fps = 40,
    start = 1,
    end = 5,
    stride = 1,
    point_names = [f'P{idx + 1}' for idx in range(3)],
    use_texture = False,
    export_folder = 'prototype/output',
    export_name = 'test'
)