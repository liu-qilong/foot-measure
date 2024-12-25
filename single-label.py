FILE_PATH = 'data/stl/020_l.stl'
OUTPUT_PATH = 'data/stl/020_l-extra.pkl'
POINT_NAMES = ['MLA-AP', 'MLA-PP', 'MMB', 'LMB', 'AA']

from measure import label

df = label.label(
    file = FILE_PATH,
    point_names = POINT_NAMES,
    use_texture = False,
    export_name='020_l-extra',
)