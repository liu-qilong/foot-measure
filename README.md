# `foot-measure`

This is a simple tool for:

- Labelling foot landmarks from 3D foot scans.
- Calculate foot measurements from the labelled landmarks and the foot scans.

## Setup

First install [mesh4d](https://github.com/liu-qilong/mesh4d) package and create a virtual environment:

```
git clone https://github.com/liu-qilong/mesh4d.git
cd mesh4d
conda create -n mesh4d
conda activate mesh4d
pip install -r requirements.txt
python -m pip install --editable .
```

Clone this repository:

```
cd ..
git clone https://github.com/liu-qilong/foot-measure
cd foot-measure
conda activate mesh4d
```

## Usage

Under the project root directory, `batch-label-and-measure.py` is a script for batch labelling and measuring of foot scans in a folder. To use it:

### Change meta data

- At the beginning of the script, change the `mesh_folder` and the `mesh_type` accordingly.

```
mesh_folder = 'data'
mesh_type = '.stl'
```

### Landmarks labelling

- Run the script with the following command:

```
conda activate mesh4d
python batch-label-and-measure.py
```

- You will see a window showing the foot scan. Drag to rotate perspective, scroll to zoom in/out, and press `shift` and drag to pan. Click to add a key point.
- Pressing `Q` to control the proceeding or retreating of the labelling process:
    - When the number of currently added landmarks < 12, press `Q` will delete the lastly added landmark.
    - When the number of currently added landmarks = 12, press `Q` will end labelling and proceed to the next stage.
    - When the number of currently added landmarks > 12, press `Q` will delete all the landmarks and restart labelling.

![img](https://github.com/liu-qilong/foot-measure/blob/main/gallery/pick-points.png?raw=true)

### LCS verification and measurements

- After labelling, you will see a window showing the foot scan and the $x$, $y$, $z$ axises of the estimated LCS (local coordinates system). If it's correct, press `Q` to proceed; otherwise, don't trust the measurement results.
- The measurement results will be outputted as a `.csv` file to the same folder of the foot scan and with the same filename as the foot scan.
- The programme will automatically proceed to the next foot scan in the folder and repeat the above steps.
- When all foot scans are labelled and measured, a `measurements.csv` will be generated in the folder with all measurement results gathered together.

![img](https://github.com/liu-qilong/foot-measure/blob/main/gallery/measurements.png?raw=true)