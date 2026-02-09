import cadquery as cq
import hashlib
import io
import tempfile
import os
import glob
import random
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
import argparse
import yaml
import numpy as np

from utils.rotation.Multiprocessing import ProcessPool
from utils.rotation.ToStl import py_file_to_mesh_file

# fontconf
os.environ['FONTCONFIG_PATH'] = '/etc/fonts'
os.environ['FONTCONFIG_FILE'] = '/etc/fonts/fonts.conf'
   

def add_log(log_path: str, detail: str, msg: str):
    f = os.open(log_path, flags=(os.O_WRONLY | os.O_CREAT | os.O_APPEND))
    os.write(f, (f'{detail}: {msg}\n').encode("utf-8"))
    os.close(f)


from rotation_plane_with_reverse import Rotator, transform_source, WorkplaneTransformer

def save_rotation_script(folder_path, local_path,  script, angle_list, make_stl=False):
    z1 = sum(angle_list[0])
    if len(angle_list) > 1:
        y1 = sum(angle_list[1])
    else:
        y1 = 0
    
    if len(angle_list) > 2:
        z2 = sum(angle_list[2])
    else:
        z2 = 0

    angle_prefix = f'Z{z1}_Y{y1}_Z{z2}'
    filepath = (Path(folder_path) / local_path)
    filepath = filepath.with_stem(f"{angle_prefix}__" + filepath.stem)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(script)

    if make_stl:
        if folder_path[-1] == '/':
            folder_path = folder_path[:-1]
        os.makedirs(folder_path + '_stl', exist_ok=True)
        try:
            py_file_to_mesh_file(filepath, out_mesh_dir=folder_path + '_stl')
        except:
            # delete script if exception in stl
            os.remove(filepath)


def rotate_script(path, N = 1, zero_angle=True, make_stl=False):
    with open(path,'r')as f:
        code = f.read()

    #1) Rotation by 0, 90, 180, or 270 degrees around the Z-axis.
    #2) Rotation by 0, 90, 180, or 270 degrees around the Z-axis, followed by a 90-degree rotation around the Y-axis, then a rotation by 0, 90, 180, or 270 degrees around the Z-axis.
    #3) Rotation by 0, 90, 180, or 270 degrees around the Z-axis, followed by a 180-degree rotation around the Y-axis.

    angles = [

        #1
        [(0, 0, 0)],
        [(0, 0, 90)],
        [(0, 0, 180)],
        [(0, 0, 270)],

        #2
        [(0, 0, 0), (0, 90, 0),(0, 0, 0)],
        [(0, 0, 0), (0, 90, 0),(0, 0, 90)],
        [(0, 0, 0), (0, 90, 0),(0, 0, 180)],
        [(0, 0, 0), (0, 90, 0),(0, 0, 270)],

        [(0, 0, 90), (0, 90, 0), (0, 0, 0)],
        [(0, 0, 90), (0, 90, 0),(0, 0, 90)],
        [(0, 0, 90), (0, 90, 0),(0, 0, 180)],
        [(0, 0, 90), (0, 90, 0),(0, 0, 270)],

        [(0, 0, 180), (0, 90, 0),(0, 0, 0)],
        [(0, 0, 180), (0, 90, 0),(0, 0, 90)],
        [(0, 0, 180), (0, 90, 0),(0, 0, 180)],
        [(0, 0, 180), (0, 90, 0),(0, 0, 270)],

        [(0, 0, 270), (0, 90, 0),(0, 0, 0)],
        [(0, 0, 270), (0, 90, 0),(0, 0, 90)],
        [(0, 0, 270), (0, 90, 0),(0, 0, 180)],
        [(0, 0, 270), (0, 90, 0),(0, 0, 270)],

        #3
        [(0, 0, 0),(0, 180, 0)],
        [(0, 0, 90),(0, 180, 0)],
        [(0, 0, 180),(0, 180, 0)],
        [(0, 0, 270),(0, 180, 0)],

    ]
    if not zero_angle:
        angles.pop(0)

    chosen_angles = random.sample(angles, N)
    
    for angle_list in chosen_angles:

        rotated_code = code
        try:
            for angle in angle_list:
                if angle == (0, 0, 0):
                    continue
                rotator = Rotator(*angle)
                rotated_code = transform_source(rotated_code, rotator)

            local_path = Path(path).relative_to(Path(config['data_path']))
            save_rotation_script(config['output_path'], local_path, rotated_code, angle_list=angle_list
                                                                                ,make_stl=make_stl)

            add_log(config['logs_path'], detail=None, msg=f'Done {path}')
        except Exception as e:
            add_log(config['logs_path'], detail=e, msg=f'Error {path}')


def script_rotations(meshes_path, N = 1, zero_angle=True, make_stl=False,
                         n_processes=16, sampling=False, num_samples = 1000):
    paths = [str(p) for p in Path(meshes_path).rglob('*.py')]
    random.seed(42)
    if sampling:
        test_paths = random.sample(paths, num_samples)
    else:
        test_paths = paths
    pool = ProcessPool(n_processes = n_processes,
                        timeout = 30,
                task_func = rotate_script,
                task_args = [[path, N, zero_angle, make_stl] for path in test_paths]
                )
    pool.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(Path(config.get('logs_path')).parent, exist_ok=True)
    lock = mp.Lock()
    try:
        config['output_path']
        script_rotations(config['data_path'],
                            N=config.get('N', 1),
                            zero_angle=config.get('zero_angle', True),
                            make_stl=config.get('make_stl', False),
                            n_processes=config.get('workers', 16))
    except Exception as e:
        print('No data_path or output_path')
        raise(e)



"""
config example:

data_path: '<path to data>'
output_path: '<path to out folder for scripts>'
logs_path: '<path to logs>'

optional arguments: 

N: 2                        # number of rotations for each script
workers: 16
zero_angle: False           # enables zero angle rotation
make_stl: True              # flag to make stl for each script
"""
