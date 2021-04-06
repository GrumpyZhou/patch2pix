import os
import time
import argparse 
import numpy as np
from utils.colmap.data_loading import sav_model_multi_ov_pairs

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/immatch_benchmark/val_as_train')
args = parser.parse_args()
data_root = args.data_root

overlaps = [0.1, 0.2, 0.3, 0.4, 0.5]
scenes = os.listdir(data_root)
print(f'Target scenes: {scenes}, ovs: {overlaps}\n')
for scene in scenes:    
    print(f'Start processing scene: {scene}')
    model_dir = os.path.join(data_root, scene, 'dense/sparse')
    t0 = time.time()
    ov_pair_dict = sav_model_multi_ov_pairs(model_dir, overlaps)
    print(f'Finished, time {time.time() - t0}')