# Copyright 2016-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import tqdm
import os
import json
import re
import logging
import numpy as np
import tifffile
import cv2 as cv
from skimage.measure import label
from skimage import io
from collections import OrderedDict
from metrics import Metrics
import argparse
import pandas as pd
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

models_logger = logging.getLogger(__name__)

def sub_run(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        line = str(line, encoding='utf-8')
        if line:
            print('Subprogram output: [{}]'.format(line))
    if p.returncode == 0:
        print('Subprogram success')
    else:
        print('Subprogram failed')
    return

def draw_boxplot(directory, output_path):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.xlsx')]
    
    # initialize
    data = {}
    methods = []
    eval_indexs = []
    
    # Read each Excel file and extract metrics data
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        base_name = os.path.basename(file_path).replace('.xlsx', '')
        method_name = base_name.split('_cell_segmentation')[0]
        methods.append(method_name)
        data[method_name] = df
        
    methods = sorted(methods)

    eval_indexs = [index for index in data[methods[0]].columns[1:]]  # get evaluation index
    
    eval_pd = dict([(eval_index, pd.DataFrame()) for eval_index in eval_indexs])
    for i, eval_index in enumerate(eval_indexs):
        for j, method in enumerate(methods):
            eval_pd[eval_index][method] = pd.DataFrame(data[method][eval_index])
    
    # Draw boxplot
    fig, axes = plt.subplots(1, len(eval_indexs), figsize=(5*len(eval_indexs), 6))
    
    for i, key in enumerate(eval_pd):
        sns.boxplot(data=eval_pd[key], ax=axes[i])
        axes[i].set_title(key + ' Comparison')
        axes[i].set_xlabel('Algorithm')
        axes[i].set_ylabel(key)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'benchmark-boxplot.png'))

def search_files(file_path, exts):
    file_path = file_path.replace('.ipynb_checkpoints', '')
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if '.ipynb_checkpoints' in root: continue
        if len(files) == 0:
            continue
        for f in files:
            if '.ipynb_checkpoints' in f: continue
            fn, ext = os.path.splitext(f)
            if ext in exts: files_.append(os.path.join(root, f))
    
    return files_

class CellSegEval(object):
    def __init__(self, method: str = None):
        self._method = method
        self._gt_list = list()
        self._dt_list = list()
        self._object_metrics = None
        self._suitable_shape = None

    def set_method(self, method: str):
        self._method = method

    def _load_image(self, image_path: str):
        arr_ = np.zeros(self._suitable_shape, dtype=np.uint8)
        arr = tifffile.imread(image_path, key=0)  # Read first frame
        h, w = arr.shape
        arr_[:h, :w] = arr
        arr_ = label(arr_, connectivity=2)
        return arr_

    def evaluation(self, gt_path: str, dt_path: str):
        dt_path = dt_path.replace('.ipynb_checkpoints', '')
        gt_path = gt_path.replace('.ipynb_checkpoints', '')
        for i in [gt_path, dt_path]:
            assert os.path.exists(i), '{} is not exists'.format(i)
        
        print(f'gt:{gt_path}\ndt:{dt_path}')
        if os.path.isfile(gt_path):
            self._gt_list = [gt_path]
        else:
            img_lst = search_files(gt_path, ['.tif','.png','.jpg'])
            self._gt_list = [i for i in img_lst if 'mask' in i.lower()]
        
        if os.path.isfile(dt_path):
            self._dt_list = [dt_path]
        else:
            self._dt_list = search_files(dt_path, ['.tif','.png','.jpg'])
        
        # New matching logic
        matched_gt = []
        for dt_file in self._dt_list:
            base = os.path.basename(dt_file)
            # Case 1: New format (abc-img_v3_mask.tif -> abc-mask.tif)
            if '-img_' in base and ('_mask.' in base or '_masks.' in base):
                prefix = base.split('-img_')[0]
                gt_file = os.path.join(gt_path, f"{prefix}-mask{os.path.splitext(base)[1]}")
            # Case 2: Old format (imgX.tif -> maskX.tif)
            else:
                gt_file = dt_file.replace('img', 'mask').replace(dt_path, gt_path)
            
            if os.path.exists(gt_file):
                matched_gt.append(gt_file)
            else:
                print(f"Warning: GT file not found for {dt_file}")
        
        self._gt_list = matched_gt
        assert len(self._gt_list) > 0, 'No matching GT files found'
        
        gt_arr = list()
        dt_arr = list()
        shape_list = list()
        for i in self._dt_list:
            dt = tifffile.imread(i, key=0)
            shape_list.append(dt.shape)
        
        w = np.max(np.array(shape_list)[:, 1])
        h = np.max(np.array(shape_list)[:, 0])
        self._suitable_shape = (h, w)
        models_logger.info('Uniform size {} into {}'.format(list(set(shape_list)), self._suitable_shape))
        
        for i in tqdm.tqdm(self._dt_list, desc='Load data {}'.format(self._method)):
            gt = self._load_image(image_path=self._find_gt_path(i, dt_path, gt_path))
            dt = self._load_image(image_path=i)
            assert gt.shape == dt.shape, 'Shape of GT are not equal to DT'
            gt_arr.append(gt)
            dt_arr.append(dt)
        
        gt_arr = np.array(gt_arr)
        dt_arr = np.array(dt_arr)
        pm = Metrics(self._method, cutoff1=0.55)
        models_logger.info('Start evaluating the test set, which will take some time.')
        object_metrics = pm.calc_object_stats(gt_arr, dt_arr)
        self._object_metrics = object_metrics.drop(
            labels=['gained_detections', 'missed_det_from_merge', 'gained_det_from_split', 
                   'true_det_in_catastrophe', 'pred_det_in_catastrophe', 'merge', 
                   'split', 'catastrophe', 'seg', 'n_pred', 'n_true', 'correct_detections',
                   'missed_detections'], axis=1)
        self._object_metrics.index = [os.path.basename(d) for d in self._dt_list]
        return self._object_metrics.mean().to_dict()

    def _find_gt_path(self, dt_file, dt_path, gt_path):
        base = os.path.basename(dt_file)
        # Case 1: New format (abc-img_v3_mask.tif -> abc-mask.tif)
        if '-img_' in base and ('_mask.' in base or '_masks.' in base):
            prefix = base.split('-img_')[0]
            return os.path.join(gt_path, f"{prefix}-mask{os.path.splitext(base)[1]}")
        # Case 2: Old format (imgX.tif -> maskX.tif)
        return dt_file.replace('img', 'mask').replace(dt_path, gt_path)

    def dump_info(self, save_path: str):
        import time
        t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        save_path_ = os.path.join(save_path, '{}_cell_segmentation_{}.xlsx'.format(self._method, t))
        self._object_metrics.to_excel(save_path_)
        models_logger.info('The evaluation results is stored under {}'.format(save_path_))

def get_auto_colors(methods):
    """Generate automatic color palette for methods"""
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    return {method: color_palette[i % len(color_palette)] 
            for i, method in enumerate(sorted(methods))}

def main(args, para):
    dataset_name = os.path.basename(os.path.dirname(args.gt_path))
    visible_folders = [folder for folder in os.listdir(args.dt_path) if not folder.startswith('.')]
    methods = sorted(visible_folders)  # Automatically sort methods alphabetically
    colors = get_auto_colors(methods)
    
    print(f'dataset_name:{dataset_name}')
    print(f'methods:{methods}')
    
    dct = {}
    gt_path = os.path.join(args.gt_path)
    dataset_dct = {}
    
    for m in methods:
        dt_path = os.path.join(args.dt_path, m)
        cse = CellSegEval(m)
        v = cse.evaluation(gt_path=gt_path, dt_path=dt_path)
        dataset_dct[m] = v
        if os.path.exists(args.output_path):
            cse.dump_info(args.output_path)
        else:
            models_logger.warn('Output path not exists, will not dump result')

    dct[dataset_name] = dataset_dct

    index = ('Precision', 'Recall', "F1", 'jaccrd', 'dice')
    
    fig, axs = plt.subplots(figsize=(16, 12))
    x = np.arange(len(index))
    width = 0.8 / len(methods)  # Auto-adjust width based on method count
    
    for i, (method, measurement) in enumerate(dataset_dct.items()):
        offset = width * i
        rects = axs.bar(x + offset, [round(val,2) for val in measurement.values()], 
                        width, label=method, color=colors[method], alpha=0.7)
        axs.bar_label(rects, padding=3, fontsize=8)
    
    axs.set_ylabel('Evaluation Index')
    axs.set_title('dataset - {}'.format(dataset_name))
    axs.set_xticks(x + width/2, index)
    axs.legend(loc='upper left', ncols=3)
    axs.set_ylim(0, 1)

    plt.savefig(os.path.join(args.output_path, '{}_benchmark.png'.format(dataset_name)))
    
    # box plot
    try:
        draw_boxplot(args.output_path, args.output_path)
    except Exception as e:
        print(f"Boxplot generation failed: {str(e)}")

usage = """ Evaluate cell segmentation """
PROG_VERSION = 'v0.0.1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-g", "--gt_path", action="store", dest="gt_path", type=str, required=True,
                        help="Input GT path.")
    parser.add_argument("-d", "--dt_path", action="store", dest="dt_path", type=str, required=True,
                        help="Input DT path.")
    parser.add_argument("-o", "--output_path", action="store", dest="output_path", type=str, required=True,
                        help="Output result path.")
    parser.set_defaults(func=main)

    (para, args) = parser.parse_known_args()
    para.func(para, args)