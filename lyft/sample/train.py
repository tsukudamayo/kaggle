import os
import random
import time
import warnings
from pathlib import Path
from tempfile import gettempdir
from typing import Dict

import l5kit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.evaluation import (compute_metrics_csv, create_chopped_dataset,
                              read_gt_csv, write_pred_csv)
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import (PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR,
                                 draw_trajectory)
from prettytable import PrettyTable
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --------
# config
# --------
cfg = {
    'format_version': 4,
    'data_path': "..//lyft-motion-prediction-autonomous-vehicles",
    'model_params': {
        'model_architecture': 'resnet34',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "model_resnet34_output",
        'lr': 1e-3,
        'weight_path':
        "/kaggle/input/lyft-pretrained-model-hv/model_multi_update_lyft_public.pth",
        'train': False,
        'predict': True
    },
    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 4
    },
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    },
    'train_params': {
        'max_num_steps': 101,
        'checkpoint_every_n_steps': 20,
    }
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    print(l5kit.__version__)
    set_seed(42)


if __name__ == '__main__':
    main()
