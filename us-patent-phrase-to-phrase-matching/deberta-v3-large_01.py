import os
import random

import scipy as sp
import numpy as np
import pandas as pd

import torch


INPUT_DIR = "../input/us-patent-phrase-to-phrase-matching/"
OUTPUT_DIR = "./"


class CFG:
    num_workers = 4
    model = "microsoft/deberta-v3-large"
    batch_size = 32
    fc_dropout = 0.2
    target_size = 1
    max_len = 133
    seed = 42
    n_fold = 4
    trn_fold = [0, 1, 2, 3]


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]

    return score


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)
test = pd.read_csv(INPUT_DIR + "test.csv")
submission = pd.read_csv(INPUT_DIR + "sample_submission.csv")
print(f"test.shape: {test.shape}")
print(f"submission.shape: {submission.shape}")
print(test.head())
print(submission.head())
