import os
import gc
import re
import ast
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter


INPUT_DIR = "../input/us-patent-phrase-to-phrase-matching/"
OUTPUT_DIR = "../output/us-patent-phrase-to-phrase-matching/"


class CFG:
    model="microsoft/deberta-v3-large"
    scheduler="cosine"
    batch_scheduler=True
    n_fold=4
    seed=42


def get_score(y_true, y_pred):
    return sp.stats.pearsonr(y_true, y_pred)[0]


def get_logger(filename=OUTPUT_DIR + "train"):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%s(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_cpc_texts():
    contexts = []
    pattern = "[A-Z]\d+"
    for file_name in os.listdir("../input/CPCSchemeXML202105/"):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
        with open(f"../input/CPCTitleList202205/cpc-section-{cpc}_20220501.txt") as f:
            s = f.read()
        pattern = f"{cpc}\t\t.+"
        result = re.findall(pattern, s)
        cpc_result = result[0].lstrip(pattern)
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f"{context}\t\t.+"
            result = re.findall(pattern, s)
            results[context] = cpc_result + ". " + result[0].lstrip(pattern)

    return results


def prepare_input(cfg, text):
    inputs = cfg.tokenizer(
        text,
        add_special_tokens=True,
        max_length=cfg.max_len,
        padding="max_length",
        return_offsets_mapping=False,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df["text"].values
        self.labels = df["score"].values

    def __len__(self, item):
        return len(self.labels)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)

        return inputs, label






LOGGER = get_logger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
seed_everything(seed=42)

# Data Loading
train = pd.read_csv(INPUT_DIR + "train.csv")
test = pd.read_csv(INPUT_DIR + "test.csv")

print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")
print(f"submission.shape {test.shape}")
print(train.head())
print(test.head())

# cpc text Loading
cpc_texts = get_cpc_texts()
torch.save(cpc_texts, OUTPUT_DIR + "cpc_texts.pth")
train["context_text"] = train["context"].map(cpc_texts)
test["context_text"] = test["context"].map(cpc_texts)
print(train.head())
print(test.head())

train["text"] = train["anchor"] + "[SEP]" + train["target"] + "[SEP]" + train["context_text"]
test["text"] = test["anchor"] + "[SEP]" + test["target"] + "[SEP]" + test["context_text"]
print(train.head())
print(test.head())

train["score"].hist()
train["context"].apply(lambda x: x[0]).value_counts()

# CV split
train["score_map"] = train["score"].map({
    0.00: 0,
    0.25: 1,
    0.5: 2,
    0.75: 3,
    1.00: 4,
})
Fold = StratifiedKFold(
    n_splits=CFG.n_fold,
    shuffle=True,
    random_state=CFG.seed
)
for n, (train_index, val_index) in enumerate(Fold.split(train, train["score_map"])):
    train.loc[val_index, "fold"] = int(n)
train["fold"] = train["fold"].astype(int)
print(train.groupby("fold").size())

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained(OUTPUT_DIR + "tokenizer/")
CFG.tokenizer = tokenizer

# Dataset
lengths_dict = {}
lengths = []
tk0 = tqdm(cpc_texts.values(), total=len(cpc_texts))

for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    lengths.append(length)

lengths_dict["context_text"] = lengths

for text_col in ["anchor", "target"]:
    lengths = []
    tk0 = tqdm(train[text_col].fillna("").values, total=len(train))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(length)
    lengths_dict[text_col] = lengths
CFG.max_len = max(lengths_dict["anchor"]) + max(lengths_dict["target"]) + max(lengths_dict["context_text"]) + 4
LOGGER.info(f"max_len: {CFG.max_len}")

train_dataset = TrainDataset(CFG, train)
inputs, label = train_dataset[0]
print(inputs)
print(label)
