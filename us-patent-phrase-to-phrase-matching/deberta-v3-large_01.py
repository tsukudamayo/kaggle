import os
import random
from typing import Optional, Tuple
import gc

import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel


INPUT_DIR = "../input/us-patent-phrase-to-phrase-matching/"
OUTPUT_DIR = "./"


class CFG:
    num_workers = 4
    path = "../input/us-patent-phrase-to-phrase-matching/pppm-deberta-v3-large-closing-the-cv-lb-gap/"
    config_path = path + "config.pth"    
    model = "microsoft/deberta-v3-large"
    batch_size = 32
    fc_dropout = 0.2
    target_size = 1
    max_len = 133
    seed = 42
    n_fold = 4
    trn_fold = [0, 1, 2, 3]


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


def prepare_input(cfg: CFG, text):
    inputs = cfg.tokenizer(
        text,
        add_special_tokens=True,
        max_length=cfg.max_len,
        padding="max_length",
        return_offset_mapping=False,
    )

    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)

    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg: CFG, df: pd.DataFrame):
        self.cfg = cfg
        self.texts = df["text"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs


class CustomModel(nn.Module):
    def __init__(
        self,
        cfg: CFG,
        config_path: Optional[str] = None,
        pretrained: Optional[bool] = False,
    ):
        super().__init__()
        self.cfg = cfg

        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                cfg.model,
                output_hidden_states=True,
            )
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(
                cfg.model,
                config=self.config,
            )
        else:
            self.model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1),
        )
        self._init_weights(self.attention)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs: Tuple[torch.FloatTensor]) -> torch.Tensor:
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states)

        return feature

    def forward(self, inputs: Tuple[torch.FloatTensor]) -> nn.Dropout:
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))

        return output


def inference_fn(
    test_loader: DataLoader,
    model: CustomModel,
    device: torch.device,
) -> np.ndarray:
    preds = []
    model.eval()
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        inputs[k] = v.to(device)
    with torch.no_grad():
        y_preds: CustomModel = model(inputs)
    preds.append(y_preds.sigmoid().to("cpu").numpy())
    predictions = np.concatenate(preds)

    return predictions


def get_cpc_texts():
    contexts = []
    pattern = "[A-Z]\d+"
    for file_name in os.listdir("../input/cp



seed_everything(seed=42)
# train
train = pd.read_csv(INPUT_DIR + "train.csv")
# test
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
test = pd.read_csv(INPUT_DIR + "test.csv")
submission = pd.read_csv(INPUT_DIR + "sample_submission.csv")
print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")
print(f"submission.shape: {submission.shape}")
print(train.head())
print(test.head())
print(submission.head())

CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path + "tokenizer/")

cpc_texts = torch.load(CFG.path + "cpc_texts.pth")
test["contenxt_text"] = test["context"].map(cpc_texts)
print(test.head())

test["text"] = test["anchor"] + "[SEP]" + test["target"] + "[SEP]" + test["context_text"]
print(test.head())

test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
    drop_last=False,
)
predictions = []
for fold in CFG.trn_fold:
    model = CustomModel(
        CFG,
        config_path=CFG.config_path,
        pretrained=False
    )
    state = torch.load(
        CFG.path + f"{CFG.model.replace("/", "-")}_fold{fold}_best.pth",
        map_location=torch.device("cpu")
    )
    model.load_state_dict(state["model"])
    prediction = inference_fn(test_loader, model, device)
    predictions.append(prediction)

    del model, state, prediction
    gc.collect()

    torch.cuda.empty_cache()
predictions = np.mean(predictions, axis=0)

sumission["score"] = predictions
print(submission.head())
sumission[["id", "score"]].to_csv("submission.csv", index=False)

