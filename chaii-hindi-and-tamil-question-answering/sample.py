import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc

gc.enable()
import json
import math
import multiprocessing
import random
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from string import punctuation

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from sklearn import model_selection
from torch.nn import Parameter
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (MODEL_FOR_QUESTION_ANSWERING_MAPPING, WEIGHTS_NAME,
                          AdamW, AutoConfig, AutoModel, AutoTokenizer,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup, logging)

logging.set_verbosity_warning()
logging.set_verbosity_error()
try:
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else num_cpus - 1

    return optimal_value


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class Configration:
    model_type = "xlm_roberta"
    model_name_or_path = "../../model/xlm-roberta-large-squad-v2"
    config_name = "../../model/xlm-roberta-large-squad-v2"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "01"
    gradient_accumulation_steps = 2

    # tokenizer
    tokenizer_name = "../../model/xlm-roberta-large-squad-v2"
    max_seq_length = 400
    doc_stride = 135

    # train
    epochs = 1
    train_batch_size = 4
    eval_batch_size = 128

    # optimizer
    optimizer_type = "AdamW"
    learning_rate = 1e-5
    weight_decay = 1e-2
    epsilon = 1e-8
    max_grad_norm = 1.0

    # scheduler
    decay_name = "linear-warmup"
    warmup_ratio = 0.1

    # logging
    logging_steps = 10

    # evaluate
    output_dir = "output"
    seed = 2021


# Dataset_Retriever class
class DatasetRetriver(Dataset):
    def __init__(self, features, mode="train"):
        super(Dataset_Retriver, self).__init__()
        self.features = features
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        if self.mode == "train":
            return {
                "input_ids":
                torch.tensor(feature["input_ids"], dtype=torch.long),
                "attention_mask":
                torch.tensor(feature["attention_mask"], dtype=torch.long),
                "offset_mapping":
                torch.tensor(feature["offset_mapping"], dtype=torch.long),
                "start_position":
                torch.tensor(feature["start_position"], dtype=torch.long),
                "end_position":
                torch.tensor(feature["end_position"], dtype=torch.long),
            }
        else:
            return {
                "input_ids":
                torch.tensor(feature["input_ids"], dtype=torch.long),
                "attention_mask":
                torch.tensor(feature["attention_mask"], dtype=torch.long),
                "offset_mapping":
                torch.tensor(feature["offset_mapping"], dtype=torch.long),
                "id":
                feature["example_id"],
                "context":
                feature["context"],
                "question":
                feature["question"],
            }


class Model(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(Model, self).__init__()
        self.config = config
        self.xlm_roberta = AutoModel.from_pretrained(modelname_or_path,
                                                     config=config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        outputs = self.xlm_roberta(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        qa_logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

    def Make_Model(args):
        config = AutoConfig.from_pretrained(args.config_name)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        model = Model(args.model_name_or_path, config=config)

        return config, tokenizer, model
















    
