import collections
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


def Prepare_Test_Features(args, example, tokenizer):
    exaple["question"] = example["question"].lstrip()

    tokenized_example = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=args.max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    features = []
    for i in range(len(tokenized_example["input_ids"])):
        feature = {}
        feature["example_id"] = example["id"]
        feature["context"] = example["context"]
        feature["question"] = example["question"]
        feature["input_ids"] = tokenized_example["input_ids"][i]
        feature["attention_mask"] = tokenized_example["attention_mask"][i]
        feature["offset_mapping"] = tokenized_example["offset_mapping"][i]
        feature["sequence_ids"] = [
            0 if i is None else i for i in tokenized_example.sequence_ids(i)
        ]

        features.append(feature)

    return features


def Postprocess_qa_predictions(
    examples,
    features,
    raw_predictions,
    n_best_size=20,
    max_answer_length=30,
):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    
    for i, feature in enumerate(features):
        feature_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions")
    print(f"split into {len(features)} features.")

    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []

        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            sequence_ids = features[feature_index]["offset_mapping"]
            context_index = 1
            features[feature_index]["offset_mapping"] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1:-n_best_size - 1:-1]\
              .tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size - 1:-1]\
              .tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append({
                         "score": start_logits[start_index] + end_logits[end_index],
                         "text": context[start_char: end_char],
                    })

        if len(valid_answers) > 0:
            best_answer = sorted(
                valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = { "text": "", "score": 0.0 }
        predictions[example["id"]] = best_answer["text"]

    return predictions


test_df = pd.read_csv(
    "../../data/chaii-hindi-and-tamil-question-answering/test.csv"
)
test_df["context"] = test_df["context"].apply(lambda x: "".join(x.split()))
test_df["question"] = test_df["question"].apply(lambda x: "".join(x.split()))

tokenizer = AutoTokenizer.from_pretrained(Configration().tokenizer_name)
test_features = []
for i, row in test_df.
