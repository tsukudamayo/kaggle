import random
import os
import ast
import itertools

import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
from sklearn.metrics import f1_score
import torch
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def micro_f1 (preds, truths):
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)

    return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1

    return binary


def span_micro_f1(preds, truths):
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(
            np.max(pred) if len(pred) else 0,
            np.max(truth) if len(truth) else 0,
        )
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))

    return micro_f1(bin_preds, bin_truths)


def create_labels_for_scoring(df: pd.DataFrame):
    # example: ['0 1', '3 4'] -> ['0 1; 3 4']
    df['location_for_create_labels'] = [ast.literal_eval(f'[]')] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, 'location']
        if lst:
            new_lst = ';'.join(lst)
            df.loc[i, 'location_for_create_labels'] = ast.literal_eval(f'[["{new_lst}"]]')
    # create labels
    truths = []
    for location_list in df['location_for_create_labels'].values:
        truth = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)

    return truths


def get_char_probs(texts, predictions, tokenizer: AutoTokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        for idx, (offset_mapping, pred) in enumerate(zip(encoded["offset_mapping"], prediction)):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[i][start:end] = pred

    return results


def get_results(char_probs, th=0.5):
    results = []
    for char_prob in char_probs:
        result = np.where(char_prob >= th)[0] + 1
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        result = [f"{min(r)} {max(r)}" for r in result]
        result = ";".join(result)
        results.append(result)

    return results


def get_predictions(results):
    predictions = []
    for result in results:
        prediction = []
        if result != "":
            for loc in [s.split() for s in result.split(";")]:
                start, end = int(loc[0]), int(loc[1])
                prediction.append([start, end])
        predictions.append(prediction)

    return predictions


def get_score(y_true, y_pred):
    return span_micro_f1(y_true, y_pred)


class DataLoader:

    def __init__(self, data_dir):
        self.features = self.preprocess_features(pd.read_csv(data_dir + "features.csv"))
        self.patient_notes = pd.read_csv(data_dir + "patient_notes.csv")
        self.test = pd.read_csv(data_dir + "test.csv")\
          .merge(self.features, on=["feature_num", "case_num"], how="left")\
          .merge(self.patient_notes, on=["pn_num", "case_num"], how="left")
        self.submission = pd.read_csv(data_dir + "sample_submission.csv")

    def preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        features.loc[27, "feature_text"] = "Last-Pap-smear-1-year-ago"

        return features

w1 = 1
w2 = 0
w3 = 0

seed_everything(seed=42)

data_dir = "../../input/nbme-score-clinical-patient-notes/"
dataset = DataLoader(data_dir=data_dir)
test = dataset.test
features = dataset.features
patient_notes = dataset.patient_notes

print(f"test.shape: {test.shape}")
print(f"features.shape: {features.shape}")
print(f"patient_notes.shape: {patient_notes.shape}")
print(features.head())
print(patient_notes.head())
print(test.head())
