import os
from typing import TypeVar
from dataclasses import dataclass


import spacy
import warnings
import wordcloud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


RANDOM_IDX = 43
warnings.filterwarnings("ignore")


@dataclass
class Dataset():
    train: pd.DataFrame
    test: pd.DataFrame
    features: pd.DataFrame
    patient_notes: pd.DataFrame

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: pd.DataFrame,
        patient_notes: pd.DataFrame,
    ):
        self.train = train
        self.test = test
        self.features = features
        self.patient_notes = patient_notes



dataset = Dataset(
    train=pd.read_csv("../../input/nbme-score-clinical-patient-notes/train.csv"),
    test=pd.read_csv("../../input/nbme-score-clinical-patient-notes/test.csv"),
    features=pd.read_csv("../../input/nbme-score-clinical-patient-notes/features.csv"),
    patient_notes=pd.read_csv("../../input/nbme-score-clinical-patient-notes/patient_notes.csv")
)
print(dataset.train.head())
print(dataset.test.head())
print(dataset.features.head())
print(dataset.patient_notes.head())

