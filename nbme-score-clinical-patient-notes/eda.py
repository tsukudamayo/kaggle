import os
import spacy
import warnings
import wordcloud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


RANDOM_IDX = 12
warnings.filterwarnings("ignore")


def load_dataset() -> None:
    train = pd.read_csv("../data/nbme-score-clinical-patient-notes/train.csv")
    test = pd.read_csv("../data/nbme-score-clinical-patient-notes/test.csv")
    features = pd.read_csv("../data/nbme-score-clinical-patient-notes/features.csv")
    patient_notes = pd.read_csv("../data/nbme-score-clinical-patient-notes/patient_notes.csv")
    submission = pd.read_csv("../data/nbme-score-clinical-patient-notes/sample_submission.csv")

    print(train.head())
    print(test.head())
    print(features.head())
    print(patient_notes.haed())


def main():
    load_dataset()
    
    


if __name__ == "__main__":
    main()
