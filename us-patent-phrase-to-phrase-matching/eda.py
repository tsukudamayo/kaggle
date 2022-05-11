import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import seaborn as sns

import pyLDAvis
import pyLDAvis.gensim

import gensim
from gensim import corpora

import en_core_web_sm
import re
import spacy

from wordcloud import WordCloud


df = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")
print(f"train data shape: {df.shape}, test data shape: {test.shape}")
