import re
import spacy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import seaborn as sns

import pyLDAvis
import pyLDAvis.gensim_models

import gensim
from gensim import corpora

# import en_core_web_sm

from wordcloud import WordCloud

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")
print(f"train data shape: {df.shape}, test data shape: {test.shape}")

print(df.head())
print(len(np.unique(df.id)), df.shape[0])

# plot unique data in dataframe
# notably, although anchor and target are heavily related by meaning,
# the unique values vary greatly.
# However, ~7000 target values seem to be identical,
# given that there are 36473 unique entries in the df.
vals = [
    len(np.unique(df.anchor)),
    len(np.unique(df.target)),
    len(np.unique(df.context)),
]
sns.barplot(
    x=["anchor", "target", "context"],
    y=vals,
)
plt.show()

# anchor
# the number of symbols in the anchor are normally distributed
print(df.anchor.value_counts())
print(df.anchor.value_counts().reset_index().describe())

sns.set(font_scale=0.5)
fig, ax = plt.subplots(figsize=(65, 30))
sns.countplot(
    x=df.anchor,
    order=df.anchor.value_counts().index,
    ax=ax,
    color="b",
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.axhline(
    df.anchor.value_counts().reset_index().describe().loc["25%"][0],
    color="r",
    label="25% percentile",
)
ax.axhline(
    df.anchor.value_counts().reset_index().describe().loc["50%"][0],
    color="orange",
    label="50% percentile",
)
ax.axhline(
    df.anchor.value_counts().reset_index().describe().loc["75%"][0],
    color="r",
    label="75% percentile",
)
plt.title("Counts of Anchors", fontsize=40)
plt.legend(fontsize=40)
plt.show()

# the number of symbols in the anchor are normally distributed
sns.set(font_scale=0.6)
symbols = []
for i in df.anchor:
    symbols.append(len(i))

sns.countplot(x=symbols, color="b")
plt.title("Number of letters in Anchors")

# the anchors contain 1-5 words; most of them contain 2 or 3 words
word_count = []
for i in df.anchor:
    word_count.append(len(i.split()))
sns.countplot(x=word_count, color="b")
plt.title("Number of words in Anchors")

# target
print(df.target.value_counts())
print(df.target.value_counts().reset_index().describe())

# Checking numbers in anchor feature
# Code from: https://www.kaggle.com/code/remekkinas/eda-and-feature-engineering/notebook
# 5 anchors contain numbers
# generally these names are rather cryptic
pattern = "[0-9]"
mask = df["anchor"].str.contains(pattern, na=False)
df["nun_anchor"] = mask
df[mask]["anchor"].value_counts()
plt.show()

print(df[df.anchor == "conh2"])

# the number of symbols in the target are (beautifully) normally distributed
sns.set(font_scale=0.4)
symbols = []
for i in df["target"]:
    symbols.append(len(i))
sns.countplot(x=symbols, color="b")
plt.title("Number of letters in Anchors")
plt.show()

# the targets contain 1-15 words; most of them contain 1 to 3 words
sns.set(font_scale=0.75)
word_count = []
for i in df["target"]:
    word_count.append(len(i.split()))
sns.countplot(x=word_count, color="b")
plt.title("Number of words in Anchors")
plt.show()

# context
# Dropping the int of the context to cluster on general category (called gen_cat)
df["gen_cat"] = 0
for index in df.index:
    df["gen_cat"].iloc[index] = df.context.iloc[index][0]

print(df.context.value_counts())
print(df.context.value_counts().reset_index())


pattern = "[0-9]"
mask = df["target"].str.contains(pattern, na=False)
df["num_target"] = mask
print(df[mask]["target"].value_counts())

df[df.target == "h2o product"]

sns.set(font_scale=1.5)
fig, ax = plt.subplots(figsize=(65, 30))
sns.countplot(
    x=df.context,
    order=df.context.value_counts().index,
    ax=ax,
    color="b",
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
ax.axhline(
    df.context.value_counts().reset_index().describe().loc["25%"][0],
    color="r",
    linewidth=3,
    label="25% percentile"
)
ax.axhline(
    df.context.value_counts().reset_index().describe().loc["50%"][0],
    color="orange",
    linewidth=3,
    label="50% percentile"
)
ax.axhline(
    df.context.value_counts().reset_index().describe().loc["75%"][0],
    color="r",
    linewidth=3,
    label="75% percentile"
)
plt.title("Counts of Context", fontsize=40)
plt.legend(fontsize=40)

sns.set(font_scale=1.5)
fig, ax =plt.subplots(figsize=(25,10))
sns.countplot(
    x=df.gen_cat,
    order=df.gen_cat.value_counts().index,
    ax=ax,
    color="b"
)
ax.set_xticklabels(ax.get_xticklabels());
ax.axhline(
    df.gen_cat.value_counts().reset_index().describe().loc["25%"][0],
    color="r",
    linewidth=3,
    label="25% percentile"
)
ax.axhline(
    df.gen_cat.value_counts().reset_index().describe().loc["50%"][0],
    color="orange",
    linewidth=3,
    label="50% percentile"
)
ax.axhline(
    df.gen_cat.value_counts().reset_index().describe().loc["75%"][0],
    color="r",
    linewidth=3,
    label="75% percentile"
)
plt.title("Counts of general Categories", fontsize=25)
plt.legend(fontsize=15)

print(
    df[df.anchor == "activating position"].context.nunique(),
    df[df.anchor == "activating position"].gen_cat.nunique(),
)

print(df[df.anchor == "activating position"])

# How many unique contexts are given in train?
np.unique(test.context)
f"{len(np.unique(test.context))} unique values"

# Closer look at the contexts which only have a few entries
df[df.contexts == "F26"]
# it will probably be hard to train models on this little data.
# is there a way to arbitrarily increase the combinations for these contexts?

# Closer look at the contexts which only have a few entries
df[df.contexts == "A62"]
# some of these word combinations seem wildly different.
# also, some of these word combinations seem again ambigiously placed: 
# matel phase -> metal of material = 0.5
# metal phase -> metal material = 0.25

list(df["gen_cat"].unique())
# we would expect B, E, F, G and H to be close to another! (just from general domains)
