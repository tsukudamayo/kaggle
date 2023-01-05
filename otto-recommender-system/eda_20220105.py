from datetime import timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


DATA_PATH = Path("../input/otto-recommender-system")
TRAIN_PATH = DATA_PATH/"train.jsonl"
TEST_PATH = DATA_PATH/"test.jsonl"
SAMPLE_SUB_PATH = Path("../input/otto-recommender-system/sample_submission.csv")


def main():
    sample_size = 150000
    chunks = pd.read_json(TRAIN_PATH, lines=True, chunksize=sample_size)

    sample_train_df = None
    for c in chunks:
        sample_train_df = c
        break

    if sample_train_df is not None:
        # Data structure
        sample_train_df.set_index("session", drop=True, inplace=True)
        print(sample_train_df.head())

        example_session = sample_train_df.iloc[0].item()
        print("len(example_session):", len(example_session))
        print("example_session[0]: ", example_session[0])

        time_elapsed = example_session[-1]["ts"] - example_session[0]["ts"]
        print("str(timedelta(milliseconds=time_elapsed)): ", str(timedelta(milliseconds=time_elapsed)))
        action_counts = {}
        for action in example_session:
            action_counts[action["type"]] = action_counts.get(action["type"], 0) + 1
        print("action_counts: ", action_counts)

        # Initial EDA
        action_counts_list, article_id_counts_list, session_length_time_list, session_length_action_list = ([] for i in range(4))
        overall_action_counts = {}
        overall_article_id_counts = {}

        for i, row in tqdm(sample_train_df.iterrows(), total=len(sample_train_df)):
            actions = row["events"]
            action_counts = {}
            article_id_counts = {}
            for action in actions:
                action_counts[action["type"]] = action_counts.get(action["type"], 0) + 1
                article_id_counts[action["aid"]] = article_id_counts.get(action["aid"], 0) + 1
                overall_action_counts[action["type"]] = overall_action_counts.get(action["type"], 0) + 1
                overall_article_id_counts[action["aid"]] = overall_article_id_counts.get(action["aid"], 0) + 1

            session_length_time = actions[-1]["ts"] - actions[0]["ts"]

            action_counts_list.append(action_counts)
            article_id_counts_list.append(article_id_counts)
            session_length_time_list.append(session_length_time)
            session_length_action_list.append(len(actions))

        sample_train_df["action_counts"] = action_counts_list
        sample_train_df["article_id_counts"] = article_id_counts_list
        sample_train_df["session_length_unix"] = session_length_time_list
        sample_train_df["session_length_hours"] = sample_train_df["session_length_unix"]*2.77778e-7
        sample_train_df["session_length_action"] = session_length_action_list

        # Actions
        total_actions = sum(overall_action_counts.values())
        plt.figure(figsize=(8, 6))
        sns.barplot(
            x=list(overall_action_counts.keys()),
            y=[i/total_actions for i in overall_action_counts.values()]
        )
        plt.title(f"Action frequency", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.ylabel("Category", fontsize=12)
        plt.show()
        

if __name__ == "__main__":
    main()
