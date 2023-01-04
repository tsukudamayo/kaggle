import os
import sys
import pickle
import glob
import gc
from collections import Counter
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm


VER = 4
READ_CT = 5
FILES = glob.glob("../input/otto-chunk-data-inparquet-format/*_parquet/*")
CHUNK = int(np.ceil(len(FILES)/6))
type_weight = {
    0: 1,
    1: 6,
    2: 3,
}
type_labels = {
    "clicks": 0,
    "carts": 1,
    "orders": 2,
}

DISK_PIECES = 4
SIZE = 1.86e6/DISK_PIECES


def read_file(f: str) -> pd.DataFrame:
    # pd -> cudf
    df = pd.read_parquet(f)
    df.ts = (df.ts/1000).astype("int32")
    df["type"] = df["type"].map(type_labels).astype("int8")

    return df


def main():
    for PART in range(DISK_PIECES):
        print()
        print("### DISK PART", PART + 1)

        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(FILES))
            print(f"Processing files {a} thru {b - 1} in groups of {READ CT}...")

            for k in range(a, b, READ_CT):
                df = [read_file(FILES[k])]
                for i in range(1, READ_CT):
                    if k+i < b:
                        df.append(read_file(FILES[k+i]))
                    # pd -> cudf
                    df = pd.concat(df, ignore_index=True, axis=0)
                    df = df.sort_values(["session", "ts"], ascending=[True, False])
                    df = df.reset_index(drop=True)
                    df["n"] = df.groupby("session").cumcount()
                    df = df.loc[df.n < 30].drop("n", axis=1)
                    df = df.merge(df, on="session")
                    df = df.loc[((df.ts_x - df.ts_y).abs() < 24*60*60) & (df.aid_x != df.aid_y)]
                    df = df.loc[(df.aid_x >= PART * SIZE) & (df.aid_x < (PART + 1) * SIZE)]
                    df = df[["session", "aid_x", "aid_y", "type_y"]]\
                      .drop_duplicates(["session", "aid_x", "aid_y"])
                    df["wgt"] = df.type_y.map(type_weight)
                    df = df[["aid_x", "aid_y", "wgt"]]
                    df.wgt = df.wgt.astype("float32")
                    df = df.groupby(["aid_x", "aid_y"]).wgt.sum()
                    if k == a:
                        tmp2 = df
                    else:
                        tmp = tmp.add(tmp2, fill_value=0)
                    print(k, ", ", end="")

                print()
                if a == 0:
                    tmp = tmp2
                else:
                    tmp = tmp.add(tmp2, fill_value=0)
                    del tmp2, df
                    gc.collect()

                
                    
                    
