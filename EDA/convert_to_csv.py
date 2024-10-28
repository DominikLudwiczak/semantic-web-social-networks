from read_data import read_data_json, read_labels
import pandas as pd


def convert_to_csv(prefix=""):
    users_df = read_data_json(prefix=prefix)
    labels = read_labels(prefix=prefix)

    merged = pd.merge(users_df, labels, on="user_id")
    merged.to_csv(f"{prefix}midterm-2018/data.csv", index=False)
