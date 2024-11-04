import json
import pandas as pd


def read_data_json(prefix=""):

    with open(
        f"{prefix}midterm-2018/midterm-2018_processed_user_objects.json", "r"
    ) as f:
        data = json.load(f)
        users_df = pd.DataFrame(data)
        return users_df


def read_labels(prefix=""):
    file_path = f"{prefix}midterm-2018/midterm-2018.tsv"

    labels = pd.read_csv(file_path, sep="\t")
    labels.columns = ["user_id", "label"]
    return labels


if __name__ == "__main__":
    read_data_json()
