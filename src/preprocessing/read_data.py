import json
import pandas as pd


def read_data_json(prefix: str = ""):
    """
    Reads data from .json file
    Parameters:
        prefix (str): directory prefix
    """
    with open(
        f"{prefix}midterm-2018/midterm-2018_processed_user_objects.json", "r"
    ) as f:
        data = json.load(f)
    users_df = pd.DataFrame(data)
    users_df["description"] = (
        users_df["description"]
        .fillna("")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("\r", " ")
    )
    return users_df


def read_labels(prefix: str = "") -> pd.DataFrame:
    """
    Reads labels from .csv file
    Parameters:
        prefix (str): directory prefix
    """
    file_path = f"{prefix}midterm-2018/midterm-2018.tsv"

    labels = pd.read_csv(file_path, sep="\t", header=None)
    labels.columns = ["user_id", "label"]
    return labels


if __name__ == "__main__":
    read_data_json()
