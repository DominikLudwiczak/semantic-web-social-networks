from preprocessing.read_data import read_data_json, read_labels
import pandas as pd


def create_features() -> tuple[pd.DataFrame, list[int]]:
    df = pd.read_csv("midterm-2018/data.csv")
    dtype_spec = {
        "description": "str",
        "lang": "str",
        "verified": "bool",
        "geo_enabled": "bool",
        "profile_use_background_image": "bool",
        "default_profile": "bool",
        "followers_count": "int64",
        "friends_count": "int64",
        "listed_count": "int64",
        "favourites_count": "int64",
        "statuses_count": "int64",
        "label": "str",
    }

    for column, dtype in dtype_spec.items():
        df[column] = df[column].astype(dtype)

    # Description
    df["description"] = df["description"].apply(len).astype("Int32")

    # Creation Date
    df["user_created_at"] = pd.to_datetime(df["user_created_at"], errors="coerce")
    df["user_created_at"] = df["user_created_at"].astype("int64") // 10**9

    # Language
    df = pd.get_dummies(df, columns=["lang"], prefix="lang", drop_first=True)

    # Bool as 0 1
    df[df.select_dtypes(include=["bool"]).columns] = df.select_dtypes(
        include=["bool"]
    ).astype(int)

    # Round int columns
    columns_to_round = [
        "followers_count",
        "friends_count",
        "listed_count",
        "favourites_count",
        "statuses_count",
    ]
    df[columns_to_round] = (
        df[columns_to_round].apply(lambda x: (x / 100).round() * 100).astype("Int64")
    )

    y = df["label"].map({"human": 0, "bot": 1})
    y = list(y)

    X = df.drop(
        columns=[
            "probe_timestamp",
            "user_id",
            "screen_name",
            "name",
            "url",
            "protected",
            "tid",
            "label",
        ],
    )
    return X, y
