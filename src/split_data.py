import os
import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.get_data import read_params
except ModuleNotFoundError:
    from get_data import read_params

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")


def split_and_saved_data(config_path: str = "params.yaml") -> None:
    """
    Split raw dataset into train and test sets and save to processed directory.
    """
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset CSV not found at: {raw_data_path}")

    df = pd.read_csv(raw_data_path, sep=",")
    train, test = train_test_split(
        df,
        test_size=split_ratio,
        random_state=random_state
    )

    os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)

    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")
    logging.info(f"Split dataset: Train shape={train.shape}, Test shape={test.shape}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)