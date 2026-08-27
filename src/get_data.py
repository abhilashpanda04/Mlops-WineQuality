import os
import yaml
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")


def read_params(config_path: str = "params.yaml") -> dict:
    """
    Read YAML configuration parameters from specified path.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_data(config_path: str = "params.yaml") -> pd.DataFrame:
    """
    Load raw data from configured data source path.
    """
    config = read_params(config_path)
    data_path = config["data_source"]["s3_source"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data source CSV file not found at: {data_path}")
    logging.info(f"Reading dataset from {data_path}")
    df = pd.read_csv(data_path, sep=",", encoding="utf-8")
    return df


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)
    logging.info(f"Loaded dataset successfully with shape: {data.shape}")


