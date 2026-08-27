import os
import argparse
import logging

try:
    from src.get_data import read_params, get_data
except ModuleNotFoundError:
    from get_data import read_params, get_data

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")


def load_and_save(config_path: str = "params.yaml") -> None:
    """
    Fetch data from datasource, clean column names, and save to raw dataset location.
    """
    config = read_params(config_path)
    df = get_data(config_path)
    new_cols = [col.replace(" ", "_") for col in df.columns]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    df.to_csv(raw_data_path, sep=",", index=False, header=new_cols, encoding="utf-8")
    logging.info(f"Raw data saved successfully to {raw_data_path} with shape {df.shape}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)