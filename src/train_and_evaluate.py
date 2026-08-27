import os
import sys
import json
import joblib
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

try:
    from src.get_data import read_params
except ModuleNotFoundError:
    from get_data import read_params

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")


def eval_metrics(actual, pred):
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    r2 = float(r2_score(actual, pred))
    return rmse, mae, r2


def train_and_evaluate(config_path: str = "params.yaml") -> None:
    """
    Train ElasticNet model, evaluate metrics, save reports, and sync model artifact.
    """
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    webapp_model_dir = config.get("webapp_model_dir", "prediction_service/model/model.joblib")

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = [config["base"]["target_col"]]

    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Processed train ({train_data_path}) or test ({test_data_path}) datasets missing.")

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    lr = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state
    )
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    logging.info(f"Elasticnet model (alpha={alpha:.6f}, l1_ratio={l1_ratio:.6f}):")
    logging.info(f"  RMSE: {rmse:.4f}")
    logging.info(f"  MAE:  {mae:.4f}")
    logging.info(f"  R2:   {r2:.4f}")

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    os.makedirs(os.path.dirname(scores_file), exist_ok=True)
    os.makedirs(os.path.dirname(params_file), exist_ok=True)

    with open(scores_file, "w") as f:
        json.dump({"rmse": rmse, "mae": mae, "r2": r2}, f, indent=4)

    with open(params_file, "w") as f:
        json.dump({"alpha": alpha, "l1_ratio": l1_ratio}, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(lr, model_path)
    logging.info(f"Model saved to {model_path}")

    # Synchronize model artifact to webapp model directory
    os.makedirs(os.path.dirname(webapp_model_dir), exist_ok=True)
    joblib.dump(lr, webapp_model_dir)
    logging.info(f"Synced model copy to webapp model path {webapp_model_dir}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)