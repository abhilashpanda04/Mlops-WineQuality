import os
import yaml
import joblib
import numpy as np
import pandas as pd
from typing import Union

# Feature list in expected model input order
FEATURE_KEYS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

# Expected domain numeric ranges for features
EXPECTED_RANGES = {
    "fixed_acidity": (4.0, 16.0),
    "volatile_acidity": (0.1, 2.0),
    "citric_acid": (0.0, 1.5),
    "residual_sugar": (0.5, 20.0),
    "chlorides": (0.01, 1.0),
    "free_sulfur_dioxide": (1.0, 100.0),
    "total_sulfur_dioxide": (5.0, 350.0),
    "density": (0.98, 1.05),
    "pH": (2.5, 4.5),
    "sulphates": (0.2, 2.5),
    "alcohol": (7.0, 16.0)
}


class NotInRange(Exception):
    def __init__(self, message="Value not in expected range"):
        self.message = message
        super().__init__(self.message)


class FormValidationError(Exception):
    def __init__(self, message="Invalid input parameters"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path: str = "params.yaml") -> dict:
    """Read YAML parameters file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file missing at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def validate_input(dict_request: dict) -> list:
    """
    Validate input dictionary keys, data types, and value bounds.
    Returns array of floats in feature order.
    """
    validated_values = []
    for key in FEATURE_KEYS:
        if key not in dict_request:
            raise FormValidationError(f"Missing required parameter: '{key}'")
        try:
            val = float(dict_request[key])
        except (ValueError, TypeError):
            raise FormValidationError(f"Parameter '{key}' must be a valid float number, got '{dict_request[key]}'")

        min_val, max_val = EXPECTED_RANGES[key]
        if not (min_val <= val <= max_val):
            raise NotInRange(f"Parameter '{key}' value {val} is outside expected range ({min_val} to {max_val})")

        validated_values.append(val)

    return validated_values


def predict(data: Union[np.ndarray, pd.DataFrame], config_path: str = "params.yaml") -> float:
    """
    Load trained model and compute quality prediction.
    """
    config = read_params(config_path)
    model_path = config.get("webapp_model_dir", "prediction_service/model/model.joblib")
    
    if not os.path.exists(model_path):
        fallback = os.path.join(config.get("model_dir", "saved_models"), "model.joblib")
        if os.path.exists(fallback):
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or {fallback}. Train the model first.")

    model = joblib.load(model_path)
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=FEATURE_KEYS)

    prediction = model.predict(data)
    result = float(np.clip(prediction[0], 0, 10))
    return round(result, 2)



def api_response(dict_request: dict, config_path: str = "params.yaml") -> dict:
    """
    Process dictionary payload for API endpoint and return standard response dict.
    """
    try:
        data = np.array([validate_input(dict_request)])
        response = predict(data, config_path=config_path)
        return {"prediction": response, "status": "success"}
    except (NotInRange, FormValidationError) as e:
        return {"error": str(e), "status": "error"}
    except Exception as e:
        return {"error": f"Internal prediction error: {str(e)}", "status": "error"}
