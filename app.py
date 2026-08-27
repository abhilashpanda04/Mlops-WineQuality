import os
from flask import Flask, render_template, request, jsonify
import numpy as np

from prediction_service.prediction import (
    predict,
    validate_input,
    api_response,
    NotInRange,
    FormValidationError,
    FEATURE_KEYS,
    EXPECTED_RANGES
)

params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for service monitoring."""
    return jsonify({"status": "healthy", "service": "Wine Quality Prediction API"}), 200


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.is_json:
                json_data = request.get_json()
                response = api_response(json_data, config_path=params_path)
                status_code = 200 if response.get("status") == "success" else 400
                return jsonify(response), status_code
            elif request.form:
                form_data = dict(request.form)
                validated_vals = validate_input(form_data)
                data = np.array([validated_vals])
                prediction_res = predict(data, config_path=params_path)
                return render_template("index.html", response=prediction_res, form_data=form_data)
        except (NotInRange, FormValidationError) as e:
            return render_template("index.html", error=str(e), form_data=request.form), 400
        except Exception as e:
            return render_template("index.html", error=f"Unexpected error: {str(e)}", form_data=request.form), 500
    
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    """Dedicated API endpoint accepting JSON payload."""
    try:
        json_data = request.get_json(force=True)
        response = api_response(json_data, config_path=params_path)
        status_code = 200 if response.get("status") == "success" else 400
        return jsonify(response), status_code
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
