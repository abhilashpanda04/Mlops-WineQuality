# Wine Quality Prediction with MLOps & Flask

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Web%20Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![MLOps](https://img.shields.io/badge/Data%20Version%20Control-DVC-orange.svg)](https://dvc.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end production-ready MLOps project that builds, tracks, evaluates, and serves a machine learning regression model (ElasticNet) to predict red wine quality based on physicochemical properties.

---

## 📌 Project Overview

This repository demonstrates best practices for building an automated, reproducible machine learning pipeline (MLOps) integrated with a modern Flask web interface and REST API endpoints.

Key features:
- **Reproducible ML Pipeline**: Orchestrated via **DVC (Data Version Control)** to manage data extraction, preprocessing, train-test splitting, model training, and metrics tracking.
- **Robust Machine Learning Model**: Scikit-Learn **ElasticNet** regression predicting wine quality scores on a scale from 0 to 10.
- **Modern Web Interface**: Clean, responsive **Bootstrap 5 UI** with preset sample buttons, form input validation, and score visualization.
- **RESTful API**: `/predict` and `/` endpoints supporting JSON payloads with custom domain validation and exception handling.
- **Testing & Quality Assurance**: Automated unit and integration tests via **Pytest**, multi-environment matrix testing via **Tox**, and linting with **Flake8**.
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing, pipeline validation, and deployment readiness.

---

## 📂 Repository Structure

```
Mlops-WineQuality/
├── .github/workflows/     # CI/CD GitHub Actions workflow definition
│   └── ci-cd.yaml
├── data/                  # DVC-managed directory
│   ├── raw/               # Cleaned raw data csv
│   └── processed/         # Train and test datasets
├── data_given/            # Raw input source dataset tracked by DVC
│   └── winequality.csv
├── notebooks/             # Exploratory Data Analysis (EDA) notebooks
├── prediction_service/    # Core prediction engine & validation module
│   ├── model/             # Synced active model binary (model.joblib)
│   └── prediction.py      # Input validation & inference methods
├── report/                # Tracked metrics and hyperparameter reports
│   ├── params.json
│   └── scores.json
├── saved_models/          # Model artifact directory
├── src/                   # Pipeline source modules
│   ├── get_data.py        # Config & data loader
│   ├── load_data.py       # Raw dataset processing
│   ├── split_data.py      # Train/test splitter
│   └── train_and_evaluate.py # ElasticNet training & evaluation
├── tests/                 # Unit & integration tests
│   ├── conftest.py        # Pytest fixtures
│   ├── test_app.py        # Flask app endpoint tests
│   ├── test_config.py     # Configuration tests
│   └── test_prediction.py # Validation & inference tests
├── webapp/                # Flask web application assets
│   ├── static/            # CSS & JS scripts
│   └── templates/         # Jinja2 HTML templates (Bootstrap 5)
├── app.py                 # Flask web server & API entrypoint
├── dvc.yaml               # DVC pipeline stages and dependencies
├── params.yaml            # Central configuration & hyperparameters
├── requirements.txt       # Python dependency specifications
├── setup.py               # Package setup file
└── tox.ini                # Tox test configuration
```

---

## ⚙️ Quickstart & Local Setup

### 1. Prerequisites
- Python 3.9+
- Git & DVC

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/abhilashpanda04/Mlops-WineQuality.git
cd Mlops-WineQuality

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and editable package
pip install -r requirements.txt
```

---

## 🔄 Reproducing the ML Pipeline with DVC

The ML pipeline consists of three sequential stages defined in `dvc.yaml`:
1. `load_data`: Cleans raw dataset column headers and saves to `data/raw/`.
2. `split_data`: Performs train-test split (`80/20`) based on `params.yaml`.
3. `train_and_evaluate`: Fits `ElasticNet`, logs evaluation metrics (RMSE, MAE, R2), and saves model artifacts.

To reproduce the full pipeline:

```bash
dvc repro
```

To view current metrics and metric diffs:

```bash
# Display evaluation scores
dvc metrics show

# Compare metrics across git commits/experiments
dvc metrics diff
```

---

## 🚀 Running the Flask Application

Start the Flask development server:

```bash
python app.py
```

Open your browser and navigate to:
- **Web UI**: [http://localhost:5000](http://localhost:5000)
- **Health Check**: [http://localhost:5000/health](http://localhost:5000/health)

---

## 🔌 API Usage

### Endpoint: `POST /predict`

Send a `POST` request with a JSON body containing all 11 physicochemical feature inputs:

#### Request (cURL):
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.70,
    "citric_acid": 0.00,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

#### Success Response (`200 OK`):
```json
{
  "prediction": 5.58,
  "status": "success"
}
```

#### Error Response (`400 Bad Request`):
```json
{
  "error": "Parameter 'fixed_acidity' value 999.0 is outside expected range (4.0 to 16.0)",
  "status": "error"
}
```

---

## 🧪 Testing & Code Quality

Run the test suite with verbose output:

```bash
pytest -v
```

Run linter checks:

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

Run test suite across Python versions via Tox:

```bash
tox
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

