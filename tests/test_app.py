import json


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"


def test_index_get(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Wine Quality Predictor" in response.data


def test_index_post_form_valid(client, valid_input):
    response = client.post("/", data=valid_input)
    assert response.status_code == 200
    assert b"Predicted Wine Quality Score" in response.data


def test_index_post_form_invalid(client, invalid_input_range):
    response = client.post("/", data=invalid_input_range)
    assert response.status_code == 400
    assert b"Validation Error" in response.data


def test_index_post_json_valid(client, valid_input):
    response = client.post(
        "/",
        data=json.dumps(valid_input),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "prediction" in data


def test_predict_api_endpoint(client, valid_input):
    response = client.post(
        "/predict",
        data=json.dumps(valid_input),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "prediction" in data
