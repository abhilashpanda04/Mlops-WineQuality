import pytest
import numpy as np
from prediction_service.prediction import (
    validate_input,
    predict,
    api_response,
    NotInRange,
    FormValidationError
)


def test_validate_input_valid(valid_input):
    vals = validate_input(valid_input)
    assert len(vals) == 11
    assert vals[0] == 7.4
    assert vals[-1] == 9.4


def test_validate_input_out_of_range(invalid_input_range):
    with pytest.raises(NotInRange):
        validate_input(invalid_input_range)


def test_validate_input_invalid_type(invalid_input_type):
    with pytest.raises(FormValidationError):
        validate_input(invalid_input_type)


def test_validate_input_missing_key(valid_input):
    incomplete = valid_input.copy()
    del incomplete["alcohol"]
    with pytest.raises(FormValidationError):
        validate_input(incomplete)


def test_predict(valid_input):
    vals = np.array([validate_input(valid_input)])
    score = predict(vals)
    assert isinstance(score, float)
    assert 0 <= score <= 10


def test_api_response_success(valid_input):
    res = api_response(valid_input)
    assert res["status"] == "success"
    assert "prediction" in res
    assert 0 <= res["prediction"] <= 10


def test_api_response_failure(invalid_input_range):
    res = api_response(invalid_input_range)
    assert res["status"] == "error"
    assert "error" in res
