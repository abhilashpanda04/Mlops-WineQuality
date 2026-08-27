import pytest
from prediction_service.prediction import read_params, NotInRange, FormValidationError


def test_read_params():
    config = read_params("params.yaml")
    required_keys = ["base", "data_source", "load_data", "split_data", "estimators", "reports", "model_dir"]
    for key in required_keys:
        assert key in config, f"Key '{key}' missing from params.yaml"


def test_not_in_range_exception():
    with pytest.raises(NotInRange) as exc_info:
        raise NotInRange("Value out of bound")
    assert "Value out of bound" in str(exc_info.value)


def test_form_validation_error_exception():
    with pytest.raises(FormValidationError) as exc_info:
        raise FormValidationError("Missing parameter")
    assert "Missing parameter" in str(exc_info.value)