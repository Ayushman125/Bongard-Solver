import json
import pytest
from integration.data_validator import DataValidator

def test_validate_success(tmp_path):
    example = {
      "module": "s1_al", "timestamp": 0, "latency_ms": 1,
      "com": [1.0,2.0],
      "inertia_tensor": [[1,0],[0,1]],
      "support_polygon": [[0,0],[1,0],[1,1]]
    }
    dv = DataValidator()
    # assume schemas/s1_al_output.schema.json is present
    dv.validate(example, "s1_al_output.schema.json")

def test_validate_failure():
    dv = DataValidator()
    with pytest.raises(Exception):
        dv.validate({}, "s1_al_output.schema.json")
