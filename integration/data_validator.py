__version__ = "0.1.0"
"""
Data Validator
Version: 0.1.0

Enforces JSON schemas for event logs using jsonschema.
Glob-loads all schemas in schemas/.
"""

import glob
import json
import os
from integration.data_validator import DataValidator

def validate_all():
    dv = DataValidator()
    for path in glob.glob("schemas/*.schema.json"):
        schema_name = os.path.basename(path)
        example = json.load(open(path))
        dv.validate(example, schema_name)
    print("Schema validation OK")
