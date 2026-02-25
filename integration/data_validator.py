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

def validate_all():
    dv = DataValidator()
    for path in glob.glob("schemas/*.schema.json"):
        schema_name = os.path.basename(path)
        example = json.load(open(path))
        dv.validate(example, schema_name)
    print("Schema validation OK")

class DataValidator:
    def __init__(self):
        import jsonschema, glob, os, json
        self._schemas = {}
        for path in glob.glob('schemas/*.schema.json'):
            name = os.path.basename(path)
            with open(path, 'r') as f:
                self._schemas[name] = json.load(f)

    def validate(self, data: dict, schema_name: str):
        if schema_name not in self._schemas:
            raise KeyError(f"Unknown schema: {schema_name}")
        import jsonschema
        jsonschema.validate(data, self._schemas[schema_name])
