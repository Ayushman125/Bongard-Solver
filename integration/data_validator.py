__version__ = "0.1.0"
"""
Data Validator
Version: 0.1.0

Enforces JSON schemas for event logs using jsonschema.
Glob-loads all schemas in schemas/.
"""

import json, glob
from jsonschema import validate, RefResolver

SCHEMA_DIR = "schemas/"

class DataValidator:
    def __init__(self, schema_dir="schemas"):
        import os
        self._schemas = {}
        for path in glob.glob(os.path.join(schema_dir, "*.schema.json")):
            name = os.path.basename(path)
            with open(path) as f:
                self._schemas[name] = json.load(f)
        self._resolver = RefResolver(f"file:///{schema_dir}/", None)

    def validate(self, data: dict, schema_name: str):
        schema = self._schemas[schema_name]
        validate(instance=data, schema=schema, resolver=self._resolver)
