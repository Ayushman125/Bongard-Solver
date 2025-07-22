"""
Data Validator
Version: 0.1.0

Enforces JSON schemas for event logs using jsonschema.
Glob-loads all schemas in schemas/.
"""

__version__ = "0.1.0"

import json, glob
from jsonschema import validate, RefResolver

SCHEMA_DIR = "schemas/"

class DataValidator:
    def __init__(self):
        self._schemas = {
            path.rsplit("/",1)[-1]: json.load(open(path))
            for path in glob.glob(f"{SCHEMA_DIR}/*.schema.json")
        }
        self._resolver = RefResolver(f"file:///{SCHEMA_DIR}/", None)

    def validate(self, data: dict, schema_name: str):
        schema = self._schemas[schema_name]
        validate(instance=data, schema=schema, resolver=self._resolver)
