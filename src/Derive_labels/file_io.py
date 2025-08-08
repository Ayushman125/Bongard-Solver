import os
import csv

class FileIO:
    _shape_attributes = None
    _shape_defs = None

    @staticmethod
    def _load_tsv(path):
        if not os.path.exists(path):
            return []
        with open(path, newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f, delimiter='\t'))

    @classmethod
    def get_shape_attributes(cls):
        if cls._shape_attributes is None:
            tsv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Bongard-LOGO', 'data', 'human_designed_shapes_attributes.tsv'))
            cls._shape_attributes = cls._load_tsv(tsv_path)
        return cls._shape_attributes

    @classmethod
    def get_shape_defs(cls):
        if cls._shape_defs is None:
            tsv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Bongard-LOGO', 'data', 'human_designed_shapes.tsv'))
            cls._shape_defs = cls._load_tsv(tsv_path)
        return cls._shape_defs

    @classmethod
    def get_shape_attribute_map(cls):
        # Map from shape function name to attribute dict
        return {row['shape function name']: row for row in cls.get_shape_attributes() if row.get('shape function name')}

    @classmethod
    def get_shape_def_map(cls):
        return {row['shape function name']: row for row in cls.get_shape_defs() if row.get('shape function name')}