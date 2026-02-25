from dataclasses import dataclass
from typing import Any, List


@dataclass
class ExampleItem:
    task_id: str
    category: str
    label: int
    image_path: str
    program: Any


@dataclass
class Episode:
    task_id: str
    category: str
    support_pos: List[ExampleItem]
    support_neg: List[ExampleItem]
    query_pos: ExampleItem
    query_neg: ExampleItem
