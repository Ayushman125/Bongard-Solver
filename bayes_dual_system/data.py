import json
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

from .types import Episode, ExampleItem


class RawShapeBongardLoader:
    def __init__(
        self,
        root_path: str,
        split_file: Optional[str] = None,
    ) -> None:
        self.root_path = root_path
        self.split_file = split_file or os.path.join(root_path, "ShapeBongard_V2_split.json")

        with open(self.split_file, "r", encoding="utf-8") as f:
            self.splits: Dict[str, List[str]] = json.load(f)

        self.programs: Dict[str, Dict[str, Tuple[List, List]]] = {}
        for category in ["ff", "bd", "hd"]:
            prog_file = os.path.join(root_path, category, f"{category}_action_programs.json")
            with open(prog_file, "r", encoding="utf-8") as f:
                self.programs[category] = json.load(f)

    def available_splits(self) -> List[str]:
        return sorted(self.splits.keys())

    @staticmethod
    def infer_category(task_id: str) -> str:
        return task_id.split("_")[0]

    def _image_path(self, task_id: str, label: int, image_index: int) -> str:
        category = self.infer_category(task_id)
        return os.path.join(
            self.root_path,
            category,
            "images",
            task_id,
            str(label),
            f"{image_index}.png",
        )

    def _task_programs(self, task_id: str) -> Tuple[List, List]:
        category = self.infer_category(task_id)
        pos, neg = self.programs[category][task_id]
        return pos, neg

    def _example(self, task_id: str, label: int, image_index: int) -> ExampleItem:
        category = self.infer_category(task_id)
        pos_programs, neg_programs = self._task_programs(task_id)
        program = pos_programs[image_index] if label == 1 else neg_programs[image_index]
        return ExampleItem(
            task_id=task_id,
            category=category,
            label=label,
            image_path=self._image_path(task_id, label, image_index),
            program=program,
        )

    def build_episode(
        self,
        task_id: str,
        query_index: int = 6,
        support_indices: Optional[List[int]] = None,
    ) -> Episode:
        if support_indices is None:
            support_indices = [idx for idx in range(7) if idx != query_index]

        support_pos = [self._example(task_id, 1, idx) for idx in support_indices]
        support_neg = [self._example(task_id, 0, idx) for idx in support_indices]
        query_pos = self._example(task_id, 1, query_index)
        query_neg = self._example(task_id, 0, query_index)

        return Episode(
            task_id=task_id,
            category=self.infer_category(task_id),
            support_pos=support_pos,
            support_neg=support_neg,
            query_pos=query_pos,
            query_neg=query_neg,
        )

    def iter_split(
        self,
        split_name: str,
        query_index: int = 6,
        max_episodes: Optional[int] = None,
        seed: int = 0,
        shuffle: bool = False,
    ) -> Iterator[Episode]:
        task_ids = list(self.splits[split_name])
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(task_ids)

        if max_episodes is not None:
            task_ids = task_ids[:max_episodes]

        for task_id in task_ids:
            yield self.build_episode(task_id=task_id, query_index=query_index)
