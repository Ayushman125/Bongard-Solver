"""
1/3 Detection on Repeated Relations
Phase 1 Module
"""

from typing import List, Tuple

class QuantifierModule:
    def detect(self, relations: List[Tuple[object, object]]) -> dict:
        freq = {}
        output = {"1": [], "3": []}
        for r in relations:
            freq[r] = freq.get(r, 0) + 1
        for r, count in freq.items():
            if count > 1:
                output["1"].append(r)
            else:
                output["3"].append(r)
        return output
