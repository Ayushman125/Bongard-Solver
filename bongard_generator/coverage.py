"""
Tracks coverage of semantic concepts during generation.
"""
from collections import defaultdict

class CoverageTracker:
    def __init__(self, coverage_target):
        self.coverage_target = coverage_target
        self.rule_coverage = defaultdict(int)
        self.object_coverage = defaultdict(int)

    def record(self, rule, objects):
        """
        Records the generation of a scene for a given rule and objects.
        """
        self.rule_coverage[rule.name] += 1
        for obj in objects:
            # Create a unique key for the object based on its attributes
            obj_key = f"{obj.get('shape', 'none')}_{obj.get('color', 'none')}_{obj.get('size', 'none')}"
            self.object_coverage[obj_key] += 1

    def report(self):
        """
        Prints a report of the current coverage status.
        """
        report_str = "--- Coverage Report ---\n"
        
        report_str += "\nRule Coverage:\n"
        for rule_name, count in sorted(self.rule_coverage.items()):
            report_str += f"  - {rule_name}: {count} scenes\n"
            
        report_str += "\nObject Coverage (Top 20):\n"
        sorted_objects = sorted(self.object_coverage.items(), key=lambda item: item[1], reverse=True)
        for obj_key, count in sorted_objects[:20]:
            report_str += f"  - {obj_key}: {count} instances\n"
            
        report_str += "\n--- End of Report ---\n"
        
        print(report_str)
        return report_str
