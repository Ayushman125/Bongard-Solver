import sys
from pathlib import Path
nvlabs_path = Path(__file__).parent.parent / "Bongard-LOGO"
sys.path.insert(0, str(nvlabs_path))
from bongard import LineAction, ArcAction

def _universal_action_str(self):
    if isinstance(getattr(self, 'raw_command', None), str):
        return self.raw_command
    if hasattr(self, 'line_type'):
        return f"line_{self.line_type}_{self.line_length}-{self.turn_angle}"
    if hasattr(self, 'arc_type'):
        return f"arc_{self.arc_type}_{self.arc_radius}_{self.arc_angle}-{self.turn_angle}"
    return ""

for cls in (LineAction, ArcAction):
    cls.__str__ = cls.__repr__ = _universal_action_str
