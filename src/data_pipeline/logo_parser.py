import re
from math import cos, sin, radians

class LogoParser:
    def parse_logo_script_from_lines(self, lines):
        """
        Accepts a list of LOGO command strings (as lines), returns ordered vertex list.
        This is used for Bongard-LOGO JSON action programs, where each image's program is a list of command strings.
        """
        import re
        from math import cos, sin, radians
        pattern = re.compile(r'(F|B|R|L)\s*(-?\d+(?:\.\d+)?)')
        x, y = 0, 0
        angle = 0
        vertices = [(x, y)]
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = pattern.match(line)
            if not match:
                continue
            cmd, val = match.group(1), float(match.group(2))
            if cmd == 'F':
                rad = radians(angle)
                x += val * cos(rad)
                y += val * sin(rad)
            elif cmd == 'B':
                rad = radians(angle)
                x -= val * cos(rad)
                y -= val * sin(rad)
            elif cmd == 'R':
                angle -= val
                angle %= 360
            elif cmd == 'L':
                angle += val
                angle %= 360
            vertices.append((x, y))
        return vertices
    """
    Parses .logo turtle command scripts into ordered vertex lists.
    """

    def parse_logo_script(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        pattern = re.compile(r'(F|B|R|L)\s*(-?\d+(?:\.\d+)?)')
        x, y = 0, 0
        angle = 0  # Degrees, 0 points right
        vertices = [(x, y)]
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = pattern.match(line)
            if not match:
                continue
            cmd, val = match.group(1), float(match.group(2))
            if cmd == 'F':
                rad = radians(angle)
                x += val * cos(rad)
                y += val * sin(rad)
            elif cmd == 'B':
                rad = radians(angle)
                x -= val * cos(rad)
                y -= val * sin(rad)
            elif cmd == 'R':
                angle -= val
                angle %= 360
            elif cmd == 'L':
                angle += val
                angle %= 360
            vertices.append((x, y))
        return vertices
