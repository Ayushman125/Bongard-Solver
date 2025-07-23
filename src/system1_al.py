__version__ = "0.1.0"
"""
System-1 Abstraction Layer (S1-AL)
Provides methods to extract domain-invariant features:
 - extract_com(mask) -> (x_com, y_com)
 - extract_inertia_tensor(mask) -> 2×2 numpy array
 - extract_support_polygon(mask) -> shapely Polygon
 - extract_all(mask) -> dict with all above + latency
"""

import time, json
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path

# existing free‐functions (extract_com, extract_inertia_tensor, etc.) go here…

_LOG_PATH = Path("logs/s1_al.jsonl")
_LOG_PATH.parent.mkdir(exist_ok=True, parents=True)

class System1AbstractionLayer:
    def extract_features(self, img: np.ndarray) -> dict:
        """
        Extract features as a dictionary for explainability and test compatibility.
        Keys: com_x, com_y, I00, I01, I10, I11, hole_count, sym_horizontal, sym_vertical, object_count
        """
        from scipy import ndimage
        attrs = self.extract_attributes(img)

        com_x, com_y = attrs['com']
        I = np.asarray(attrs['inertia'], dtype=float).flatten()
        hcount = float(attrs['hole_count'])
        sym = attrs['symmetry']
        hsym = 1.0 if sym.get('horizontal', False) else 0.0
        vsym = 1.0 if sym.get('vertical', False) else 0.0
        mask = (img > 0)
        labeled, count = ndimage.label(mask)
        obj_count = float(count)

        return {
            "com_x": com_x,
            "com_y": com_y,
            "I00": I[0],
            "I01": I[1],
            "I10": I[2],
            "I11": I[3],
            "hole_count": hcount,
            "sym_horizontal": hsym,
            "sym_vertical": vsym,
            "object_count": obj_count
        }
    def extract_attributes(self, img: np.ndarray) -> dict:
        """
        Extract physics-proxy attributes from a 2D mask image.
        Returns a dict with keys: com, inertia, support_surfaces, hole_count, symmetry.
        """
        mask = (img > 0)
        com = self.extract_com(mask)
        inertia = self.extract_inertia_tensor(mask)
        supports = self.extract_support_polygon(mask)
        hole_count = self._count_holes(mask)
        symmetry = self._compute_symmetry(mask)
        return {
            'com': com,
            'inertia': inertia,
            'support_surfaces': supports,
            'hole_count': hole_count,
            'symmetry': symmetry
        }

    def _compute_symmetry(self, mask: np.ndarray) -> dict:
        """
        Returns True/False for horizontal and vertical symmetry of the mask.
        An empty mask or one with perfect mirror symmetry will be True.
        """
        horiz = bool(np.all(mask == np.flipud(mask)))
        vert  = bool(np.all(mask == np.fliplr(mask)))
        return {'horizontal': horiz, 'vertical': vert}

    def _count_holes(self, mask: np.ndarray) -> int:
        from collections import deque
        h, w = mask.shape[:2]
        bg = ~mask
        visited = np.zeros((h, w), dtype=bool)
        q = deque()

        # 1) mark border-connected background
        for i in range(h):
            for j in (0, w-1):
                if bg[i, j] and not visited[i, j]:
                    visited[i, j] = True
                    q.append((i, j))
        for j in range(w):
            for i in (0, h-1):
                if bg[i, j] and not visited[i, j]:
                    visited[i, j] = True
                    q.append((i, j))

        while q:
            x, y = q.popleft()
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = x+dx, y+dy
                if 0 <= nx < h and 0 <= ny < w and bg[nx, ny] and not visited[nx, ny]:
                    visited[nx, ny] = True
                    q.append((nx, ny))

        # 2) anything in bg & not visited is a hole
        hole_count = 0
        for i in range(h):
            for j in range(w):
                if bg[i, j] and not visited[i, j]:
                    hole_count += 1
                    visited[i, j] = True
                    q.append((i, j))
                    while q:
                        x, y = q.popleft()
                        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                            nx, ny = x+dx, y+dy
                            if 0 <= nx < h and 0 <= ny < w and bg[nx, ny] and not visited[nx, ny]:
                                visited[nx, ny] = True
                                q.append((nx, ny))
        return hole_count
    def __init__(self):
        pass

    def extract_com(self, mask: np.ndarray) -> tuple[float, float]:
        """Compute center of mass of binary mask. If mask is empty, default to image center."""
        h, w = mask.shape[:2]
        coords = np.argwhere(mask)
        if coords.size == 0:
            return float(h) / 2, float(w) / 2
        row_coords = coords[:, 0]
        col_coords = coords[:, 1]
        x = row_coords.mean() + 0.5
        y = col_coords.mean() + 0.5
        return x, y

    def extract_inertia_tensor(self, mask: np.ndarray) -> np.ndarray:
        """Compute 2×2 inertia tensor of binary mask."""
        x0, y0 = self.extract_com(mask)
        ys, xs = np.nonzero(mask)
        dx, dy = xs - x0, ys - y0
        Ixx = (dy**2).sum()
        Iyy = (dx**2).sum()
        Ixy = -(dx * dy).sum()
        return np.array([[Ixx, Ixy], [Ixy, Iyy]], dtype=float)

    def extract_support_polygon(self, mask: np.ndarray) -> Polygon:
        """Compute convex‐hull support polygon of binary mask."""
        ys, xs = np.nonzero(mask)
        pts = list(zip(xs, ys))
        return Polygon(pts).convex_hull

    def extract_all(self, mask: np.ndarray) -> dict:
        """
        Run all three feature extractors, log JSON event, 
        and return a dict containing:
          - module, timestamp, latency_ms, com, inertia_tensor, support_polygon
        """
        t0 = time.time()
        com = self.extract_com(mask)
        inertia = self.extract_inertia_tensor(mask).tolist()
        poly = list(self.extract_support_polygon(mask).exterior.coords)
        latency = (time.time() - t0) * 1000
        event = {
            "module": "s1_al",
            "timestamp": time.time(),
            "latency_ms": latency,
            "com": com,
            "inertia_tensor": inertia,
            "support_polygon": poly,
        }
        with _LOG_PATH.open("a") as f:
            f.write(json.dumps(event) + "\n")
        return event
