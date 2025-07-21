import torch
import math
from torch_geometric.data import Data

SHAPES = ["circle", "square", "triangle", "pentagon", "star", "arc", "zigzag", "prototype", "line", "hexagon"]
COLORS = ["black", "white", "red", "blue", "green"]
FILLS = ["solid", "hollow", "striped", "dotted", "texture"]

def one_hot(idx, size):
    v = [0.0] * size
    v[idx] = 1.0
    return v

def build_scene_graph(objects, cfg):
    """
    objects: list of dicts with keys x,y,size,shape,color,fill
    cfg: config with canvas_size and optional gnn_radius
    """
    node_feats, edge_index = [], [[], []]
    N = len(objects)
    # Ensure canvas_size is always int
    if hasattr(cfg, 'canvas_size'):
        try:
            cfg.canvas_size = int(cfg.canvas_size)
        except Exception:
            cfg.canvas_size = 128
    # Build node features
    for obj in objects:
        fv = []
        # shape one-hot (handle unknown shape)
        shape_val = obj.get("shape", "circle")
        try:
            shape_idx = SHAPES.index(shape_val)
        except ValueError:
            shape_idx = 0
        fv += one_hot(shape_idx, len(SHAPES))
        # color one-hot (handle unknown color)
        color_val = obj.get("color", "black")
        try:
            color_idx = COLORS.index(color_val)
        except ValueError:
            color_idx = 0
        fv += one_hot(color_idx, len(COLORS))
        # fill one-hot (handle unknown fill)
        fill_val = obj.get("fill", "solid")
        try:
            fill_idx = FILLS.index(fill_val)
        except ValueError:
            fill_idx = 0
        fv += one_hot(fill_idx, len(FILLS))
        # normalized size
        # Ensure obj["size"] is numeric before division
        size_val = obj["size"]
        if isinstance(size_val, str):
            try:
                size_val = int(size_val)
            except ValueError:
                size_val = 0
        fv.append(size_val / cfg.canvas_size)
        # normalized position
        fv.append(obj["x"] / cfg.canvas_size)
        fv.append(obj["y"] / cfg.canvas_size)
        node_feats.append(fv)

    # Build edges: connect nodes closer than r*canvas_size
    gnn_radius = getattr(cfg, 'gnn_radius', 0.15)  # Default to 0.15 if not present
    r = gnn_radius * cfg.canvas_size
    for i in range(N):
        xi, yi = objects[i]["x"], objects[i]["y"]
        for j in range(i + 1, N):
            xj, yj = objects[j]["x"], objects[j]["y"]
            if math.hypot(xi - xj, yi - yj) <= r:
                edge_index[0] += [i, j]
                edge_index[1] += [j, i]

    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    # Single graph: batch zeros
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data
