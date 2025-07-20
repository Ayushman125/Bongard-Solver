import torch
import math
from torch_geometric.data import Data

SHAPES = ["circle","square","triangle","pentagon","star","arc","zigzag","prototype"]
COLORS = ["black","white","red","blue","green"]
FILLS  = ["solid","hollow","striped","dotted"]

def one_hot(idx, size):
    v = [0.0]*size
    v[idx] = 1.0
    return v

def build_scene_graph(objects, cfg):
    """
    objects: list of dicts with keys x,y,size,shape,color,fill
    cfg: config with canvas_size and optional gnn_radius
    """
    node_feats, edge_index = [], [[],[]]
    N = len(objects)
    # Build node features
    for obj in objects:
        fv = []
        # shape one-hot
        fv += one_hot(SHAPES.index(obj.get("shape","circle")), len(SHAPES))
        # color one-hot
        fv += one_hot(COLORS.index(obj.get("color","black")), len(COLORS))
        # fill one-hot
        fv += one_hot(FILLS.index(obj.get("fill","solid")), len(FILLS))
        # normalized size
        fv.append(obj["size"]/cfg.canvas_size)
        # normalized position
        fv.append(obj["x"]/cfg.canvas_size)
        fv.append(obj["y"]/cfg.canvas_size)
        node_feats.append(fv)

    # Build edges: connect nodes closer than r*canvas_size
    r = cfg.gnn_radius * cfg.canvas_size
    for i in range(N):
        xi, yi = objects[i]["x"], objects[i]["y"]
        for j in range(i+1,N):
            xj, yj = objects[j]["x"], objects[j]["y"]
            if math.hypot(xi-xj, yi-yj) <= r:
                edge_index[0] += [i,j]
                edge_index[1] += [j,i]

    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    # Single graph: batch zeros
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data
