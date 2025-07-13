import json, cv2, numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from pathlib import Path
from argparse import ArgumentParser

DSL_MAP = {
    "left_of":  "(move obj{0} left-of obj{1})",
    "above":    "(move obj{0} above obj{1})",
    "overlap":  "(group obj{0} obj{1})"
}

def compute_angle_and_scale(mask):
    ys, xs = np.where(mask>0)
    if len(xs)<5:
        return 0.0, 1.0
    pts = np.vstack((xs,ys)).astype(np.float32).T
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    area  = len(xs)
    return angle, area

def synthesize_programs(graph_dir, mask_dir, out_dir):
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    for gp in Path(graph_dir).glob("*.json"):
        G    = json_graph.node_link_graph(json.load(gp.open()))
        img_id = gp.stem
        # load combined mask (pixel values = object_id)
        mask_fp = Path(mask_dir)/f"{img_id}.png"
        mask    = cv2.imread(str(mask_fp), cv2.IMREAD_UNCHANGED) if mask_fp.exists() else None

        steps = []
        # basic relational ops
        for u, v, d in G.edges(data=True):
            rel = d.get("relation")
            if rel in DSL_MAP:
                steps.append(DSL_MAP[rel].format(u, v))

        # rotation & scale
        if mask is not None:
            # compute median area for normalization
            areas = [(mask==(i+1)).sum() for i in G.nodes()]
            median = np.median(areas) or 1
            for node in G.nodes():
                obj_mask = (mask==(node+1)).astype(np.uint8)
                angle, area = compute_angle_and_scale(obj_mask)
                scale = np.sqrt(area/median)
                steps.append(f"(rotate obj{node} {angle:.1f})")
                steps.append(f"(scale obj{node} {scale:.2f})")

        prog = "\n".join(steps) or "; no operations"
        (Path(out_dir)/f"{img_id}.lisp").write_text(prog)

if __name__=="__main__":
    p = ArgumentParser()
    p.add_argument("--graph_dir", default="graphs")
    p.add_argument("--mask_dir",  default="masks")
    p.add_argument("--out_dir",   default="programs")
    args = p.parse_args()
    synthesize_programs(args.graph_dir, args.mask_dir, args.out_dir)
import json
import cv2
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from pathlib import Path
from argparse import ArgumentParser

DSL_MAP = {
    "left_of": "(move obj{0} left-of obj{1})",
    "above":   "(move obj{0} above obj{1})",
    "overlap": "(group obj{0} obj{1})"
}

def compute_mask_angle(mask):
    # Fit a rotated rectangle to the mask to get angle
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return 0.0
    pts = np.vstack((xs, ys)).T.astype(np.float32)
    box = cv2.minAreaRect(pts)
    angle = box[-1]
    return angle

def synthesize_programs(graph_dir, mask_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for gfp in Path(graph_dir).glob("*.json"):
        G = json_graph.node_link_graph(json.load(gfp.open()))
        img_id = gfp.stem
        # load corresponding mask
        mask_fp = Path(mask_dir) / f"{img_id}.png"
        mask_agg = cv2.imread(str(mask_fp), cv2.IMREAD_UNCHANGED) if mask_fp.exists() else None

        steps = []
        # simple relational ops
        for u, v, d in G.edges(data=True):
            rel = d.get("relation")
            if rel in DSL_MAP:
                steps.append(DSL_MAP[rel].format(u, v))

        # scale & rotate per node if mask available
        if mask_agg is not None:
            # object IDs stored as pixel values
            for node in G.nodes():
                single_mask = (mask_agg == (node+1)).astype(np.uint8)
                area = single_mask.sum()
                angle = compute_mask_angle(single_mask)
                # scale relative to median area of all objs
                areas = [(mask_agg==(i+1)).sum() for i in G.nodes()]
                median = np.median(areas) or 1
                scale = np.sqrt(area/median)
                steps.append(f"(rotate obj{node} {angle:.1f})")
                steps.append(f"(scale obj{node} {scale:.2f})")

        prog = "\n".join(steps) if steps else "; no operations"
        (Path(out_dir)/f"{img_id}.lisp").write_text(prog)

if __name__=="__main__":
    p = ArgumentParser()
    p.add_argument("--graph_dir", default="graphs")
    p.add_argument("--mask_dir",  default="masks/seg")  
    p.add_argument("--out_dir",   default="programs")
    args = p.parse_args()
    synthesize_programs(args.graph_dir, args.mask_dir, args.out_dir)
import json
import networkx as nx
from pathlib import Path
import json
import networkx as nx
from pathlib import Path

DSL_OPS = {
    "left_of": "(move {0} left_of {1})",
    "above":  "(move {0} above {1})",
    "overlap":"(group {0} {1})"
}

def synthesize(json_graph_dir, out_prog_dir):
    Path(out_prog_dir).mkdir(exist_ok=True, parents=True)

    from networkx.readwrite import json_graph
    for gp in graph_dir.glob("*.json"):
        G = json_graph.node_link_graph(json.loads(gp.read_text()))
        steps = []
        for u, v, d in G.edges(data=True):
            rel = d.get("relation")
            if rel in DSL_OPS:
                steps.append(DSL_MAP[rel].format(u, v))

        prog = "\n".join(steps) or "; no relations"
        (Path(out_prog_dir) / f"{gp.stem}.lisp").write_text(prog)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--graph_dir",   default="graphs")
    p.add_argument("--out_prog_dir",default="programs")
    args = p.parse_args()
    synthesize(args.graph_dir, args.out_prog_dir)
