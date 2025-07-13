import json
import networkx as nx
from pathlib import Path

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

def extract_relations(json_dir: str, out_graph_dir: str):
    json_dir = Path(json_dir)
    out_dir  = Path(out_graph_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ann_fp in json_dir.glob("*.json"):
        data = json.load(ann_fp.open())
        G = nx.DiGraph()
        objs = data.get("objects", [])
        # Nodes
        for idx, obj in enumerate(objs):
            G.add_node(idx, class_id=obj["class_id"])

        # Precompute centers & sizes
        centers = [((o["bbox"][0] + o["bbox"][2]) / 2,
                    (o["bbox"][1] + o["bbox"][3]) / 2) for o in objs]
        sizes   = [((o["bbox"][2] - o["bbox"][0]),
                    (o["bbox"][3] - o["bbox"][1])) for o in objs]

        # Edges
        for i in range(len(objs)):
            for j in range(len(objs)):
                if i == j: continue
                xi, yi = centers[i];  wi, hi = sizes[i]
                xj, yj = centers[j];  wj, hj = sizes[j]

                # spatial predicates
                if xi + wi/2 < xj - wj/2:
                    G.add_edge(i, j, relation="left_of")
                if yi + hi/2 < yj - hj/2:
                    G.add_edge(i, j, relation="above")
                if compute_iou(objs[i]["bbox"], objs[j]["bbox"]) > 0.0:
                    G.add_edge(i, j, relation="overlap")

        # Save as JSON node-link
        from networkx.readwrite import json_graph
        node_link = json_graph.node_link_data(G)
        out_fp = out_dir / f"{ann_fp.stem}.json"
        json.dump(node_link, out_fp.open(), indent=2)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir",       default="raw/annotations")
    p.add_argument("--out_graph_dir",  default="graphs")
    args = p.parse_args()
    extract_relations(args.json_dir, args.out_graph_dir)
