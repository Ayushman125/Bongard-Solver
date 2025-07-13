import networkx as nx
import json
import numpy as np
import os
from collections import defaultdict

def load_graph(path):
    """Load a relational graph from JSON."""
    with open(path) as f:
        return nx.node_link_graph(json.load(f))

def load_yolo(path):
    """Load YOLO predictions from txt file."""
    preds = []
    for line in open(path):
        c,x,y,w,h = map(float, line.split())
        preds.append({'class':int(c),'x':x,'y':y,'w':w,'h':h})
    return preds

def iou(boxA, boxB):
    """Compute Intersection-over-Union for two boxes."""
    xA1, yA1, wA, hA = boxA['x'], boxA['y'], boxA['w'], boxA['h']
    xB1, yB1, wB, hB = boxB['x'], boxB['y'], boxB['w'], boxB['h']
    # Convert YOLO center format to corners
    def to_corners(x, y, w, h):
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return x1, y1, x2, y2
    a1, a2, a3, a4 = to_corners(xA1, yA1, wA, hA)
    b1, b2, b3, b4 = to_corners(xB1, yB1, wB, hB)
    # Intersection
    xi1, yi1 = max(a1, b1), max(a2, b2)
    xi2, yi2 = min(a3, b3), min(a4, b4)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    areaA = (a3 - a1) * (a4 - a2)
    areaB = (b3 - b1) * (b4 - b2)
    union_area = areaA + areaB - inter_area
    return inter_area / union_area if union_area > 0 else 0

def nms(boxes, scores, iou_thresh=0.5):
    """Non-Maximum Suppression for boxes."""
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        remove = [0]
        for j in range(1, len(idxs)):
            if iou(boxes[i], boxes[idxs[j]]) > iou_thresh:
                remove.append(j)
        idxs = np.delete(idxs, remove)
    return keep

def check_relation(g, p):
    """Check relational graph against predictions. Returns list of violations and stats."""
    violations = []
    stats = defaultdict(int)
    for u,v,d in g.edges(data=True):
        rel = d.get('relation')
        if rel == 'left-of':
            if p[u]['x'] > p[v]['x']:
                violations.append((u,v,'left-of'))
        elif rel == 'above':
            if p[u]['y'] > p[v]['y']:
                violations.append((u,v,'above'))
        elif rel == 'touches':
            # Simple proximity check
            if iou(p[u], p[v]) < 0.05:
                violations.append((u,v,'touches'))
        elif rel == 'contains':
            # Check if box u contains box v
            x1, y1, w1, h1 = p[u]['x'], p[u]['y'], p[u]['w'], p[u]['h']
            x2, y2, w2, h2 = p[v]['x'], p[v]['y'], p[v]['w'], p[v]['h']
            # Convert to corners
            def to_corners(x, y, w, h):
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
                return x1, y1, x2, y2
            ux1, uy1, ux2, uy2 = to_corners(x1, y1, w1, h1)
            vx1, vy1, vx2, vy2 = to_corners(x2, y2, w2, h2)
            if not (ux1 <= vx1 and uy1 <= vy1 and ux2 >= vx2 and uy2 >= vy2):
                violations.append((u,v,'contains'))
        stats[rel] += 1
    return violations, dict(stats)

def label_quality(preds, confs=None, iou_thresh=0.5):
    """Run NMS and IoU checks, return filtered preds and flagged outliers."""
    if confs is None:
        confs = [1.0]*len(preds)
    keep_idxs = nms(preds, confs, iou_thresh)
    filtered = [preds[i] for i in keep_idxs]
    outliers = []
    for i, p in enumerate(preds):
        if i not in keep_idxs:
            outliers.append(p)
    return filtered, outliers

def log_metadata(image_id, preds, violations, stats, out_path):
    """Log metadata for analysis and curriculum sampling."""
    meta = {
        'id': image_id,
        'obj_count': len(preds),
        'rel_violations': len(violations),
        'rel_stats': stats,
        'avg_area': float(np.mean([p['w']*p['h'] for p in preds]) if preds else 0),
        'violations': violations
    }
    with open(out_path, 'a') as f:
        f.write(json.dumps(meta) + '\n')

if __name__ == '__main__':
    # Example usage
    graph_path = 'rel_graphs/img_001.json'
    yolo_path = 'predictions/img_001.txt'
    image_id = os.path.splitext(os.path.basename(yolo_path))[0]
    graph = load_graph(graph_path)
    preds = load_yolo(yolo_path)
    # Simulate confidences if available
    confs = [0.9 for _ in preds] # Replace with actual confidences if available
    filtered_preds, outliers = label_quality(preds, confs)
    violations, stats = check_relation(graph, filtered_preds)
    print('Violations:', violations)
    print('Relation stats:', stats)
    print('Outlier boxes:', outliers)
    log_metadata(image_id, filtered_preds, violations, stats, 'metadata_log.jsonl')
